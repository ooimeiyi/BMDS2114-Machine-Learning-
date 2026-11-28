"""Train, tune and evaluate XGBoost using the shared preprocessing split.

Run preprocessing.py first. The training flow follows RandomForest.py: tune on a
stratified training sample, refit selected stages on a larger sample, and score
the untouched test split. Outputs are written in the format consumed by UI.py.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from xgboost import XGBClassifier

from preprocessing import TARGET_NAMES, load_split

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results" / "xgboost_full"
RANDOM_STATE = 42
TUNE_ROWS = 100_000
FINAL_ROWS = 500_000
TEST_BATCH_ROWS = 100_000
CV_FOLDS = 3
N_ITER = 8
FINE_ITER = 6
LABELS = [0, 1, 2]
NAMES = [TARGET_NAMES[i] for i in LABELS]
SCORING = {"accuracy": "accuracy", "precision_macro": "precision_macro",
           "recall_macro": "recall_macro", "f1_macro": "f1_macro"}
BASELINE_PARAMS = {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.08,
                   "min_child_weight": 5, "subsample": 0.8,
                   "colsample_bytree": 0.8, "reg_lambda": 2.0}
PARAM_DISTRIBUTIONS = {
    "n_estimators": [150, 250, 350],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 5, 10, 20],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_lambda": [1.0, 2.0, 5.0],
}


def classifier(params):
    return XGBClassifier(objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", tree_method="hist",
                         random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
                         **params)


def stratified_rows(y, n):
    if n is None or n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y,
                              random_state=RANDOM_STATE)
    return np.sort(idx)


def test_metrics(model, X, y):
    probs = model.predict_proba(X)
    pred = model.classes_[probs.argmax(axis=1)]
    return ({"accuracy": accuracy_score(y, pred),
             "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
             "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
             "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
             "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
             "roc_auc_macro": roc_auc_score(y, probs, multi_class="ovr", average="macro")},
            pred, probs)


def evaluate(name, params, X_fit, y_fit, X_test, y_test, check_idx):
    start = time.time()
    model = classifier(params).fit(X_fit, y_fit)
    metrics, pred, probs = test_metrics(model, X_test, y_test)
    train_f1 = f1_score(y_fit[check_idx], model.predict(X_fit.iloc[check_idx]),
                        average="macro", zero_division=0)
    return {"name": name, "params": dict(params), "model": model,
            "metrics": metrics, "pred": pred, "proba": probs,
            "train_f1": train_f1, "fit_s": time.time() - start}


def search_scores(search):
    i = search.best_index_
    return {k: float(search.cv_results_[f"mean_test_{k}"][i]) for k in SCORING}


def narrow_space(params):
    return {
        "n_estimators": sorted({max(100, params["n_estimators"] - 100), params["n_estimators"], params["n_estimators"] + 100}),
        "max_depth": sorted({max(2, params["max_depth"] - 2), params["max_depth"], params["max_depth"] + 2}),
        "learning_rate": sorted({max(0.01, params["learning_rate"] * 0.7), params["learning_rate"], min(0.3, params["learning_rate"] * 1.3)}),
        "min_child_weight": sorted({max(1, params["min_child_weight"] // 2), params["min_child_weight"], params["min_child_weight"] * 2}),
        "subsample": sorted({max(0.5, params["subsample"] - 0.1), params["subsample"], min(1.0, params["subsample"] + 0.1)}),
        "colsample_bytree": sorted({max(0.5, params["colsample_bytree"] - 0.1), params["colsample_bytree"], min(1.0, params["colsample_bytree"] + 0.1)}),
        "reg_lambda": sorted({max(0.1, params["reg_lambda"] / 2), params["reg_lambda"], params["reg_lambda"] * 2}),
    }


def result_row(name, metrics):
    return {"Model": name, "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision_macro"], "Recall": metrics["recall_macro"],
            "F1": metrics["f1_macro"], "F1 weighted": metrics["f1_weighted"],
            "ROC AUC": metrics["roc_auc_macro"]}


def per_class_table(y, pred):
    report = classification_report(y, pred, labels=LABELS, target_names=NAMES,
                                   output_dict=True, zero_division=0)
    frame = pd.DataFrame(report).T.loc[NAMES, ["precision", "recall", "f1-score", "support"]]
    frame["support"] = frame["support"].astype(int)
    return frame.rename(columns={"precision": "Precision", "recall": "Recall",
                                 "f1-score": "F1", "support": "Rows"})


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    X_train, X_test, y_train, y_test = load_split()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tune_idx = stratified_rows(y_train, TUNE_ROWS)
    final_idx = stratified_rows(y_train, FINAL_ROWS)
    X_tune, y_tune = X_train.iloc[tune_idx].reset_index(drop=True), y_train[tune_idx]
    X_final, y_final = X_train.iloc[final_idx].reset_index(drop=True), y_train[final_idx]
    check_idx = stratified_rows(y_final, min(100_000, len(y_final)))
    print(f"Train rows: {len(y_train):,}; tuning sample: {len(X_tune):,}; final fit: {len(X_final):,}; test: {len(X_test):,}")

    cvres = cross_validate(classifier(BASELINE_PARAMS), X_tune, y_tune, cv=cv,
                           n_jobs=1, scoring=SCORING, return_train_score=True)
    baseline_cv = {k: float(cvres[f"test_{k}"].mean()) for k in SCORING}
    baseline = evaluate("Baseline", BASELINE_PARAMS, X_final, y_final,
                        X_test, y_test, check_idx)
    baseline.pop("model")
    baseline.pop("proba")
    print("Baseline CV:", baseline_cv)

    search = RandomizedSearchCV(classifier({}), PARAM_DISTRIBUTIONS,
                                n_iter=N_ITER, cv=cv, scoring=SCORING,
                                refit="f1_macro", random_state=RANDOM_STATE,
                                n_jobs=1, verbose=1)
    search.fit(X_tune, y_tune)
    coarse_params, coarse_cv = dict(search.best_params_), search_scores(search)
    pd.DataFrame(search.cv_results_).to_csv(RESULT_DIR / "xgb_search.csv", index=False)
    coarse = evaluate("Tuned (random search)", coarse_params, X_final, y_final,
                      X_test, y_test, check_idx)
    coarse.pop("proba")
    del search

    fine_search = RandomizedSearchCV(classifier({}), narrow_space(coarse_params),
                                     n_iter=FINE_ITER, cv=cv, scoring=SCORING,
                                     refit="f1_macro", random_state=RANDOM_STATE,
                                     n_jobs=1, verbose=1)
    fine_search.fit(X_tune, y_tune)
    fine_cv_found = search_scores(fine_search)
    pd.DataFrame(fine_search.cv_results_).to_csv(RESULT_DIR / "xgb_fine_search.csv", index=False)
    if fine_cv_found["f1_macro"] > coarse_cv["f1_macro"]:
        fine_params = dict(fine_search.best_params_)
        coarse.pop("model")
        fine = evaluate("Fine-tuned", fine_params, X_final, y_final,
                        X_test, y_test, check_idx)
        fine_cv = fine_cv_found
    else:
        fine, fine_cv = dict(coarse, name="Fine-tuned"), coarse_cv
    del fine_search

    stages = [baseline, coarse, fine]
    cvs = [baseline_cv, coarse_cv, fine_cv]
    table = pd.DataFrame([result_row(r["name"], r["metrics"]) for r in stages])
    per_class = pd.concat({r["name"]: per_class_table(y_test, r["pred"]) for r in stages})
    per_class.round(4).to_csv(RESULT_DIR / "xgb_per_class.csv")
    summary = table.copy()
    summary["CV accuracy"] = [c["accuracy"] for c in cvs]
    summary["CV F1"] = [c["f1_macro"] for c in cvs]
    summary["train F1"] = [r["train_f1"] for r in stages]
    summary["fit seconds"] = [r["fit_s"] for r in stages]
    summary.round(4).to_csv(RESULT_DIR / "xgb_results_summary.csv", index=False)

    best = fine
    report = classification_report(y_test, best["pred"], labels=LABELS,
                                   target_names=NAMES, digits=4, zero_division=0)
    (RESULT_DIR / "xgb_report.txt").write_text(
        "XGBoost model evaluation\n\n" + table.round(4).to_string(index=False) +
        "\n\nFinal model parameters\n" + repr(best["params"]) +
        "\n\nClassification report\n" + report +
        f"\nTotal run time: {(time.time() - started) / 60:.1f} min\n", encoding="utf-8")
    cm = confusion_matrix(y_test, best["pred"], labels=LABELS)
    pd.DataFrame(cm, index=NAMES, columns=NAMES).to_csv(RESULT_DIR / "xgb_confusion.csv")
    joblib.dump(best["model"], RESULT_DIR / "xgb_fine_tuned.joblib", compress=3)
    pd.Series(best["model"].feature_importances_, index=X_train.columns,
              name="importance").sort_values(ascending=False).to_csv(RESULT_DIR / "xgb_feature_importance.csv")
    print("\nTest set results:")
    print(table.round(4).to_string(index=False))
    print("\nFinal model classification report:\n", report)
    print(f"\nSaved UI model and results to {RESULT_DIR}")


if __name__ == "__main__":
    main()
