"""
Random Forest training for BDRoadRisk accident severity (3 classes), for the data
produced by preprocessing_full.py (already imputed, encoded and scaled).

Run preprocessing_full.py first and keep this file in the same folder.

Why this script samples rows
----------------------------
A Random Forest on millions of rows is too slow for cross-validated tuning, so:
  * Tuning (baseline CV, randomized search, fine-tuning) uses a stratified sample
    of TUNE_ROWS training rows.
  * Every model is then fitted on a larger stratified sample of FINAL_ROWS rows.
  * ALL models are evaluated on the WHOLE test set (never sampled), and the test
    set is never used to choose parameters (only the CV score is).

Steps
-----
  1. Baseline Random Forest (fixed reasonable settings).
  2. Randomized search over a wide space (coarse tuning, macro F1).
  3. Fine-tuning: a second, narrow randomized search around the best settings
     of step 2 (neighbouring values of every parameter). It is used only if its
     CV macro F1 is better; otherwise the step-2 settings are kept.
  4. Results table on the test set: accuracy, precision, recall, F1 (overall and
     per class), overfitting check, confusion matrix, comparison plot.
  5. Feature importance (grouped per original feature) and error analysis.
  6. Save the final model, tables, plots and a text report to
     results/random_forest_full/.
"""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split

from preprocessing import NOMINAL, ORDINAL, TARGET_NAMES, load_split

# ----------------------------------------------------------------------------
# Settings (edit these)
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results" / "random_forest_full"
RANDOM_STATE = 42
TUNE_ROWS = 100_000                          # training rows for cross-validation and the searches
FINAL_ROWS = 500_000                         # training rows to fit the final models (None = all; needs a lot of RAM)
IMPORTANCE_ROWS = 100_000                    # test rows for permutation importance
CV_FOLDS = 3
N_ITER = 15                                  # coarse search: random parameter sets to try
FINE_ITER = 12                               # fine-tuning: random parameter sets to try (grid has 81+ combinations)
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = [TARGET_NAMES[i] for i in CLASS_LABELS]
ERROR_GROUP_COLUMNS = ["weather_condition", "light_condition", "road_surface",
                       "time_of_day", "vehicle_type", "road_class", "road_quality"]

# Depth and leaf size are limited because a fully grown forest on hundreds of
# thousands of rows needs gigabytes of memory.
BASELINE_PARAMS = dict(n_estimators=100, max_depth=16, min_samples_leaf=10)

PARAM_DISTRIBUTIONS = {                      # coarse search space
    "n_estimators": randint(100, 400),
    "max_depth": [8, 12, 16, 20, 24],
    "min_samples_split": randint(10, 60),
    "min_samples_leaf": randint(5, 60),
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "class_weight": [None, "balanced", "balanced_subsample"],   # classes are imbalanced
}
SCORING = {"accuracy": "accuracy", "precision_macro": "precision_macro",
           "recall_macro": "recall_macro", "f1_macro": "f1_macro"}

REPORT = []


def log(text=""):
    print(text)
    REPORT.append(str(text))


def heading(title, char="="):
    log("\n" + char * 78)
    log(title)
    log(char * 78)


def stratified_rows(y, n, seed=RANDOM_STATE):
    """Sorted row positions of a stratified random subsample (all rows if n is None or too large)."""
    y = np.asarray(y)
    if n is None or n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def class_counts(y):
    c = pd.Series(np.asarray(y)).value_counts().sort_index().rename(index=TARGET_NAMES)
    return ", ".join(f"{k} {v:,} ({100 * v / c.sum():.1f}%)" for k, v in c.items())


def fmt_params(d):
    return ", ".join(f"{k}={v}" for k, v in d.items())


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def test_metrics(model, X_test, y_test):
    """Accuracy, macro precision / recall / F1, weighted F1 and macro ROC AUC on the test set."""
    proba = model.predict_proba(X_test)
    pred = model.classes_[proba.argmax(axis=1)]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "f1_weighted": f1_score(y_test, pred, average="weighted"),
        "roc_auc_macro": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
    }
    return metrics, pred, proba


def evaluate(name, params, X_fit, y_fit, X_test, y_test, check_idx):
    """Fit one forest on the final sample and evaluate it on the whole test set."""
    t0 = time.time()
    model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1).fit(X_fit, y_fit)
    fit_s = time.time() - t0
    metrics, pred, proba = test_metrics(model, X_test, y_test)
    train_f1 = f1_score(y_fit[check_idx], model.predict(X_fit.iloc[check_idx]), average="macro")
    return {"name": name, "params": dict(params), "model": model, "metrics": metrics,
            "pred": pred, "proba": proba, "train_f1": train_f1, "fit_s": fit_s}


def search_cv(search):
    """Mean CV scores of the best parameter set of a finished search."""
    i = search.best_index_
    return {k: float(search.cv_results_[f"mean_test_{k}"][i]) for k in SCORING}


def top_table(search, n=5):
    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_f1_macro")
    top = pd.DataFrame({
        "rank": res["rank_test_f1_macro"], "cv_f1": res["mean_test_f1_macro"],
        "cv_f1_std": res["std_test_f1_macro"], "cv_acc": res["mean_test_accuracy"],
        "cv_precision": res["mean_test_precision_macro"], "cv_recall": res["mean_test_recall_macro"],
        "params": res["params"].apply(fmt_params)})
    return res, top.head(n)


def fine_grid(p):
    """Narrow search space: the best value of each parameter and its neighbours."""
    n, d = int(p["n_estimators"]), int(p["max_depth"])
    s, l = int(p["min_samples_split"]), int(p["min_samples_leaf"])
    mf = p["max_features"]
    return {
        "n_estimators": sorted({max(50, n - 100), n, n + 100}),
        "max_depth": sorted({max(4, d - 4), d, d + 4}),
        "min_samples_split": sorted({max(2, s - 5), s, s + 5}),
        "min_samples_leaf": sorted({max(1, l - 3), l, l + 3}),
        "max_features": [mf] if isinstance(mf, str) else sorted({max(0.1, round(mf - 0.1, 2)), mf, min(1.0, round(mf + 0.1, 2))}),
        "class_weight": [p["class_weight"]],
    }


def results_table(results):
    """One row per model: the four headline metrics (+ weighted F1 and ROC AUC)."""
    return pd.DataFrame([{
        "Model": r["name"], "Accuracy": r["metrics"]["accuracy"],
        "Precision": r["metrics"]["precision_macro"], "Recall": r["metrics"]["recall_macro"],
        "F1": r["metrics"]["f1_macro"], "F1 weighted": r["metrics"]["f1_weighted"],
        "ROC AUC": r["metrics"]["roc_auc_macro"]} for r in results])


def per_class_table(y_true, pred):
    rep = classification_report(y_true, pred, labels=CLASS_LABELS, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.loc[CLASS_NAMES, ["precision", "recall", "f1-score", "support"]]
    df["support"] = df["support"].astype(int)
    return df.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Rows"})


def feature_groups(columns):
    """Encoded column -> original feature (one-hot columns and month sin/cos are grouped)."""
    groups = {}
    for c in columns:
        base = next((k for k in NOMINAL if c.startswith(k + "_")), None)
        if base is None and c.endswith(("_sin", "_cos")):
            base = c.rsplit("_", 1)[0]
        groups.setdefault(base or c, []).append(c)
    return groups


def grouped_permutation_importance(model, X, y, n_repeats=3, seed=RANDOM_STATE):
    """Drop in macro F1 when ALL encoded columns of one original feature are shuffled together."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    base = f1_score(y, model.predict(X), average="macro")
    out = {}
    for name, cols in feature_groups(X.columns).items():
        original = X[cols].to_numpy(copy=True)
        drops = []
        for _ in range(n_repeats):
            X[cols] = original[rng.permutation(len(X))]
            drops.append(base - f1_score(y, model.predict(X), average="macro"))
        X[cols] = original
        out[name] = float(np.mean(drops))
    return pd.Series(out)


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_comparison(table, path):
    plt = _plt()
    if plt is None:
        return
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x, w = np.arange(len(metrics)), 0.8 / len(table)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for k, (_, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + (k - (len(table) - 1) / 2) * w, row[metrics].astype(float), w, label=row["Model"])
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.08)
    ax.set_title("Test set: precision, recall and F1 are macro averages")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, title, path):
    plt = _plt()
    if plt is None:
        return
    norm = cm / cm.sum(axis=1, keepdims=True)           # row-normalised: recall per class
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(series, title, xlabel, path, top=15):
    plt = _plt()
    if plt is None:
        return
    s = series.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.5))
    ax.barh(s.index, s.values, color="#3b6ea5")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_analysis(pred, proba, X_test, y_test):
    """Where does the model go wrong? (log only)"""
    y = np.asarray(y_test)
    wrong = pred != y
    log(f"Misclassified test rows: {wrong.sum():,} of {len(y):,} ({wrong.mean():.1%})")
    pairs = pd.Series(list(zip(y[wrong], pred[wrong]))).value_counts()
    log("Most common mistakes (actual -> predicted):")
    for (a, p), n in pairs.items():
        log(f"  {TARGET_NAMES[a]:>11} -> {TARGET_NAMES[p]:<11} {n:9,d}  ({n / wrong.sum():.0%} of all errors)")
    conf = proba.max(axis=1)
    log(f"Mean confidence when correct: {conf[~wrong].mean():.3f} | when wrong: {conf[wrong].mean():.3f}")

    rows = []
    for col in ERROR_GROUP_COLUMNS:
        if col in NOMINAL:
            for level in NOMINAL[col]:
                name = f"{col}_{level}"
                if name in X_test.columns:
                    rows.append((col, level, X_test[name].to_numpy() == 1))
        elif col in ORDINAL and col in X_test.columns:
            codes = X_test[col].to_numpy().round().astype(int)
            for k, level in enumerate(ORDINAL[col]):
                rows.append((col, level, codes == k))
    table = pd.DataFrame([{"feature": f, "level": lv, "rows": int(m.sum()), "error_rate": wrong[m].mean()}
                          for f, lv, m in rows if m.sum() >= 50])
    if len(table):
        log(f"Overall error rate: {wrong.mean():.3f}. Feature levels with the highest error rate:")
        log(table.sort_values("error_rate", ascending=False).head(6).round(3).to_string(index=False))


def forest_info(rf):
    depths = [t.get_depth() for t in rf.estimators_]
    nodes = [t.tree_.node_count for t in rf.estimators_]
    return {"trees": len(rf.estimators_), "depth_mean": float(np.mean(depths)),
            "depth_max": int(np.max(depths)), "nodes_mean": float(np.mean(nodes)),
            "nodes_total": int(np.sum(nodes))}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    X_train, X_test, y_train, y_test = load_split()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    tune_i = stratified_rows(y_train, TUNE_ROWS)
    final_i = stratified_rows(y_train, FINAL_ROWS)
    X_tune, y_tune = X_train.iloc[tune_i].reset_index(drop=True), y_train[tune_i]
    X_final, y_final = X_train.iloc[final_i].reset_index(drop=True), y_train[final_i]
    check_i = stratified_rows(y_final, 100_000)         # train rows for the overfitting check

    heading("1. DATA AND SETTINGS")
    n_all = len(X_train) + len(X_test)
    log(f"Total rows       : {n_all:,} (train {len(X_train):,} = {100 * len(X_train) / n_all:.0f}%, "
        f"test {len(X_test):,} = {100 * len(X_test) / n_all:.0f}%)")
    log(f"Input columns    : {X_train.shape[1]} encoded columns from {len(feature_groups(X_train.columns))} original features")
    log("Original features: " + ", ".join(feature_groups(X_train.columns)))
    log("Train classes    : " + class_counts(y_train))
    log("Test classes     : " + class_counts(y_test))
    log(f"Missing cells    : train {int(X_train.isna().sum().sum())}, test {int(X_test.isna().sum().sum())}")
    log(f"Tuning sample    : {len(X_tune):,} stratified rows ({CV_FOLDS}-fold CV; coarse search {N_ITER} sets, fine-tuning {FINE_ITER} sets)")
    log(f"Final-fit sample : {len(X_final):,} stratified rows (all models)")
    log(f"Evaluation       : the whole test set ({len(X_test):,} rows)")
    log("Baseline settings: " + fmt_params(BASELINE_PARAMS))

    # ---- 2. Baseline --------------------------------------------------------------
    heading("2. BASELINE RANDOM FOREST")
    cvres = cross_validate(RandomForestClassifier(**BASELINE_PARAMS, random_state=RANDOM_STATE, n_jobs=-1),
                           X_tune, y_tune, cv=cv, n_jobs=1, scoring=SCORING, return_train_score=True)
    base_cv = {k: float(cvres[f"test_{k}"].mean()) for k in SCORING}
    log("CV on the tuning sample: " + ", ".join(f"{k} {v:.4f}" for k, v in base_cv.items()))
    base = evaluate("Baseline", BASELINE_PARAMS, X_final, y_final, X_test, y_test, check_i)
    log(f"Fitted on {len(X_final):,} rows in {base['fit_s']:.0f} s")
    base.pop("model"); base.pop("proba")

    # ---- 3. Coarse search ------------------------------------------------------------
    heading("3. RANDOMIZED SEARCH (coarse tuning, tuning sample)")
    t0 = time.time()
    search = RandomizedSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                                PARAM_DISTRIBUTIONS, n_iter=N_ITER, cv=cv, scoring=SCORING,
                                refit="f1_macro", random_state=RANDOM_STATE, n_jobs=1)
    search.fit(X_tune, y_tune)
    tuned_cv = search_cv(search)
    best_params = dict(search.best_params_)
    res, top = top_table(search)
    res.to_csv(RESULT_DIR / "rf_search.csv", index=False)
    log("Best parameters: " + fmt_params(best_params))
    log(f"CV macro F1: {tuned_cv['f1_macro']:.4f} (baseline {base_cv['f1_macro']:.4f}, "
        f"change {tuned_cv['f1_macro'] - base_cv['f1_macro']:+.4f}) | search time {(time.time() - t0) / 60:.1f} min")
    log(f"Top {len(top)} parameter sets (all saved to rf_search.csv):")
    log(top.round(4).to_string(index=False))
    del search

    tuned = evaluate("Tuned (random search)", best_params, X_final, y_final, X_test, y_test, check_i)
    log(f"\nTuned model fitted in {tuned['fit_s']:.0f} s | test F1 {tuned['metrics']['f1_macro']:.4f}")

    # ---- 4. Fine-tuning -----------------------------------------------------------------
    heading("4. FINE-TUNING (narrow search around the best parameters)")
    grid = fine_grid(best_params)
    log("Fine search space: " + "; ".join(f"{k} {v}" for k, v in grid.items()))
    t0 = time.time()
    fsearch = RandomizedSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                                 grid, n_iter=FINE_ITER, cv=cv, scoring=SCORING,
                                 refit="f1_macro", random_state=RANDOM_STATE, n_jobs=1)
    fsearch.fit(X_tune, y_tune)
    fine_cv_found = search_cv(fsearch)
    res, top = top_table(fsearch)
    res.to_csv(RESULT_DIR / "rf_fine_search.csv", index=False)
    log(f"Top {len(top)} parameter sets (all saved to rf_fine_search.csv):")
    log(top.round(4).to_string(index=False))
    log(f"Search time: {(time.time() - t0) / 60:.1f} min")
    improved = fine_cv_found["f1_macro"] > tuned_cv["f1_macro"]
    log(f"CV macro F1: coarse best {tuned_cv['f1_macro']:.4f} -> fine best {fine_cv_found['f1_macro']:.4f} "
        f"({fine_cv_found['f1_macro'] - tuned_cv['f1_macro']:+.4f})")
    if improved:
        fine_params, fine_cv = dict(fsearch.best_params_), fine_cv_found
        tuned.pop("model"); tuned.pop("proba")          # free memory before fitting the next forest
        fine = evaluate("Fine-tuned", fine_params, X_final, y_final, X_test, y_test, check_i)
        log("Fine-tuning found better settings: " + fmt_params(fine_params))
    else:
        fine_cv = tuned_cv
        fine = dict(tuned, name="Fine-tuned")
        tuned.pop("model"); tuned.pop("proba")
        log("Fine-tuning found nothing better by CV; the coarse-search settings are kept.")
    del fsearch

    # ---- 5. Results ------------------------------------------------------------------------
    heading("5. RESULTS ON THE TEST SET")
    results, cvs = [base, tuned, fine], [base_cv, tuned_cv, fine_cv]
    table = results_table(results)
    log("Precision, recall and F1 are macro averages over the 3 classes (every class counts equally).\n")
    log(table.round(4).to_string(index=False))
    d = table.set_index("Model")
    log("\nFine-tuned minus baseline: " + ", ".join(
        f"{m} {d.loc['Fine-tuned', m] - d.loc['Baseline', m]:+.4f}" for m in ["Accuracy", "Precision", "Recall", "F1"]))

    log("\nPer class (precision, recall, F1 for each severity class):")
    pc = pd.concat({r["name"]: per_class_table(y_test, r["pred"]) for r in results})
    log(pc.round(4).to_string())
    pc.round(4).to_csv(RESULT_DIR / "rf_per_class.csv")

    log("\nOverfitting check (macro F1; a large train-to-test gap means overfitting):")
    gap = pd.DataFrame({"train F1": [r["train_f1"] for r in results],
                        "CV F1": [c["f1_macro"] for c in cvs],
                        "test F1": [r["metrics"]["f1_macro"] for r in results]},
                       index=[r["name"] for r in results])
    gap["train-test gap"] = gap["train F1"] - gap["test F1"]
    log(gap.round(4).to_string())

    best_r = fine
    log(f"\nFinal model: {best_r['name']} ({fmt_params(best_r['params'])})")
    log("Classification report (final model):")
    log(classification_report(y_test, best_r["pred"], labels=CLASS_LABELS, target_names=CLASS_NAMES, digits=3))
    cm = confusion_matrix(y_test, best_r["pred"], labels=CLASS_LABELS)
    log("Confusion matrix (rows = actual, columns = predicted):")
    log(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    plot_confusion(cm, "Random Forest (final) - test set", RESULT_DIR / "confusion.png")
    plot_comparison(table, RESULT_DIR / "metrics_comparison.png")

    summary = table.copy()
    summary["CV accuracy"] = [c["accuracy"] for c in cvs]
    summary["CV F1"] = [c["f1_macro"] for c in cvs]
    summary["train F1"] = [r["train_f1"] for r in results]
    summary["fit seconds"] = [r["fit_s"] for r in results]
    summary.round(4).to_csv(RESULT_DIR / "rf_results_summary.csv", index=False)

    # ---- 6. Feature importance -----------------------------------------------------------------
    heading("6. FEATURE IMPORTANCE (final model)")
    model = best_r["model"]
    imp_i = stratified_rows(y_test, IMPORTANCE_ROWS)
    perm_s = grouped_permutation_importance(model, X_test.iloc[imp_i].reset_index(drop=True), y_test[imp_i])
    log(f"Permutation importance on {len(imp_i):,} test rows, per ORIGINAL feature (drop in macro F1 when shuffled):")
    log(perm_s.sort_values(ascending=False).round(4).to_string())
    plot_importance(perm_s, "Permutation importance", "Drop in macro F1", RESULT_DIR / "perm_importance.png")
    plot_importance(pd.Series(model.feature_importances_, index=X_train.columns),
                    "Impurity importance (encoded columns)", "Mean decrease in impurity",
                    RESULT_DIR / "impurity_importance.png")

    # ---- 7. Error analysis -------------------------------------------------------------------------
    heading("7. ERROR ANALYSIS (final model)")
    error_analysis(best_r["pred"], best_r["proba"], X_test, y_test)

    # ---- 8. Save -------------------------------------------------------------------------------------
    heading("8. FINAL MODEL")
    joblib.dump(model, RESULT_DIR / "rf_fine_tuned.joblib", compress=3)
    info = forest_info(model)
    size_mb = (RESULT_DIR / "rf_fine_tuned.joblib").stat().st_size / 1e6
    log(f"Trees: {info['trees']} | depth mean {info['depth_mean']:.1f}, max {info['depth_max']} | "
        f"nodes per tree mean {info['nodes_mean']:.0f}, total {info['nodes_total']:,}")
    log(f"Saved model: rf_fine_tuned.joblib ({size_mb:.1f} MB) | total run time: {(time.time() - t_start) / 60:.1f} min")
    (RESULT_DIR / "rf_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nResults saved to: {RESULT_DIR}")