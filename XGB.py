"""Train and evaluate an XGBoost model from preprocessing.py chunk outputs.

This file implements the XGBoost part of the project. Run it after
preprocessing.py has completed successfully. It uses XGBoost external-memory
training so the full training split does not need to be stacked in RAM.

The parameter trials and all reported scores are generated from the data;
review them and write your own interpretation for the assignment report.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
TRAIN_DIR = DATA_DIR / "train_processed"
TEST_DIR = DATA_DIR / "test_processed"
FEATURE_NAMES_FILE = DATA_DIR / "feature_names.csv"
RESULTS_DIR = DATA_DIR / "xgb_results"

CLASS_NAMES = ["No_Accident", "Minor", "Major"]
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
RANDOM_STATE = 42
MAX_BIN = 256
TUNING_BOOST_ROUNDS = 250
FINAL_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 25
NTHREAD = 8

# Tune a small set of meaningful configurations on a subset of the training
# chunks, then refit the selected configuration on all training chunks.
PARAMETER_TRIALS = [
    {"max_depth": 6, "learning_rate": 0.08, "min_child_weight": 5, "class_weight_power": 0.0},
    {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 10, "class_weight_power": 0.5},
    {"max_depth": 8, "learning_rate": 0.08, "min_child_weight": 5, "class_weight_power": 1.0},
]


def numbered_files(folder: Path, prefix: str, suffix: str) -> list[Path]:
    """Return chunk paths ordered numerically (0, 1, 2, ... 10)."""
    files = list(folder.glob(f"{prefix}_*{suffix}"))
    return sorted(files, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def get_chunk_pairs(folder: Path) -> list[tuple[Path, Path]]:
    """Pair each sparse feature chunk with its same-numbered label chunk."""
    x_files = numbered_files(folder, "X", ".npz")
    y_files = numbered_files(folder, "y", ".csv")
    x_numbers = [int(path.stem.split("_")[1]) for path in x_files]
    y_numbers = [int(path.stem.split("_")[1]) for path in y_files]

    if not x_files or x_numbers != y_numbers:
        raise FileNotFoundError(
            f"Expected matching X_n.npz and y_n.csv files in {folder}. "
            f"Found X chunks {x_numbers[:5]}... and y chunks {y_numbers[:5]}..."
        )

    return list(zip(x_files, y_files))


def read_labels(path: Path) -> np.ndarray:
    """Read string target labels and encode them as 0, 1, and 2."""
    frame = pd.read_csv(path, dtype={"accident_severity": "string"})
    if list(frame.columns) != ["accident_severity"]:
        raise ValueError(f"Unexpected label columns in {path}: {list(frame.columns)}")

    encoded = frame["accident_severity"].map(LABEL_TO_ID)
    if encoded.isna().any():
        unknown = sorted(frame.loc[encoded.isna(), "accident_severity"].dropna().unique())
        raise ValueError(f"Unknown target labels in {path}: {unknown}")
    return encoded.to_numpy(dtype=np.int32)


def count_labels(pairs: list[tuple[Path, Path]]) -> Counter:
    """Count class labels without loading feature matrices into memory."""
    counts = Counter()
    for _, y_path in pairs:
        y = read_labels(y_path)
        counts.update(y.tolist())
    return counts


def inverse_frequency_weights(counts: Counter, power: float = 1.0) -> dict[int, float]:
    """Return tunable inverse-frequency weights; power 0 means unweighted."""
    total = sum(counts.values())
    present_classes = [label for label in range(len(CLASS_NAMES)) if counts[label] > 0]
    if len(present_classes) != len(CLASS_NAMES):
        raise ValueError(f"Training split is missing classes: {counts}")
    return {
        label: (total / (len(present_classes) * counts[label])) ** power
        for label in present_classes
    }


class SparseChunkIterator(xgb.DataIter):
    """Stream saved CSR chunks to XGBoost's external-memory matrix."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        cache_prefix: Path,
        class_weights: dict[int, float] | None = None,
    ) -> None:
        super().__init__(cache_prefix=str(cache_prefix), release_data=True)
        self.pairs = pairs
        self.class_weights = class_weights
        self.position = 0

    def next(self, input_data) -> bool:
        if self.position >= len(self.pairs):
            return False

        x_path, y_path = self.pairs[self.position]
        X = sp.load_npz(x_path).tocsr().astype(np.float32)
        y = read_labels(y_path)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Row mismatch for {x_path.name} and {y_path.name}: "
                f"{X.shape[0]} features versus {y.shape[0]} labels"
            )

        batch = {"data": X, "label": y}
        if self.class_weights is not None:
            batch["weight"] = np.asarray(
                [self.class_weights[int(label)] for label in y],
                dtype=np.float32,
            )

        input_data(**batch)
        self.position += 1
        return True

    def reset(self) -> None:
        self.position = 0


def make_external_matrix(
    pairs: list[tuple[Path, Path]],
    cache_prefix: Path,
    class_weights: dict[int, float] | None = None,
    reference=None,
):
    iterator = SparseChunkIterator(pairs, cache_prefix, class_weights)
    matrix = xgb.ExtMemQuantileDMatrix(
        iterator,
        max_bin=MAX_BIN,
        ref=reference,
        enable_categorical=False,
    )
    return iterator, matrix


def base_parameters() -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": len(CLASS_NAMES),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_bin": MAX_BIN,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "seed": RANDOM_STATE,
        "nthread": NTHREAD,
    }


def score_validation(model: xgb.Booster, pairs: list[tuple[Path, Path]]) -> dict:
    """Calculate validation scores in chunks and return the key metrics."""
    y_true_parts = []
    y_pred_parts = []

    for x_path, y_path in pairs:
        X = sp.load_npz(x_path).tocsr().astype(np.float32)
        y_true = read_labels(y_path)
        dmatrix = xgb.DMatrix(X)
        probabilities = model.predict(dmatrix)
        y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_test_set(
    model: xgb.Booster,
    pairs: list[tuple[Path, Path]],
    feature_names: list[str],
) -> tuple[dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Evaluate once on the untouched test set and produce error summaries."""
    y_true_parts = []
    y_pred_parts = []
    chunk_rows = []

    for chunk_index, (x_path, y_path) in enumerate(pairs):
        X = sp.load_npz(x_path).tocsr().astype(np.float32)
        y_true = read_labels(y_path)
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: {X.shape[1]} in {x_path.name}, "
                f"{len(feature_names)} names in {FEATURE_NAMES_FILE.name}"
            )
        dmatrix = xgb.DMatrix(X)
        probabilities = model.predict(dmatrix)
        y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)
        chunk_rows.append(
            {
                "chunk": chunk_index,
                "rows": int(y_true.size),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
            }
        )

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    metrics = {
        "test_rows": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
    }

    confusion = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    confusion.index.name = "actual"
    confusion.columns.name = "predicted"
    return metrics, matrix, confusion, pd.DataFrame(chunk_rows)


def main() -> None:
    if int(xgb.__version__.split(".")[0]) < 3:
        raise RuntimeError(
            f"XGBoost {xgb.__version__} is installed. This full-data streaming "
            "script requires XGBoost >= 3.0.0; upgrade with: "
            "python -m pip install --upgrade xgboost"
        )

    if not FEATURE_NAMES_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURE_NAMES_FILE} is missing. Run preprocessing.py successfully first."
        )

    train_pairs_all = get_chunk_pairs(TRAIN_DIR)
    test_pairs = get_chunk_pairs(TEST_DIR)

    # Reserve every tenth training chunk for validation. This keeps validation
    # separate from model fitting and leaves the test split untouched until the
    # final evaluation. The chunk split is deterministic and reproducible.
    validation_pairs = [
        pair for index, pair in enumerate(train_pairs_all) if index % 10 == 0
    ]
    train_pairs = [
        pair for index, pair in enumerate(train_pairs_all) if index % 10 != 0
    ]
    tuning_pairs = [
        pair for index, pair in enumerate(train_pairs) if index % 10 == 0
    ]

    if not train_pairs or not validation_pairs or not tuning_pairs:
        raise ValueError(
            "Not enough training chunks for the train/validation/tuning split. "
            "At least 20 training chunks are recommended."
        )

    feature_names = pd.read_csv(FEATURE_NAMES_FILE)["feature_name"].astype(str).tolist()
    first_X = sp.load_npz(train_pairs[0][0])
    if first_X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature matrix has {first_X.shape[1]} columns but feature_names.csv "
            f"has {len(feature_names)} names. Rerun preprocessing.py."
        )
    del first_X

    train_counts = count_labels(train_pairs)
    class_weights = inverse_frequency_weights(train_counts)
    total_train_rows = sum(train_counts.values())

    print(f"XGBoost version: {xgb.__version__}")
    print(f"Training chunks: {len(train_pairs)} ({total_train_rows:,} rows)")
    print(f"Tuning chunks: {len(tuning_pairs)}")
    print(f"Validation chunks: {len(validation_pairs)}")
    print(f"Test chunks: {len(test_pairs)}")
    print("Training class counts:", {CLASS_NAMES[k]: train_counts[k] for k in range(3)})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuning_results = []

    # Small parameter comparison using only a subset of training chunks.
    # It compares no weighting, partial weighting, and full inverse-frequency
    # weighting. Validation always keeps the natural class distribution.
    with tempfile.TemporaryDirectory(prefix="xgb_tuning_", dir=RESULTS_DIR) as cache_dir:
        cache = Path(cache_dir)
        for trial_number, trial in enumerate(PARAMETER_TRIALS, start=1):
            trial_cache = cache / f"trial_{trial_number}"
            trial_cache.mkdir()
            trial_weights = inverse_frequency_weights(
                train_counts, power=trial["class_weight_power"]
            )
            tune_iter, d_tune = make_external_matrix(
                tuning_pairs,
                trial_cache / "tuning",
                class_weights=trial_weights,
            )
            valid_iter, d_valid = make_external_matrix(
                validation_pairs,
                trial_cache / "validation",
                reference=d_tune,
            )

            params = base_parameters()
            params.update(
                {key: value for key, value in trial.items() if key != "class_weight_power"}
            )
            print(f"\nParameter trial {trial_number}/{len(PARAMETER_TRIALS)}: {trial}")
            model = xgb.train(
                params=params,
                dtrain=d_tune,
                num_boost_round=TUNING_BOOST_ROUNDS,
                evals=[(d_valid, "validation")],
                callbacks=[
                    xgb.callback.EarlyStopping(
                        rounds=EARLY_STOPPING_ROUNDS,
                        metric_name="mlogloss",
                        data_name="validation",
                        save_best=True,
                    )
                ],
                verbose_eval=25,
            )
            scores = score_validation(model, validation_pairs)
            tuning_results.append(
                {
                    "trial": trial_number,
                    **trial,
                    **scores,
                    "best_iteration": int(model.best_iteration),
                }
            )
            print("Validation scores:", scores)
            del d_tune, d_valid, tune_iter, valid_iter, model

        tuning_frame = pd.DataFrame(tuning_results).sort_values(
            ["macro_f1", "balanced_accuracy"], ascending=False
        )
        tuning_frame.to_csv(RESULTS_DIR / "parameter_trials.csv", index=False)
        best_trial_number = int(tuning_frame.iloc[0]["trial"])
        best_trial = PARAMETER_TRIALS[best_trial_number - 1]
        best_rounds = max(1, int(tuning_frame.iloc[0]["best_iteration"]) + 1)
        class_weights = inverse_frequency_weights(
            train_counts, power=best_trial["class_weight_power"]
        )

    print(f"\nSelected parameters: {best_trial}")
    print(f"Selected boosting rounds from tuning: {best_rounds}")

    # Refit on all training chunks. Validation chunks are still kept out of
    # fitting and are used only for early stopping.
    with tempfile.TemporaryDirectory(prefix="xgb_final_", dir=RESULTS_DIR) as cache_dir:
        cache = Path(cache_dir)
        final_train_iter, d_train = make_external_matrix(
            train_pairs,
            cache / "train",
            class_weights=class_weights,
        )
        final_valid_iter, d_valid = make_external_matrix(
            validation_pairs,
            cache / "validation",
            reference=d_train,
        )

        final_params = base_parameters()
        final_params.update(
            {key: value for key, value in best_trial.items() if key != "class_weight_power"}
        )
        model = xgb.train(
            params=final_params,
            dtrain=d_train,
            num_boost_round=max(FINAL_BOOST_ROUNDS, best_rounds),
            evals=[(d_valid, "validation")],
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=EARLY_STOPPING_ROUNDS,
                    metric_name="mlogloss",
                    data_name="validation",
                    save_best=True,
                )
            ],
            verbose_eval=25,
        )
        del d_train, d_valid, final_train_iter, final_valid_iter

    model_path = RESULTS_DIR / "xgb_accident_severity.json"
    model.save_model(model_path)

    metrics, matrix, confusion, chunk_stability = evaluate_test_set(
        model, test_pairs, feature_names
    )
    metrics["best_iteration"] = int(model.best_iteration)
    metrics["selected_parameters"] = best_trial
    metrics["class_weights"] = {
        CLASS_NAMES[label]: float(weight) for label, weight in class_weights.items()
    }
    metrics["test_class_counts"] = {
        CLASS_NAMES[label]: int(count)
        for label, count in count_labels(test_pairs).items()
    }

    with (RESULTS_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    class_report = metrics["classification_report"]
    report_rows = []
    for class_name in CLASS_NAMES + ["macro avg", "weighted avg"]:
        values = class_report[class_name]
        report_rows.append(
            {
                "class": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1-score": values["f1-score"],
                "support": values["support"],
            }
        )
    pd.DataFrame(report_rows).to_csv(
        RESULTS_DIR / "classification_report.csv", index=False
    )
    confusion.to_csv(RESULTS_DIR / "confusion_matrix.csv")
    chunk_stability.to_csv(RESULTS_DIR / "test_chunk_stability.csv", index=False)

    # Count errors by true/predicted class pair for the report's error analysis.
    error_rows = []
    for actual_id in range(len(CLASS_NAMES)):
        for predicted_id in range(len(CLASS_NAMES)):
            if actual_id != predicted_id:
                error_rows.append(
                    {
                        "actual": CLASS_NAMES[actual_id],
                        "predicted": CLASS_NAMES[predicted_id],
                        "count": int(matrix[actual_id, predicted_id]),
                    }
                )
    pd.DataFrame(error_rows).sort_values("count", ascending=False).to_csv(
        RESULTS_DIR / "error_summary.csv", index=False
    )

    # Map XGBoost's f0, f1, ... names back to preprocessing.py's feature names.
    importance = model.get_score(importance_type="gain")
    importance_rows = [
        {
            "feature": name,
            "gain": float(importance.get(f"f{index}", 0.0)),
        }
        for index, name in enumerate(feature_names)
    ]
    pd.DataFrame(importance_rows).sort_values("gain", ascending=False).to_csv(
        RESULTS_DIR / "feature_importance_gain.csv", index=False
    )

    print("\nFinal test metrics:")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
    print("\nPer-class report:")
    print(pd.DataFrame(report_rows).round(4).to_string(index=False))
    print(f"\nModel and evaluation files saved in: {RESULTS_DIR}")
    print("These are measured results; explain their meaning in your own words.")


if __name__ == "__main__":
    main()
