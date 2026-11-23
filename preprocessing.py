
# ============================================================
# BDRoadRisk - FULL DATASET PREPROCESSING
# Memory-efficient version for 16GB RAM
# ============================================================

import os
import gc
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import csr_matrix, hstack, save_npz


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "traffic_data.csv"
FINAL_OUTPUT_DIR = BASE_DIR / "processed_data"
# Build a complete new output tree first. The previous results stay in place
# unless every preprocessing step finishes successfully.
OUTPUT_DIR = BASE_DIR / ".processed_data_staging"

CHUNK_SIZE = 100000
RANDOM_STATE = 42
TEST_SIZE = 0.20
MEDIAN_SAMPLE_SIZE = 5000

# Clear only our dedicated staging directory; never start by deleting the
# currently published results.
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

TARGET = "accident_severity"

NUMERICAL = [
    "latitude",
    "longitude",
    "speed_kmh",
    "acceleration_ms2",
    "visibility_km",
    "temperature_celsius",
    "traffic_density",
    "vehicle_age_years",
    "driver_age",
    "reaction_time_sec",
    "speed_limit_zone",
    "month"
]

CATEGORICAL = [
    "weather_condition",
    "road_surface",
    "junction_type",
    "light_condition",
    "time_of_day",
    "vehicle_type",
    "driver_experience_level",
    "road_quality",
    "alcohol_influence",
    "seatbelt_worn",
    "road_class"
]

FEATURES = NUMERICAL + CATEGORICAL

VALID_LABELS = [
    "No_Accident",
    "Minor",
    "Major"
]

MISSING_CATEGORY = "__MISSING__"


# ============================================================
# 2. FIRST PASS - COUNT TARGET CLASSES
# ============================================================

print("\nStep 1: Counting target classes...")

target_counts = Counter()

for chunk in pd.read_csv(
    FILE_PATH,
    usecols=[TARGET],
    chunksize=CHUNK_SIZE
):

    target_counts.update(
        chunk[TARGET]
        .dropna()
        .value_counts()
        .to_dict()
    )

print("\nOriginal Target Distribution:")
print(target_counts)

if not any(target_counts[label] for label in VALID_LABELS):
    raise ValueError(f"No rows found with valid target labels in {FILE_PATH}")


# ============================================================
# 3. SECOND PASS - STRATIFIED TRAIN-TEST SPLIT
# ============================================================

print("\nStep 2: Splitting full dataset...")

train_file = os.path.join(
    OUTPUT_DIR, "train_raw.csv"
)

test_file = os.path.join(
    OUTPUT_DIR, "test_raw.csv"
)

# Remove previous raw files
for path in [train_file, test_file]:

    if os.path.exists(path):
        os.remove(path)

rng = np.random.default_rng(RANDOM_STATE)

first_train = True
first_test = True

train_counts = Counter()
test_counts = Counter()

for chunk in pd.read_csv(
    FILE_PATH,
    chunksize=CHUNK_SIZE
):

    chunk.columns = chunk.columns.str.strip()

    # Keep valid target labels
    chunk = chunk[
        chunk[TARGET].isin(VALID_LABELS)
    ].copy()

    if chunk.empty:
        continue

    # Random split within each target class
    random_values = rng.random(len(chunk))

    train_mask = np.zeros(
        len(chunk),
        dtype=bool
    )

    for label in VALID_LABELS:

        label_mask = (
            chunk[TARGET]
            .eq(label)
            .to_numpy()
        )

        train_mask[label_mask] = (
            random_values[label_mask] >= TEST_SIZE
        )

    train_chunk = chunk.loc[train_mask]
    test_chunk = chunk.loc[~train_mask]

    # Save training data
    train_chunk.to_csv(
        train_file,
        mode="a",
        header=first_train,
        index=False
    )

    # Save testing data
    test_chunk.to_csv(
        test_file,
        mode="a",
        header=first_test,
        index=False
    )

    first_train = False
    first_test = False

    train_counts.update(
        train_chunk[TARGET]
        .value_counts()
        .to_dict()
    )

    test_counts.update(
        test_chunk[TARGET]
        .value_counts()
        .to_dict()
    )

    print(
        f"Training: {sum(train_counts.values()):,} | "
        f"Testing: {sum(test_counts.values()):,}"
    )

    del chunk, train_chunk, test_chunk
    gc.collect()


print("\nTrain-Test Split Completed!")

print("\nTraining Distribution:")
print(train_counts)

print("\nTesting Distribution:")
print(test_counts)

if sum(train_counts.values()) == 0:
    raise ValueError("The training split is empty; check the input data and split settings")

if sum(test_counts.values()) == 0:
    raise ValueError("The testing split is empty; check the input data and split settings")


# ============================================================
# 4. THIRD PASS - CALCULATE MEDIAN
# ============================================================

print("\nStep 3: Calculating numerical medians...")

median_samples = []
median_rng = np.random.default_rng(RANDOM_STATE)

category_values = {
    col: set() for col in CATEGORICAL
}

for chunk in pd.read_csv(
    train_file,
    chunksize=CHUNK_SIZE
):

    # Convert numerical columns
    for col in NUMERICAL:

        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce"
        )

    # Collect random samples for median estimation
    sample = chunk[NUMERICAL].sample(
        n=min(MEDIAN_SAMPLE_SIZE, len(chunk)),
        # Use a different seed per chunk. Reusing the same seed samples the
        # same row positions from every chunk and can bias the estimate.
        random_state=int(median_rng.integers(0, np.iinfo(np.int32).max))
    )

    median_samples.append(sample)

    # Discover categorical values
    for col in CATEGORICAL:

        values = (
            chunk[col]
            .dropna()
            .astype(str)
            .unique()
        )

        category_values[col].update(values)

    del chunk, sample
    gc.collect()

if not median_samples:
    raise ValueError("No training rows were produced; cannot fit preprocessors")


# Calculate approximate median
median_data = pd.concat(
    median_samples,
    ignore_index=True
)

numeric_medians = median_data[NUMERICAL].median()

if numeric_medians.isna().any():

    raise ValueError(
        "Some numerical columns contain only missing values!"
    )

print("\nNumerical Median Values:")
print(numeric_medians)

del median_samples, median_data
gc.collect()


# ============================================================
# 5. FOURTH PASS - FIT STANDARD SCALER
# ============================================================

print("\nStep 4: Fitting StandardScaler...")

scaler = StandardScaler()

for chunk in pd.read_csv(
    train_file,
    chunksize=CHUNK_SIZE
):

    for col in NUMERICAL:

        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce"
        )

    # Median imputation
    numeric_data = chunk[NUMERICAL].fillna(
        numeric_medians
    )

    # Incremental scaler fitting
    scaler.partial_fit(numeric_data)

    del chunk, numeric_data
    gc.collect()

print("StandardScaler fitted successfully!")


# ============================================================
# 6. FIFTH PASS - FIT ONE-HOT ENCODER
# ============================================================

print("\nStep 5: Fitting One-Hot Encoder...")

categories = []

for col in CATEGORICAL:

    if not category_values[col]:

        raise ValueError(
            f"No valid categories found for {col}"
        )

    # Include missing value as a category
    column_categories = sorted(
        category_values[col]
    )

    if MISSING_CATEGORY in column_categories:

        raise ValueError(
            f"Reserved missing category already exists in {col}"
        )

    column_categories.append(MISSING_CATEGORY)

    categories.append(column_categories)


encoder = OneHotEncoder(
    categories=categories,
    handle_unknown="ignore",
    sparse_output=True
)

# Fit encoder using one artificial row
encoder.fit(
    pd.DataFrame({
        col: [categories[i][0]]
        for i, col in enumerate(CATEGORICAL)
    })
)

print("One-Hot Encoder fitted successfully!")


# ============================================================
# 7. SAVE PREPROCESSORS
# ============================================================

print("\nStep 6: Saving preprocessing objects...")

joblib.dump(
    scaler,
    os.path.join(OUTPUT_DIR, "scaler.pkl")
)

joblib.dump(
    encoder,
    os.path.join(OUTPUT_DIR, "encoder.pkl")
)

joblib.dump(
    numeric_medians,
    os.path.join(OUTPUT_DIR, "numeric_medians.pkl")
)

print("Preprocessors saved successfully!")


# ============================================================
# 8. SEVENTH PASS - TRANSFORM DATA IN CHUNKS
# ============================================================

print("\nStep 7: Transforming full datasets...")

for split_name, input_file in [
    ("train", train_file),
    ("test", test_file)
]:

    output_folder = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_processed"
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # Remove old processed files
    for filename in os.listdir(output_folder):

        if filename.endswith(".npz") or filename.endswith(".csv"):

            os.remove(
                os.path.join(output_folder, filename)
            )

    part_number = 0

    for chunk in pd.read_csv(
        input_file,
        chunksize=CHUNK_SIZE
    ):

        # Separate target
        y = chunk[TARGET].copy()

        # ----------------------------------------------------
        # Numerical Processing
        # ----------------------------------------------------

        for col in NUMERICAL:

            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce"
            )

        # Median imputation
        numeric_data = chunk[NUMERICAL].fillna(
            numeric_medians
        )

        # Standardization
        X_num = scaler.transform(
            numeric_data
        )

        # ----------------------------------------------------
        # Categorical Processing
        # ----------------------------------------------------

        categorical_data = chunk[
            CATEGORICAL
        ].copy()

        for col in CATEGORICAL:

            categorical_data[col] = (
                categorical_data[col]
                .fillna(MISSING_CATEGORY)
                .astype(str)
            )

        # One-hot encoding
        X_cat = encoder.transform(
            categorical_data
        )

        # ----------------------------------------------------
        # Combine Features
        # ----------------------------------------------------

        X_processed = hstack(
            [
                csr_matrix(X_num),
                X_cat
            ],
            format="csr"
        )

        # ----------------------------------------------------
        # Save Processed Data
        # ----------------------------------------------------

        save_npz(
            os.path.join(
                output_folder,
                f"X_{part_number}.npz"
            ),
            X_processed
        )

        y.to_csv(
            os.path.join(
                output_folder,
                f"y_{part_number}.csv"
            ),
            index=False
        )

        part_number += 1

        print(
            f"{split_name}: Processed part {part_number} "
            f"| Rows: {len(chunk):,} "
            f"| Features: {X_processed.shape[1]}"
        )

        del (
            chunk,
            numeric_data,
            categorical_data,
            X_num,
            X_cat,
            X_processed,
            y
        )

        gc.collect()


# ============================================================
# 9. SAVE FEATURE NAMES
# ============================================================

print("\nStep 8: Saving feature names...")

feature_names = (
    NUMERICAL +
    list(
        encoder.get_feature_names_out(
            CATEGORICAL
        )
    )
)

pd.Series(
    feature_names,
    name="feature_name"
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_names.csv"
    ),
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("FULL DATASET PREPROCESSING COMPLETED!")
print("========================================")

print(
    "Original valid rows:",
    sum(target_counts[label] for label in VALID_LABELS)
)

print(
    "Training rows:",
    sum(train_counts.values())
)

print(
    "Testing rows:",
    sum(test_counts.values())
)

print(
    "Original features:",
    len(FEATURES)
)

print(
    "Processed features:",
    len(feature_names)
)

print(
    "Training chunks:",
    len([
        f for f in os.listdir(
            os.path.join(OUTPUT_DIR, "train_processed")
        ) if f.endswith(".npz")
    ])
)

print(
    "Testing chunks:",
    len([
        f for f in os.listdir(
            os.path.join(OUTPUT_DIR, "test_processed")
        ) if f.endswith(".npz")
    ])
)

print(
    "Output folder:",
    FINAL_OUTPUT_DIR
)

print("\nAll preprocessing tasks completed!")

# Publish only after every artifact has been written. Keep a recoverable
# backup if the final rename fails (for example, when files are open on Windows).
backup_dir = BASE_DIR / ".processed_data_previous"
if backup_dir.exists():
    shutil.rmtree(backup_dir)
if FINAL_OUTPUT_DIR.exists():
    FINAL_OUTPUT_DIR.rename(backup_dir)
try:
    OUTPUT_DIR.rename(FINAL_OUTPUT_DIR)
except Exception:
    if backup_dir.exists() and not FINAL_OUTPUT_DIR.exists():
        backup_dir.rename(FINAL_OUTPUT_DIR)
    raise
if backup_dir.exists():
    shutil.rmtree(backup_dir)

print("Published outputs to:", FINAL_OUTPUT_DIR)
