import importlib
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- settings
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "traffic_data.csv"
OUT_DIR = HERE / "data" / "processed"
CHUNK_ROWS = 500_000            # lower if memory is tight
PIECE_ROWS = 1_000_000          # rows transformed/written at a time
TEST_SIZE = 0.20
SEED = 42

# ------------------------------------------------------------------ schema
TARGET = "accident_severity"
TARGET_MAP = {"no_accident": 0, "minor": 1, "major": 2}
TARGET_NAMES = {0: "No_Accident", 1: "Minor", 2: "Major"}

# Columns removed manually (documentation only; they are not in the lists below).
REMOVED_COLUMNS = {
    "vehicle_age_years": "integer, right-skewed",
    "driver_age": "integer, weak signal",
    "reaction_time_sec": "continuous, time-conditional",
    "driver_experience_level": "ordinal: Novice < Intermediate < Experienced",
    "alcohol_influence": "ordinal: None < Suspected < Confirmed",
    "seatbelt_worn": "binary: Yes > No",
    "speed_kmh": "continuous, road-class dependent",
    "acceleration_ms2": "continuous",
    "traffic_density": "integer, non-monotonic",
}

NUMERIC = ["latitude", "longitude", "temperature_celsius", "speed_limit_zone", "visibility_km"]
CYCLIC = ["month"]                           # 1-12, sin/cos
ORDINAL = {"road_quality": ["Poor", "Fair", "Good"]}
NOMINAL = {
    "weather_condition": ["Clear", "Rain", "Fog", "Storm"],
    "road_surface": ["Dry", "Wet", "Icy", "Damaged"],
    "junction_type": ["None", "T-Junction", "Crossroad", "Roundabout"],
    "light_condition": ["Daylight", "Dusk_Dawn", "Dark_Lit", "Dark_Unlit"],
    "time_of_day": ["Morning", "Afternoon", "Evening", "Night"],
    "vehicle_type": ["Car", "Truck", "Motorcycle", "Bus", "Van"],
    "road_class": ["motorway", "primary", "secondary", "residential"],
}
LEVELS = {**ORDINAL, **NOMINAL}
CAT_DTYPES = {c: pd.CategoricalDtype(lv) for c, lv in LEVELS.items()}
NUMERIC_LIKE = NUMERIC + CYCLIC
FEATURES = NUMERIC_LIKE + list(LEVELS)
assert not set(FEATURES) & set(REMOVED_COLUMNS), "a removed column is also listed as a feature"

REPORT = []


def log(text=""):
    print(text)
    REPORT.append(str(text))


def class_table(counts, names=TARGET_NAMES):
    s = pd.Series(counts).sort_index().rename(index=names)
    return pd.DataFrame({"rows": s.astype(int), "share_%": (100 * s / s.sum()).round(2)}).to_string()


# ------------------------------------------------------- cleaning (per chunk)
def to_compact(df):
    """Raw columns -> float32 numbers and fixed-level categoricals; bad values become NaN."""
    out = {}
    for c in NUMERIC_LIKE:
        v = df[c].astype("float32")
        if c == "month":
            v = v.mask(~v.between(1, 12))                  # impossible month -> missing
        elif c == "visibility_km":
            v = v.mask(v < 0)                              # negative visibility -> missing
        out[c] = v
    for c in LEVELS:
        out[c] = df[c].str.strip().astype(CAT_DTYPES[c])   # unknown level -> NaN
    return pd.DataFrame(out, index=df.index)


def stream_and_clean():
    xs, ys, hs, n_raw = [], [], [], 0
    reader = pd.read_csv(CSV_PATH, usecols=FEATURES + [TARGET], chunksize=CHUNK_ROWS,
                         dtype={c: "float32" for c in NUMERIC_LIKE},
                         keep_default_na=False, na_values=[""])
    for i, chunk in enumerate(reader):
        n_raw += len(chunk)
        y = chunk[TARGET].str.strip().str.lower().map(TARGET_MAP)
        ok = y.notna()                                   # drop rows with unusable target
        chunk, y = chunk[ok], y[ok].astype("int8")
        X = to_compact(chunk)
        hs.append(pd.util.hash_pandas_object(X.assign(**{TARGET: y}), index=False).to_numpy())
        xs.append(X.reset_index(drop=True))
        ys.append(y.to_numpy())
        print(f"  chunk {i + 1}: {n_raw:,} rows read")
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(hs), n_raw


# ------------------------------------------------------------ preprocessor
class Preprocessor:
    """Median / most-frequent imputation, standardisation, sin/cos month,
    ordinal codes, one-hot. Fitted on the training set only."""

    def fit(self, df):
        self.fill = {c: float(df[c].median()) for c in NUMERIC_LIKE}
        self.fill.update({c: str(df[c].mode().iloc[0]) for c in LEVELS})
        self.mean, self.std = {}, {}
        for c in NUMERIC:
            x = df[c].astype("float64").fillna(self.fill[c])
            self.mean[c], self.std[c] = float(x.mean()), float(x.std(ddof=0)) or 1.0
        self.columns_ = list(self.transform(df.head(5)).columns)
        return self

    def transform(self, df):
        out = {}
        for c in NUMERIC:
            x = df[c].astype("float32").fillna(self.fill[c]).to_numpy()
            out[c] = ((x - self.mean[c]) / self.std[c]).astype("float32")
        for c in CYCLIC:
            m = df[c].astype("float32").fillna(self.fill[c]).to_numpy()
            out[c + "_sin"] = np.sin(2 * np.pi * m / 12).astype("float32")
            out[c + "_cos"] = np.cos(2 * np.pi * m / 12).astype("float32")
        for c in LEVELS:
            codes = pd.Categorical(df[c], categories=LEVELS[c]).codes.copy()
            codes[codes < 0] = LEVELS[c].index(self.fill[c])
            if c in ORDINAL:
                out[c] = codes.astype("float32")             # 0,1,2 in the stated order
            else:
                for k, level in enumerate(LEVELS[c]):
                    out[f"{c}_{level}"] = (codes == k).astype("float32")
        return pd.DataFrame(out)

    def transform_raw(self, raw_df):
        """For NEW raw rows (same columns as the CSV), e.g. in a Streamlit app."""
        return self.transform(to_compact(raw_df))


# ------------------------------------------------------------------ saving
def save_part(name, X, y, pre):
    """Transform piece by piece, write X_<name>.parquet and y_<name>.parquet (one file each)."""
    paths = [OUT_DIR / f"X_{name}.parquet", OUT_DIR / f"y_{name}.parquet"]
    wx = wy = None
    for a in range(0, len(X), PIECE_ROWS):
        tx = pa.Table.from_pandas(pre.transform(X.iloc[a:a + PIECE_ROWS]), preserve_index=False)
        ty = pa.Table.from_pandas(pd.DataFrame({TARGET: y[a:a + PIECE_ROWS]}), preserve_index=False)
        if wx is None:
            wx, wy = pq.ParquetWriter(paths[0], tx.schema), pq.ParquetWriter(paths[1], ty.schema)
        wx.write_table(tx)
        wy.write_table(ty)
    wx.close()
    wy.close()
    return paths


def load_split():
    """Return X_train, X_test, y_train, y_test (already encoded and scaled)."""
    r = lambda s: pd.read_parquet(OUT_DIR / f"{s}.parquet")
    return r("X_train"), r("X_test"), r("y_train")[TARGET], r("y_test")[TARGET]


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Streaming and cleaning the raw file...")
    X, y, h, n_raw = stream_and_clean()
    log(f"Rows read: {n_raw:,} | usable target: {len(y):,}")
    log("Raw target distribution:\n" + class_table(np.bincount(y)))
    log(f"Removed manually ({len(REMOVED_COLUMNS)}): {', '.join(REMOVED_COLUMNS)}")
    log(f"Features kept ({len(FEATURES)}): {', '.join(FEATURES)}")

    dup = pd.Series(h).duplicated().to_numpy()             # duplicates over ALL chunks
    n_dups = int(dup.sum())
    if n_dups:
        X, y = X[~dup].reset_index(drop=True), y[~dup]
    del h, dup
    log(f"Duplicates removed: {n_dups:,} | rows after cleaning: {len(y):,}")
    log(f"Missing cells to impute: {int(X.isna().sum().sum()):,}")

    tr_i, te_i = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_tr, y_tr = X.iloc[tr_i].reset_index(drop=True), y[tr_i]
    X_te, y_te = X.iloc[te_i].reset_index(drop=True), y[te_i]
    del X, y
    log(f"\nTrain: {len(y_tr):,} rows | Test: {len(y_te):,} rows")
    log("Train classes:\n" + class_table(np.bincount(y_tr)))
    log("Test classes:\n" + class_table(np.bincount(y_te)))

    # build from the IMPORTED module so preprocessor.joblib loads from any script
    pre = importlib.import_module(Path(__file__).stem).Preprocessor().fit(X_tr)
    log(f"\nEncoded columns ({len(pre.columns_)}): {', '.join(pre.columns_)}")

    for name, Xp, yp in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        for p in save_part(name, Xp, yp, pre):
            log(f"saved {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    joblib.dump(pre, OUT_DIR / "preprocessor.joblib", compress=3)
    log(f"Total time: {(time.time() - t0) / 60:.1f} min")

    (OUT_DIR / "preprocessing_report.txt").write_text("\n".join(REPORT), encoding="utf-8")