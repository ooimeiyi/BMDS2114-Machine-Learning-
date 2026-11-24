"""
Streamlit UI for BDRoadRisk accident severity prediction.

Run (from this folder, after preprocessing_full.py and the model scripts have run):
    streamlit run UI.py

Sidebar : model selection (Random Forest / XGBoost)
Tab 1   : Prediction  - enter the 14 input features, press Predict, see the predicted severity
Tab 2   : Model comparison - accuracy, precision, recall, F1 of the models (overall, per class, per stage)
Tab 3   : EDA - exploratory charts of the raw data (built from a random sample of the CSV)
Tab 4   : Major scenarios - auto-generated inputs the model predicts as Major
Tab 5   : Feature signal - which features move the Major prediction
"""
import calendar
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Importing this module is also needed to load preprocessor.joblib (it holds the Preprocessor class).
from preprocessing import CSV_PATH, FEATURES, NOMINAL, ORDINAL, TARGET, TARGET_MAP, TARGET_NAMES

# ----------------------------------------------------------------------------
# File locations (edit if your files are somewhere else)
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PRE_PATH = HERE / "data" / "processed" / "preprocessor.joblib"
MODELS = {
    "Random Forest": {
        "model": HERE / "results" / "random_forest_full" / "rf_fine_tuned.joblib",
        "summary": HERE / "results" / "random_forest_full" / "rf_results_summary.csv",
        "per_class": HERE / "results" / "random_forest_full" / "rf_per_class.csv",
    },
    "XGBoost": {
        "model": HERE / "results" / "xgboost_full" / "xgb_fine_tuned.joblib",
        "summary": HERE / "results" / "xgboost_full" / "xgb_results_summary.csv",
        "per_class": HERE / "results" / "xgboost_full" / "xgb_per_class.csv",
    },
}
METRICS = ["Accuracy", "Precision", "Recall", "F1"]
SPEED_LIMITS = [30, 50, 80, 100, 120]
CLASS_COLORS = {"No_Accident": "#2e9e5b", "Minor": "#f0a30a", "Major": "#d64545"}

# Random scenarios (tabs 3 and 4) are drawn uniformly from these ranges / levels. Edit if needed.
SAMPLE_RANGES = {"latitude": (20.6, 26.6), "longitude": (88.0, 92.7),
                 "temperature_celsius": (10.0, 40.0), "visibility_km": (0.5, 20.0)}
CATEGORICAL_NUM = {"speed_limit_zone": SPEED_LIMITS, "month": list(range(1, 13))}
CAT_LEVELS = {**ORDINAL, **NOMINAL}
ALL_FEATURES = list(SAMPLE_RANGES) + list(CATEGORICAL_NUM) + list(CAT_LEVELS)
MAJOR = 2                       # class code of Major
N_SCENARIOS = 20_000            # random scenarios scored per model

EDA_PATH = HERE / "data" / "processed" / "eda_sample.csv"   # created once from the raw CSV
EDA_FRAC = 0.02                 # share of raw rows kept for the EDA charts (about 200k of 10M)
CLASS_ORDER = list(TARGET_NAMES.values())
CLASS_SCALE = alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values()))
NAME_MAP = {k: TARGET_NAMES[v] for k, v in TARGET_MAP.items()}

st.set_page_config(page_title="BDRoadRisk", page_icon="🚦", layout="wide")


@st.cache_resource
def load_preprocessor():
    return joblib.load(PRE_PATH) if PRE_PATH.exists() else None


@st.cache_resource
def load_model(path):
    return joblib.load(path) if Path(path).exists() else None


# ----------------------------------------------------------------------------
# Sidebar: model selection
# ----------------------------------------------------------------------------
st.sidebar.header("Model")
model_name = st.sidebar.radio("Choose a model", list(MODELS))
st.sidebar.caption("Used on the Prediction, Major scenarios and Feature signal tabs.")

pre = load_preprocessor()
model = load_model(str(MODELS[model_name]["model"]))
model_file = MODELS[model_name]["model"]
model_mtime = model_file.stat().st_mtime if model_file.exists() else 0.0

st.title("🚦 BDRoadRisk - Accident Severity")
tab_predict, tab_compare, tab_eda, tab_major, tab_signal = st.tabs(
    ["Prediction", "Model comparison", "EDA", "Major scenarios", "Feature signal"])


# ----------------------------------------------------------------------------
# Tab 1: input features -> predicted severity
# ----------------------------------------------------------------------------
def pretty(name):
    return name.replace("_", " ").capitalize()


def pick(col, label, options):
    return col.selectbox(label, options, key=label)


# Charts are fixed: no zoom, pan or selection (plain Altair charts without .interactive()).
def probability_chart(df):
    base = alt.Chart(df).encode(
        x=alt.X("Class:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None))
    bars = base.mark_bar().encode(color=alt.Color(
        "Class:N", legend=None,
        scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values()))))
    labels = base.mark_text(dy=-8, fontSize=13).encode(text=alt.Text("Probability:Q", format=".1%"))
    return (bars + labels).properties(width=460, height=300)


def comparison_chart(table):
    long = table[METRICS].reset_index(names="Model").melt(
        id_vars="Model", var_name="Metric", value_name="Value")
    base = alt.Chart(long).encode(
        x=alt.X("Metric:N", sort=METRICS, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="Model:N",
        y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title=None))
    bars = base.mark_bar().encode(color=alt.Color("Model:N", title=None))
    labels = base.mark_text(dy=-8, fontSize=11).encode(text=alt.Text("Value:Q", format=".3f"))
    return (bars + labels).properties(width=560, height=320)


def hbar(df, cat, val, color, fmt, width=460):
    """Fixed horizontal bar chart, longest bar first, value printed at the end of each bar."""
    base = alt.Chart(df).encode(
        y=alt.Y(f"{cat}:N", sort="-x", title=None),
        x=alt.X(f"{val}:Q", title=None, axis=alt.Axis(format=fmt),
                scale=alt.Scale(domainMin=0, domainMax=float(df[val].max()) * 1.25 or 1)))
    bars = base.mark_bar(color=color)
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text(f"{val}:Q", format=fmt))
    return (bars + labels).properties(width=width, height=24 * len(df) + 10)


def predict_major(model, pre, raw):
    """Class probabilities for raw feature rows. Returns (probabilities, class codes)."""
    X = pre.transform_raw(raw)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        X = X[list(names)]
    return model.predict_proba(X), [int(c) for c in model.classes_]


def random_scenarios(n, seed):
    rng = np.random.default_rng(seed)
    data = {c: rng.uniform(lo, hi, n) for c, (lo, hi) in SAMPLE_RANGES.items()}
    for c, levels in {**CATEGORICAL_NUM, **CAT_LEVELS}.items():
        data[c] = rng.choice(levels, n)
    return pd.DataFrame(data)


@st.cache_data(show_spinner="Scoring random scenarios...")
def score_random_scenarios(model_name, mtime, seed, n):
    """Random raw inputs + the model's Major probability and predicted class (cached)."""
    raw = random_scenarios(n, seed)
    proba, classes = predict_major(load_model(str(MODELS[model_name]["model"])), load_preprocessor(), raw)
    raw["Major probability"] = proba[:, classes.index(MAJOR)]
    raw["Predicted"] = [TARGET_NAMES[classes[i]] for i in proba.argmax(axis=1)]
    return raw


def common_conditions(scen, group):
    """Per categorical feature: the level that is most common among the Major scenarios."""
    rows = []
    for c in list(CAT_LEVELS) + list(CATEGORICAL_NUM):
        g = group[c].astype(str).value_counts(normalize=True)
        a = scen[c].astype(str).value_counts(normalize=True)
        level = g.index[0]
        rows.append({"Feature": c, "Most common level": level, "Share in Major scenarios": g.iloc[0],
                     "Share in all scenarios": a[level], "Lift": g.iloc[0] / a[level]})
    return pd.DataFrame(rows).sort_values("Lift", ascending=False)


def level_effects(scen):
    """Average Major probability for every level of every feature (numbers are cut into 4 bins)."""
    rows = []
    for c in ALL_FEATURES:
        key = pd.cut(scen[c], bins=4).astype(str) if c in SAMPLE_RANGES else scen[c].astype(str)
        for level, value in scen.groupby(key)["Major probability"].mean().items():
            rows.append({"Feature": c, "Level": level, "Avg Major probability": value})
    return pd.DataFrame(rows)


def grouped_importance(model, pre):
    """The model's own importance, summed over the encoded columns of each original feature."""
    if not hasattr(model, "feature_importances_"):
        return None
    cols = list(getattr(model, "feature_names_in_", pre.columns_))
    out = {}
    for col, v in zip(cols, model.feature_importances_):
        base = next((k for k in NOMINAL if col.startswith(k + "_")), None)
        if base is None and col.endswith(("_sin", "_cos")):
            base = col.rsplit("_", 1)[0]
        out[base or col] = out.get(base or col, 0.0) + float(v)
    return pd.DataFrame({"Feature": list(out), "Importance": list(out.values())})


def sensitivity(model, pre, row):
    """Major probability when ONE feature of the current input is changed to each of its values."""
    variants = []
    for c in row:
        if c in SAMPLE_RANGES:
            values = [float(v) for v in np.round(np.linspace(*SAMPLE_RANGES[c], 7), 2)]
        else:
            values = {**CATEGORICAL_NUM, **CAT_LEVELS}[c]
        variants += [(c, v, {**row, c: v}) for v in values]
    proba, classes = predict_major(model, pre, pd.DataFrame([r for _, _, r in variants]))
    return pd.DataFrame({"Feature": [c for c, _, _ in variants], "Value": [str(v) for _, v, _ in variants],
                         "Major probability": proba[:, classes.index(MAJOR)]})


# ---- EDA helpers (all charts are fixed: no zoom or pan) ----------------------
def build_eda_sample():
    """One pass over the raw CSV: keep a random EDA_FRAC of the rows (features + severity)."""
    parts = []
    reader = pd.read_csv(CSV_PATH, usecols=FEATURES + [TARGET], chunksize=500_000,
                         keep_default_na=False, na_values=[""])
    for i, chunk in enumerate(reader):
        chunk[TARGET] = chunk[TARGET].str.strip().str.lower().map(NAME_MAP)
        parts.append(chunk.dropna(subset=[TARGET]).sample(frac=EDA_FRAC, random_state=42 + i))
    EDA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(EDA_PATH, index=False)


@st.cache_data
def load_eda(mtime):
    df = pd.read_csv(EDA_PATH, keep_default_na=False, na_values=[""])
    df["Major"] = (df[TARGET] == "Major").astype(int)
    return df


def level_series(df, c):
    """Values of a categorical column as text (speed limit and month as whole numbers)."""
    if c in CATEGORICAL_NUM:
        return pd.to_numeric(df[c], errors="coerce").round().astype("Int64").astype(str)
    return df[c].astype(str)


def level_order(c):
    return [str(v) for v in CATEGORICAL_NUM[c]] if c in CATEGORICAL_NUM else list(CAT_LEVELS[c])


def class_balance_chart(df):
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    frame = pd.DataFrame({"Class": counts.index, "Rows": counts.values})
    frame["Label"] = [f"{n:,} ({n / counts.sum():.1%})" for n in counts.values]
    base = alt.Chart(frame).encode(x=alt.X("Class:N", sort=CLASS_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
                                   y=alt.Y("Rows:Q", title="Rows"))
    bars = base.mark_bar().encode(color=alt.Color("Class:N", scale=CLASS_SCALE, legend=None))
    labels = base.mark_text(dy=-8).encode(text="Label:N")
    return (bars + labels).properties(width=460, height=300)


def numeric_charts(df, c, bins=25):
    """Row counts and severity mix per value range of a numeric feature."""
    cut = pd.cut(df[c], bins=bins)
    g = df.groupby([cut, df[TARGET]], observed=False).size().rename("Count").reset_index()
    g.columns = ["bin", "Class", "Count"]
    g["start"], g["end"] = g["bin"].apply(lambda b: b.left), g["bin"].apply(lambda b: b.right)
    base = alt.Chart(g[["start", "end", "Class", "Count"]]).encode(
        x=alt.X("start:Q", title=c), x2="end:Q",
        color=alt.Color("Class:N", scale=CLASS_SCALE, legend=alt.Legend(title=None)))
    counts = base.mark_bar().encode(y=alt.Y("Count:Q", title="Rows")).properties(
        width=400, height=260, title="Rows per value range")
    share = base.mark_bar().encode(y=alt.Y("Count:Q", stack="normalize", axis=alt.Axis(format="%"),
                                           title="Share of rows")).properties(
        width=400, height=260, title="Severity mix per value range")
    return counts, share


def category_charts(df, c):
    """Row counts and severity mix per level of a categorical feature."""
    g = df.groupby([level_series(df, c), df[TARGET]]).size().rename("Count").reset_index()
    g.columns = ["Level", "Class", "Count"]
    base = alt.Chart(g).encode(
        x=alt.X("Level:N", sort=level_order(c), title=c, axis=alt.Axis(labelAngle=-30)),
        color=alt.Color("Class:N", scale=CLASS_SCALE, legend=alt.Legend(title=None)))
    counts = base.mark_bar().encode(y=alt.Y("Count:Q", title="Rows")).properties(
        width=400, height=260, title="Rows per level")
    share = base.mark_bar().encode(y=alt.Y("Count:Q", stack="normalize", axis=alt.Axis(format="%"),
                                           title="Share of rows")).properties(
        width=400, height=260, title="Severity mix per level")
    return counts, share


def major_rate_table(df, min_rows=200):
    """Actual share of Major rows for every feature level (numbers cut into 5 ranges)."""
    rows = []
    for c in ALL_FEATURES:
        key = pd.cut(df[c], bins=5).astype(str) if c in SAMPLE_RANGES else level_series(df, c)
        g = df.groupby(key)["Major"].agg(["mean", "size"])
        rows += [{"Condition": f"{c} = {lvl}", "Major rate": r["mean"], "Rows": int(r["size"])}
                 for lvl, r in g.iterrows() if r["size"] >= min_rows]
    return pd.DataFrame(rows)


def correlation_heatmap(df):
    cols = list(SAMPLE_RANGES) + list(CATEGORICAL_NUM) + ["Major"]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    long = corr.reset_index(names="A").melt(id_vars="A", var_name="B", value_name="Correlation")
    base = alt.Chart(long).encode(x=alt.X("B:N", sort=cols, title=None, axis=alt.Axis(labelAngle=-30)),
                                  y=alt.Y("A:N", sort=cols, title=None))
    rect = base.mark_rect().encode(color=alt.Color(
        "Correlation:Q", legend=None, scale=alt.Scale(domain=[-1, 0, 1], range=["#3b6ea5", "#f4f4f4", "#d64545"])))
    text = base.mark_text(fontSize=11).encode(text=alt.Text("Correlation:Q", format=".2f"))
    return (rect + text).properties(width=460, height=380)


# ---- model comparison helpers ------------------------------------------------
def stage_chart(summaries, metric):
    """One metric for every tuning stage (Baseline / Tuned / Fine-tuned) of every model."""
    long = pd.concat([df[["Model", metric]].rename(columns={"Model": "Stage", metric: "Value"}).assign(Model=name)
                      for name, df in summaries.items()])
    order = list(next(iter(summaries.values()))["Model"])
    base = alt.Chart(long).encode(x=alt.X("Stage:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                                  xOffset="Model:N", y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title=None))
    bars = base.mark_bar().encode(color=alt.Color("Model:N", title=None))
    labels = base.mark_text(dy=-8, fontSize=11).encode(text=alt.Text("Value:Q", format=".3f"))
    return (bars + labels).properties(width=440, height=300)


def per_class_chart(per_class, metric):
    """One metric for each severity class, final model of every algorithm."""
    long = pd.concat([df[[metric]].reset_index(names="Class").rename(columns={metric: "Value"}).assign(Model=name)
                      for name, df in per_class.items()])
    base = alt.Chart(long).encode(x=alt.X("Class:N", sort=CLASS_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
                                  xOffset="Model:N", y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title=None))
    bars = base.mark_bar().encode(color=alt.Color("Model:N", title=None))
    labels = base.mark_text(dy=-8, fontSize=11).encode(text=alt.Text("Value:Q", format=".3f"))
    return (bars + labels).properties(width=440, height=300)


def final_per_class(path):
    """Per-class table (index = class) of the fine-tuned model from a *_per_class.csv file."""
    pc = pd.read_csv(path, index_col=[0, 1])
    stages = list(dict.fromkeys(pc.index.get_level_values(0)))
    return pc.xs("Fine-tuned" if "Fine-tuned" in stages else stages[-1], level=0)


with tab_predict:
    st.subheader(f"Input features  (model: {model_name})")
    c1, c2, c3 = st.columns(3)

    c1.markdown("**Location and time**")
    row = {
        "latitude": c1.number_input("Latitude", -90.0, 90.0, 23.78, step=0.0001, format="%.5f"),
        "longitude": c1.number_input("Longitude", -180.0, 180.0, 90.40, step=0.0001, format="%.5f"),
        "month": c1.selectbox("Month", list(range(1, 13)), index=5,
                              format_func=lambda m: calendar.month_name[m]),
        "time_of_day": pick(c1, "Time of day", NOMINAL["time_of_day"]),
    }

    c2.markdown("**Road**")
    row.update({
        "speed_limit_zone": c2.selectbox("Speed limit zone (km/h)", SPEED_LIMITS, index=1),
        "road_class": pick(c2, "Road class", NOMINAL["road_class"]),
        "road_quality": pick(c2, "Road quality", ORDINAL["road_quality"]),
        "road_surface": pick(c2, "Road surface", NOMINAL["road_surface"]),
        "junction_type": pick(c2, "Junction type", NOMINAL["junction_type"]),
    })

    c3.markdown("**Conditions and vehicle**")
    row.update({
        "weather_condition": pick(c3, "Weather condition", NOMINAL["weather_condition"]),
        "light_condition": pick(c3, "Light condition", NOMINAL["light_condition"]),
        "visibility_km": c3.number_input("Visibility (km)", 0.0, 100.0, 10.0, step=0.5),
        "temperature_celsius": c3.number_input("Temperature (°C)", -20.0, 60.0, 28.0, step=0.5),
        "vehicle_type": pick(c3, "Vehicle type", NOMINAL["vehicle_type"]),
    })

    st.divider()
    st.subheader("Prediction")

    if pre is None:
        st.error(f"Preprocessor not found: `{PRE_PATH}`. Run preprocessing_full.py first.")
    elif model is None:
        st.warning(f"{model_name} model not found: `{MODELS[model_name]['model']}`. "
                   "Train and save it first, or pick another model in the sidebar.")
    else:
        if st.button("Predict", type="primary"):
            X = pre.transform_raw(pd.DataFrame([row]))
            names = getattr(model, "feature_names_in_", None)
            if names is not None:
                X = X[list(names)]
            st.session_state["result"] = {
                "model": model_name, "row": dict(row), "X": X,
                "proba": model.predict_proba(X)[0], "classes": [int(c) for c in model.classes_]}

        res = st.session_state.get("result")
        if res is None:
            st.info("Set the inputs above, then press **Predict**.")
        else:
            if res["model"] != model_name or res["row"] != row:
                st.caption(f"⚠️ This result was made with {res['model']} and the inputs "
                           "from the last click. Press Predict again to update it.")
            proba, classes = res["proba"], res["classes"]
            label = TARGET_NAMES[classes[int(proba.argmax())]]

            show = {"No_Accident": st.success, "Minor": st.warning, "Major": st.error}[label]
            show(f"Predicted severity: **{label}**  (confidence {proba.max():.1%})")

            cols = st.columns(len(classes))
            for col, c, p in zip(cols, classes, proba):
                col.metric(TARGET_NAMES[c], f"{p:.1%}")
            st.altair_chart(probability_chart(pd.DataFrame({
                "Class": [TARGET_NAMES[c] for c in classes], "Probability": proba})))
            with st.expander("Encoded model input"):
                st.dataframe(res["X"])


# ----------------------------------------------------------------------------
# Tab 2: model comparison
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Test-set results")
    st.caption("Precision, recall and F1 are macro averages over the 3 classes. "
               "Each model's final (fine-tuned) version is compared.")

    summaries = {}
    for name, paths in MODELS.items():
        if paths["summary"].exists():
            summaries[name] = pd.read_csv(paths["summary"])
        else:
            st.info(f"{name}: no results yet (expected `{paths['summary']}`).")

    if summaries:
        final = {}
        for name, df in summaries.items():
            pick_row = df[df["Model"] == "Fine-tuned"]
            final[name] = (pick_row if len(pick_row) else df.tail(1)).iloc[0]
        table = pd.DataFrame(final).T.drop(columns="Model")
        cols = [c for c in METRICS + ["F1 weighted", "ROC AUC"] if c in table.columns]
        table = table[cols].astype(float)

        st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0), width="stretch")
        st.altair_chart(comparison_chart(table))

        st.markdown("**Compare in more detail**")
        metric = st.radio("Metric", ["F1", "Precision", "Recall", "Accuracy"], horizontal=True, key="cmp_metric")
        left, right = st.columns(2)
        left.markdown(f"{metric} by tuning stage")
        left.altair_chart(stage_chart(summaries, metric))
        right.markdown(f"{metric} per severity class (final models)")
        per_class = {name: final_per_class(MODELS[name]["per_class"])
                     for name in summaries if MODELS[name]["per_class"].exists()}
        if metric == "Accuracy":
            right.info("Accuracy is an overall number; choose F1, Precision or Recall to compare per class.")
        elif per_class:
            right.altair_chart(per_class_chart(per_class, metric))
        else:
            right.info("No per-class results found yet.")

        for name, df in summaries.items():
            with st.expander(f"{name}: all stages and per-class results"):
                st.dataframe(df.round(4), width="stretch", hide_index=True)
                pc = MODELS[name]["per_class"]
                if pc.exists():
                    st.markdown("Per class")
                    st.dataframe(pd.read_csv(pc, index_col=[0, 1]), width="stretch")


# ----------------------------------------------------------------------------
# Tab 3: auto-generated scenarios that the model predicts as Major
# ----------------------------------------------------------------------------
with tab_major:
    st.subheader(f"Scenarios predicted as Major  (model: {model_name})")
    if pre is None or model is None:
        st.warning("Preprocessor or model not found - see the Prediction tab for the expected files.")
    else:
        if st.button("Generate a new random batch"):
            st.session_state["seed"] = st.session_state.get("seed", 0) + 1
        seed = st.session_state.get("seed", 0)
        scen = score_random_scenarios(model_name, model_mtime, seed, N_SCENARIOS)
        major = scen[scen["Predicted"] == "Major"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Random scenarios tested", f"{len(scen):,}")
        m2.metric("Predicted as Major", f"{len(major):,}  ({len(major) / len(scen):.1%})")
        m3.metric("Highest Major probability", f"{scen['Major probability'].max():.1%}")
        st.caption("Scenarios are drawn uniformly at random over the feature values, so they do not follow "
                   "the real data distribution. Use them to see what the model considers risky.")

        if len(major) >= 30:
            group, note = major, "scenarios predicted as Major"
        else:
            group = scen.nlargest(max(30, int(0.05 * len(scen))), "Major probability")
            note = (f"only {len(major)} scenarios reached a Major prediction, so the {len(group)} scenarios "
                    "with the highest Major probability are used instead")
        st.markdown(f"**Sample inputs** - the 15 most likely Major scenarios ({note})")
        cols = ["Major probability", "Predicted"] + ALL_FEATURES
        top = group.sort_values("Major probability", ascending=False).head(15)[cols]
        st.dataframe(top.round({"latitude": 4, "longitude": 4, "temperature_celsius": 1,
                                "visibility_km": 1, "Major probability": 3}),
                     width="stretch", hide_index=True)

        st.markdown("**What these scenarios have in common** (lift = how much more often the level appears "
                    "than in all scenarios; 1.0x means no difference)")
        cc = common_conditions(scen, group)
        st.dataframe(cc.style.format({"Share in Major scenarios": "{:.1%}", "Share in all scenarios": "{:.1%}",
                                      "Lift": "{:.2f}x"}), width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# Tab 4: which features move the Major prediction
# ----------------------------------------------------------------------------
with tab_signal:
    st.subheader(f"Which features drive the Major prediction?  (model: {model_name})")
    if pre is None or model is None:
        st.warning("Preprocessor or model not found - see the Prediction tab for the expected files.")
    else:
        scen = score_random_scenarios(model_name, model_mtime, st.session_state.get("seed", 0), N_SCENARIOS)
        eff = level_effects(scen)
        st.caption(f"Based on {len(scen):,} random scenarios; the average Major probability is "
                   f"{scen['Major probability'].mean():.1%}.")

        left, right = st.columns(2)
        swing = (eff.groupby("Feature")["Avg Major probability"].agg(lambda x: x.max() - x.min())
                 .reset_index(name="Effect"))
        left.markdown("**Effect on Major probability**  \n(gap between the riskiest and the safest value)")
        left.altair_chart(hbar(swing, "Feature", "Effect", "#d64545", ".1%"))
        imp = grouped_importance(model, pre)
        right.markdown("**Model feature importance**  \n(all classes, built into the model)")
        if imp is not None:
            right.altair_chart(hbar(imp, "Feature", "Importance", "#3b6ea5", ".1%"))
        else:
            right.info("This model does not provide feature importances.")

        st.markdown("**Riskiest conditions** (average Major probability when the value is present)")
        top = eff.assign(Condition=eff["Feature"] + " = " + eff["Level"]).nlargest(12, "Avg Major probability")
        st.altair_chart(hbar(top, "Condition", "Avg Major probability", "#d64545", ".1%", width=560))

        st.divider()
        st.markdown("**Your current input** (from the Prediction tab): change one feature at a time")
        proba, classes = predict_major(model, pre, pd.DataFrame([row]))
        st.metric("Major probability of the current input", f"{proba[0, classes.index(MAJOR)]:.1%}")
        sens = sensitivity(model, pre, row)
        rows = []
        for c, g in sens.groupby("Feature", sort=False):
            hi, lo = g.loc[g["Major probability"].idxmax()], g.loc[g["Major probability"].idxmin()]
            rows.append({"Feature": c, "Current value": str(row[c]),
                         "Riskiest value": hi["Value"], "Major prob. at riskiest": hi["Major probability"],
                         "Safest value": lo["Value"], "Major prob. at safest": lo["Major probability"],
                         "Swing": hi["Major probability"] - lo["Major probability"]})
        summ = pd.DataFrame(rows).sort_values("Swing", ascending=False)
        st.altair_chart(hbar(summ, "Feature", "Swing", "#d64545", ".1%", width=560))
        st.dataframe(summ.style.format({"Major prob. at riskiest": "{:.1%}", "Major prob. at safest": "{:.1%}",
                                        "Swing": "{:.1%}"}), width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# Tab 3: EDA (exploratory data analysis of the raw data)
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory data analysis")
    if not EDA_PATH.exists():
        st.info(f"The EDA charts use a random {EDA_FRAC:.0%} sample of `{CSV_PATH.name}`. "
                "Building it reads the whole CSV once (a few minutes) and is saved for next time.")
        if not CSV_PATH.exists():
            st.error(f"Raw data not found: `{CSV_PATH}`.")
        elif st.button("Build EDA sample"):
            with st.spinner("Reading the CSV in chunks..."):
                build_eda_sample()
            st.rerun()
    else:
        eda = load_eda(EDA_PATH.stat().st_mtime)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows in the sample", f"{len(eda):,}")
        m2.metric("Features", len(ALL_FEATURES))
        m3.metric("Major share", f"{eda['Major'].mean():.1%}")
        m4.metric("Missing cells", f"{int(eda[ALL_FEATURES].isna().sum().sum()):,}")
        st.caption(f"Based on a random {EDA_FRAC:.0%} sample of the raw data. "
                   f"Only the {len(ALL_FEATURES)} features used by the models are shown.")

        st.markdown("**Class balance** - the target is imbalanced, so accuracy alone is misleading")
        st.altair_chart(class_balance_chart(eda))

        st.markdown("**Numeric features**")
        feat = st.selectbox("Feature", list(SAMPLE_RANGES), key="eda_num")
        c1, c2 = st.columns(2)
        chart_counts, chart_share = numeric_charts(eda, feat)
        c1.altair_chart(chart_counts)
        c2.altair_chart(chart_share)

        st.markdown("**Categorical features**")
        feat = st.selectbox("Feature", list(CAT_LEVELS) + list(CATEGORICAL_NUM), key="eda_cat")
        c1, c2 = st.columns(2)
        chart_counts, chart_share = category_charts(eda, feat)
        c1.altair_chart(chart_counts)
        c2.altair_chart(chart_share)

        st.markdown("**Actual Major rate** - conditions where Major is most common in the data "
                    f"(overall rate {eda['Major'].mean():.1%}; groups under 200 rows are left out)")
        rates = major_rate_table(eda).nlargest(12, "Major rate")
        st.altair_chart(hbar(rates, "Condition", "Major rate", "#d64545", ".1%", width=560))

        st.markdown("**Correlation** between the numeric features and Major (1 = Major, 0 = not Major)")
        st.altair_chart(correlation_heatmap(eda))

        with st.expander("Numeric summary statistics"):
            st.dataframe(eda[list(SAMPLE_RANGES) + list(CATEGORICAL_NUM)].describe().T.round(3), width="stretch")