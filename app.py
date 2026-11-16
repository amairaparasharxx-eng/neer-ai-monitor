import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_URL = "https://raw.githubusercontent.com/sudesokin/Water-Quality-Analysis-and-Forecasting/main/dataset/water_potability.csv"
FEATURES = ["ph", "Solids", "Conductivity", "Turbidity"]
TARGET = "Potability"
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "water_quality.csv"
MODEL_PATH = ROOT / "model" / "neer_model.joblib"

st.set_page_config(page_title="Neer | Water Intelligence", page_icon="💧", layout="wide")

st.markdown("""
<style>
.main {background:#f5f9fc}
.block-container {max-width:1180px;padding-top:2rem}
div[data-testid="stMetric"] {
    background:white !important;
    color:#17324d !important;
    padding:14px;
    border-radius:12px;
    border:1px solid #e2edf2;
}
div[data-testid="stMetric"] * {
    color:#17324d !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the dataset and preparing Neer's model...")
def prepare_model():
    DATA_PATH.parent.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    metrics = {
        "rows": len(df),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline, metrics

st.markdown("<h5 style='color:#087e8b;letter-spacing:3px'>NEER · WATER INTELLIGENCE</h5>", unsafe_allow_html=True)
st.title("See the patterns. Understand the water.")
st.write("An AI-powered demonstration that analyses multiple water-quality measurements together.")

with st.sidebar:
    st.header("About Neer")
    st.write("Neer explores how several water-quality measurements can be analysed together using machine learning.")
    st.info("Prototype mode: all readings entered here are simulated.")
    st.write("Model: Random Forest classifier")

try:
    model, metrics = prepare_model()
except Exception as error:
    st.error("Neer could not load the dataset or prepare its model.")
    st.exception(error)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset records", f"{metrics['rows']:,}")
c2.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
c3.metric("Test precision", f"{metrics['precision']:.1%}")
c4.metric("Test recall", f"{metrics['recall']:.1%}")

st.divider()
st.header("Virtual sensor station")
st.write("Change the simulated readings and analyse their combined pattern.")
left, right = st.columns(2)
with left:
    ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
    solids = st.slider("Solids (dataset units)", 0.0, 60000.0, 20000.0, 100.0)
with right:
    conductivity = st.slider("Electrical conductivity (µS/cm)", 0.0, 800.0, 425.0, 5.0)
    turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 3.5, 0.1)

readings = pd.DataFrame([{
    "ph": ph, "Solids": solids,
    "Conductivity": conductivity, "Turbidity": turbidity
}], columns=FEATURES)

if st.button("Analyse combined readings", type="primary", use_container_width=True):
    label = int(model.predict(readings)[0])
    probabilities = model.predict_proba(readings)[0]
    classes = list(model.named_steps["classifier"].classes_)
    score = float(probabilities[classes.index(label)])
    st.subheader("Neer model result")
    if label == 1:
        st.success("The model assigned this input to the dataset's potable class.")
    else:
        st.warning("The model assigned this input to the dataset's not-potable class.")
    st.metric("Model score for selected class", f"{score:.1%}")
    st.dataframe(readings, use_container_width=True)
    st.caption("This score is not the probability that real water is safe. This is a dataset classification using simulated readings, not a laboratory test or safety certification.")

st.divider()
st.header("How the model performed")
st.write("Metrics below describe performance on a held-out test split from this dataset only.")
chart = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 score"],
    "Score": [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
})
st.bar_chart(chart.set_index("Metric"))

importance = pd.DataFrame({
    "Measurement": FEATURES,
    "Relative model importance": model.named_steps["classifier"].feature_importances_
}).sort_values("Relative model importance", ascending=False)
st.subheader("Which inputs influenced the model most?")
st.bar_chart(importance.set_index("Measurement"))

st.divider()
st.header("About the dataset")
st.write("This prototype uses a public mirror of the commonly used Water Potability dataset. It contains historical sample records and labels; it is not a live sensor feed.")
st.markdown("[View dataset source](https://github.com/sudesokin/Water-Quality-Analysis-and-Forecasting/blob/main/dataset/water_potability.csv)")
st.warning("The dataset's Solids feature is kept under its original name and is not presented as a direct TDS sensor measurement. These four inputs cannot identify particular heavy metals or microorganisms.")
st.caption("NEER · Educational research prototype · Simulated inputs")
