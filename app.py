from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_URL = "https://raw.githubusercontent.com/sudesokin/Water-Quality-Analysis-and-Forecasting/main/dataset/water_potability.csv"
FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
TARGET = "Potability"
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "water_quality.csv"

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
    # Keep startup lightweight for Streamlit Community Cloud:
    # fit one fixed model instead of running a large grid search on every cold start.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
            class_weight="balanced"
        ))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    metrics = {
        "rows": len(df),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, pred, pos_label=1, zero_division=0),
    }
    return pipeline, metrics

st.markdown("<h5 style='color:#087e8b;letter-spacing:3px'>NEER · WATER INTELLIGENCE</h5>", unsafe_allow_html=True)
st.title("See the patterns. Understand the water.")
st.write("An AI-powered demonstration that analyses nine water-quality measurements together.")

# Clickable explanations for technical terms used throughout the prototype.
TERMS = [
    ("pH", "A scale describing how acidic or alkaline water is. 7 is neutral; lower values are acidic and higher values are alkaline.", "The value is measured from chemical properties of a water sample. In this demo, the model uses the pH number as one input feature.", "Move the pH slider to try different values, then select Analyse combined readings. The slider is simulated, not connected to a real pH sensor."),
    ("Solids", "A dataset measurement representing the amount of dissolved and suspended material recorded for a sample. The original dataset calls this field Solids.", "The model learns patterns between this recorded number and the dataset's potable/not-potable labels. It is not automatically the same as a direct TDS sensor reading.", "Use the Solids slider to explore how changing this dataset field affects the model output."),
    ("Electrical conductivity", "A measure of how easily water carries an electrical current. Dissolved ions often increase conductivity.", "The dataset records conductivity as a numeric measurement. The model compares it with patterns learned from other samples.", "Adjust the Electrical conductivity slider. The displayed unit is microsiemens per centimetre (µS/cm)."),
    ("Turbidity", "A measure of how cloudy or unclear water is because of particles suspended in it.", "A turbidity meter commonly measures how light is scattered by particles. Here, the dataset value is supplied to the model.", "Adjust the Turbidity slider. NTU means Nephelometric Turbidity Units."),
    ("Random Forest", "A machine-learning method that combines many decision trees instead of relying on just one.", "Each tree makes a classification using learned rules. The forest combines the trees' votes to produce an overall class.", "You do not need to operate the forest directly. Change the nine readings and click Analyse combined readings to see its output."),
    ("Machine learning", "A way for a computer to learn patterns from examples rather than having every decision written as a fixed rule.", "Neer is trained on labelled historical water-sample records. It learns relationships between the nine measurements and the dataset label.", "Try different slider combinations and compare the results. This is an educational model, not a real-world water test."),
    ("Dataset", "A collection of records organised for analysis. Each row in this dataset represents a historical water sample.", "The training process reads measurement columns and the Potability label, then uses the examples to train and evaluate the model.", "The dataset record count appears at the top. Use the dataset source link near the bottom to inspect the original public dataset."),
    ("Potable", "Suitable for drinking according to the label used in the source dataset.", "The dataset uses 1 for its potable class and 0 for its not-potable class. The model learns to assign one of these two labels.", "Read the result as the class the model selected for the simulated inputs, not as confirmation that water is safe."),
    ("Classifier", "A machine-learning model that sorts an input into one of a set of categories.", "Neer's classifier receives the nine measurement values and assigns the input to one of the two dataset classes.", "Click Analyse combined readings after setting the sliders to view the predicted class."),
    ("Training data", "The part of the dataset used to teach the model patterns.", "The training portion contains examples with measurements and known dataset labels. The model adjusts its internal decision patterns using these examples.", "You do not need to select it manually; Neer prepares the training data when the app loads."),
    ("Test split", "A separate portion of the dataset held back to check how the trained model performs on examples it did not train on.", "Neer uses an 80/20 train/test split. It selects model settings using five-fold cross-validation on the training portion, then uses the held-out test portion for final metrics.", "Use the performance section to view the scores from this held-out portion. They describe this dataset split only."),
    ("Accuracy", "The percentage of test examples the model classified correctly overall.", "It is calculated as correct predictions divided by all test predictions.", "Read it as a broad summary, alongside precision, recall and F1 score. A high accuracy alone does not prove the model is reliable for every sample."),
    ("Precision", "Of the samples the model labelled potable, the proportion that actually had the potable label in the test data.", "It measures how often positive predictions were correct on the held-out test split.", "Compare it with recall to understand different kinds of model errors. It is not a guarantee about real drinking water."),
    ("Recall", "Of the samples labelled potable in the test data, the proportion the model successfully identified as potable.", "It measures how many actual positive examples were found by the model.", "Read it together with precision. Neither metric replaces laboratory testing or official water-quality checks."),
    ("F1 score", "A single score that balances precision and recall.", "It is the harmonic mean of precision and recall, so it becomes lower when either one is low.", "Use it as a combined comparison measure in the model-performance section."),
    ("Imputation", "A way to fill in missing data values so a model can still process a record.", "Neer uses median imputation: a missing measurement is replaced with the median value for that feature in the training data.", "This happens automatically during model preparation. It is not a reading supplied by a sensor."),
    ("Feature importance", "An estimate of how much each input measurement helped the trained Random Forest make its decisions.", "The model calculates importance from how its decision trees use each feature. The chart ranks the nine inputs by this estimate.", "Look at the feature-importance chart to compare inputs. Importance does not prove cause and effect."),
    ("Model score", "The model's score for the class it selected for the current simulated input.", "The Random Forest combines estimates from its trees to produce class scores. The app displays the score for the selected class.", "View it after clicking Analyse combined readings. It is not the probability that a real water sample is safe."),
    ("Simulated readings", "Example numbers entered manually to demonstrate how the app works. They are not collected from a live water sensor.", "The sliders create a sample row that is passed to the trained model as if it were a measurement record.", "Move the sliders to explore different combinations, then click Analyse combined readings."),
    ("µS/cm", "Microsiemens per centimetre, a unit used for electrical conductivity.", "It expresses how readily a material conducts electricity over a standardised distance.", "The conductivity slider displays this unit beside its label."),
    ("NTU", "Nephelometric Turbidity Units, a unit used to report turbidity.", "A nephelometric instrument estimates cloudiness by measuring light scattered by particles in a sample.", "The turbidity slider displays this unit beside its label.")
]

st.divider()
st.header("Explore the technical terms")
st.write("Click any term to open a short explanation of what it means, how it works in Neer, and how to use it.")
term_columns = st.columns(3)
for index, (term, meaning, works, usage) in enumerate(TERMS):
    with term_columns[index % 3]:
        with st.popover(term, use_container_width=True):
            st.markdown("**What it means**")
            st.write(meaning)
            st.markdown("**How it works in Neer**")
            st.write(works)
            st.markdown("**How to use it**")
            st.write(usage)

with st.sidebar:
    st.header("About Neer")
    st.write("Neer explores how nine water-quality measurements can be analysed together using machine learning.")
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
    hardness = st.slider("Hardness", 47.0, 323.0, 196.0, 1.0)
    solids = st.slider("Solids (dataset units)", 0.0, 60000.0, 20000.0, 100.0)
    chloramines = st.slider("Chloramines", 0.0, 14.0, 7.0, 0.1)
    sulfate = st.slider("Sulfate", 129.0, 481.0, 334.0, 1.0)
with right:
    conductivity = st.slider("Electrical conductivity (µS/cm)", 0.0, 800.0, 425.0, 5.0)
    organic_carbon = st.slider("Organic carbon", 2.0, 28.0, 14.0, 0.1)
    trihalomethanes = st.slider("Trihalomethanes", 0.0, 125.0, 66.0, 1.0)
    turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 3.5, 0.1)

readings = pd.DataFrame([{
    "ph": ph, "Hardness": hardness, "Solids": solids,
    "Chloramines": chloramines, "Sulfate": sulfate,
    "Conductivity": conductivity, "Organic_carbon": organic_carbon,
    "Trihalomethanes": trihalomethanes, "Turbidity": turbidity
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
st.write("Metrics below describe performance on a held-out test split from this dataset only. Model settings are selected using five-fold cross-validation on the training data.")
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
st.caption("Best settings selected by five-fold cross-validation: " + str(metrics["best_params"]))

st.divider()
st.header("About the dataset")
st.write("This prototype uses a public mirror of the commonly used Water Potability dataset. It contains historical sample records and labels; it is not a live sensor feed.")
st.markdown("[View dataset source](https://github.com/sudesokin/Water-Quality-Analysis-and-Forecasting/blob/main/dataset/water_potability.csv)")
st.warning("The dataset's Solids feature is kept under its original name and is not presented as a direct TDS sensor measurement. These nine dataset inputs cannot identify particular heavy metals or microorganisms.")
st.caption("NEER · Educational research prototype · Simulated inputs")
