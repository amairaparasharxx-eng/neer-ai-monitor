"""Train Neer's Random Forest model and report held-out test metrics."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

URL = "https://raw.githubusercontent.com/sudesokin/Water-Quality-Analysis-and-Forecasting/main/dataset/water_potability.csv"
FEATURES = ["ph", "Solids", "Conductivity", "Turbidity"]
ROOT = Path(__file__).resolve().parent

def main():
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "water_quality.csv"
    if not csv_path.exists():
        print("Downloading public water-potability dataset...")
        df = pd.read_csv(URL)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    required = FEATURES + ["Potability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()
    df["Potability"] = pd.to_numeric(df["Potability"], errors="coerce")
    df = df.dropna(subset=["Potability"])
    df = df[df["Potability"].isin([0, 1])]
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df["Potability"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)
    }
    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "neer_model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    print("Saved model to model/neer_model.joblib")

if __name__ == "__main__":
    main()
