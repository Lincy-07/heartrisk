"""Loads the exported model and returns a prediction + SHAP explanation for one patient."""
import os, json
import numpy as np, pandas as pd, joblib, shap

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "model")
MODEL = joblib.load(os.path.join(MODEL_DIR, "heart_risk_pipeline.joblib"))
META = json.load(open(os.path.join(MODEL_DIR, "model_meta.json")))
FEATURES = META["features"]
_EXPL = shap.TreeExplainer(MODEL.named_steps["rf"])

LABELS = {"male": "Sex (male=1)", "age": "Age", "education": "Education level", "currentSmoker": "Current smoker",
          "cigsPerDay": "Cigarettes/day", "BPMeds": "On BP medication", "prevalentStroke": "Prior stroke",
          "prevalentHyp": "Hypertensive", "diabetes": "Diabetes", "totChol": "Total cholesterol",
          "sysBP": "Systolic BP", "diaBP": "Diastolic BP", "BMI": "BMI", "heartRate": "Resting heart rate", "glucose": "Glucose"}


def predict_one(record: dict) -> dict:
    X = pd.DataFrame([{f: record.get(f, np.nan) for f in FEATURES}])[FEATURES]
    p = float(MODEL.predict_proba(X)[0, 1])
    Xi = pd.DataFrame(MODEL.named_steps["imp"].transform(X), columns=FEATURES)
    sv = _EXPL.shap_values(Xi)
    sv = sv[1][0] if isinstance(sv, list) else (sv[0, :, 1] if sv.ndim == 3 else sv[0])
    contrib = sorted(zip(FEATURES, sv), key=lambda t: -abs(t[1]))
    thr = META["threshold"]
    return {
        "probability_10y_chd": round(p, 4),
        "flagged_elevated_risk": bool(p >= thr),
        "decision_threshold": round(thr, 3),
        "population_base_rate": round(META["positive_rate_train"], 3),
        "top_contributions": [{"feature": f, "label": LABELS.get(f, f), "value": None if pd.isna(X[f].iloc[0]) else float(X[f].iloc[0]),
                               "shap_effect": round(float(v), 4)} for f, v in contrib[:6]],
        "disclaimer": "Educational model trained on the Framingham cohort. Not a medical device or diagnosis.",
    }
