"""Web dashboard + REST API (FastAPI).

Run from the project root:   uvicorn app.api:app --port 5000
Dashboard:  http://127.0.0.1:5000/        API docs: http://127.0.0.1:5000/docs
"""
import os, io, json, random
from datetime import datetime, timedelta
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.serving import predict_one, META, LABELS
from app import storage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG, FRONT = (os.path.join(ROOT, d) for d in ("results", "figures", "frontend"))
THR = META["threshold"]; BASE = META["positive_rate_train"]

app = FastAPI(title="Heart Risk Dashboard API", version="2.0",
              description="10-year coronary heart disease risk (Framingham, AutoML project). Educational use only.")
app.mount("/figures", StaticFiles(directory=FIG), name="figures")
app.mount("/vendor", StaticFiles(directory=os.path.join(FRONT, "vendor")), name="vendor")


def dump(m):
    """Works with Pydantic v2 (model_dump) and v1 (dict)."""
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()


def band(p):
    return "Lower" if p < THR else ("Elevated" if p < 2 * THR else "High")


class Patient(BaseModel):
    male: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    age: int = Field(..., ge=30, le=80)
    education: float | None = Field(None, ge=1, le=4)
    currentSmoker: int = Field(..., ge=0, le=1)
    cigsPerDay: float | None = Field(None, ge=0, le=80)
    BPMeds: float | None = Field(None, ge=0, le=1)
    prevalentStroke: int = Field(0, ge=0, le=1)
    prevalentHyp: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    totChol: float | None = Field(None, ge=100, le=700)
    sysBP: float = Field(..., ge=80, le=300)
    diaBP: float = Field(..., ge=40, le=150)
    BMI: float | None = Field(None, ge=14, le=60)
    heartRate: float | None = Field(None, ge=40, le=150)
    glucose: float | None = Field(None, ge=40, le=400)


class PatientIn(Patient):
    name: str = Field("Anonymous", max_length=80)
    notes: str = Field("", max_length=200)


# ------------------------------------------------------------------ pages
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(FRONT, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model_name"]}


@app.get("/model-info")
def info():
    return {k: META[k] for k in ["model_name", "features", "threshold", "cv_auc", "test_metrics_tuned_threshold", "train_rows"]}


@app.post("/predict")
def predict(p: Patient):
    """Stateless prediction (nothing is saved)."""
    return predict_one(dump(p))


# ------------------------------------------------------------------ patients
@app.post("/api/predict")
def predict_and_save(p: PatientIn):
    d = dump(p); name, notes = d.pop("name"), d.pop("notes")
    r = predict_one(d); b = band(r["probability_10y_chd"])
    feats = {k: (v if v is not None else META["feature_medians"][k]) for k, v in d.items()}
    pid = storage.add_patient(name, feats, r, b, notes)
    med = META["feature_medians"]
    return {"patient_id": f"P{pid:04d}", "name": name or "Anonymous", "risk_band": b, "threshold_band_edges": [THR, 2 * THR],
            "population_base_rate": BASE, "features": feats, "medians": med, **r}


def _row(r, full=False):
    o = {"id": int(r.id), "code": r.patient_code, "name": r["name"], "date": r.created_at.isoformat(), "age": int(r.age), "sex": r.sex,
         "probability": round(float(r.probability) * 100, 1), "band": r.risk_band, "flagged": bool(r.flagged), "notes": r.notes, "source": r.source}
    if full:
        o["features"] = json.loads(r.features); o["top_factors"] = json.loads(r.top_factors); o["medians"] = META["feature_medians"]
    return o


@app.get("/api/patients")
def patients(page: int = 1, per_page: int = 10, risk: str = "all", q: str = ""):
    df = storage.all_patients()
    if len(df):
        if risk != "all": df = df[df.risk_band.str.lower() == risk.lower()]
        if q: df = df[df["name"].str.contains(q, case=False) | df.patient_code.str.contains(q, case=False)]
    total = len(df); start = (page - 1) * per_page
    rows = [_row(r) for _, r in df.iloc[start:start + per_page].iterrows()]
    return {"patients": rows, "total": total, "page": page, "per_page": per_page, "pages": max(1, -(-total // per_page))}


@app.get("/api/patients.csv")
def patients_csv():
    df = storage.all_patients()
    out = pd.DataFrame([_row(r) for _, r in df.iterrows()])
    return Response(out.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=patients.csv"})


@app.get("/api/patient/{pid}")
def patient(pid: int):
    df = storage.all_patients(); m = df[df.id == pid]
    if m.empty: raise HTTPException(404, "Patient not found")
    return _row(m.iloc[0], full=True)


@app.delete("/api/patient/{pid}")
def delete_patient(pid: int):
    storage.delete_patient(pid); return {"deleted": pid}


DEMO_NAMES = ["Arjun Sharma", "Priya Nair", "Rahul Verma", "Sunita Patel", "Karthik Raja", "Meena Krishnan", "Vijay Kumar", "Anitha Reddy", "Suresh Iyer",
              "Deepa Menon", "Ravi Shankar", "Lakshmi Das", "Mohan Pillai", "Geetha Balan", "Arun Nair", "Kavitha Rao", "Siva Subramanian", "Uma Devi",
              "Prakash Naidu", "Radha Gopal", "Manoj Tiwari", "Sneha Joshi", "Ashok Gupta", "Pooja Singh", "Ramesh Chand"]


@app.post("/api/demo")
def add_demo(n: int = 20):
    """Adds n fictional demo patients (real Framingham rows with made-up names), spread over the last 14 days."""
    n = max(1, min(n, 25))
    rows = pd.read_csv(os.path.join(ROOT, "data", "framingham.csv")).sample(n)
    names = random.sample(DEMO_NAMES, n)
    for nm, (_, row) in zip(names, rows.iterrows()):
        f = {k: (META["feature_medians"][k] if pd.isna(row[k]) else float(row[k])) for k in META["features"]}
        r = predict_one(f)
        when = datetime.now() - timedelta(days=random.randint(0, 13), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        storage.add_patient(nm, f, r, band(r["probability_10y_chd"]), "demo record", "demo", created_at=when)
    return {"added": n}


@app.delete("/api/demo")
def delete_demo():
    storage.clear_all(only_demo=True); return {"ok": True}


# ------------------------------------------------------------------ analytics
@app.get("/api/analytics")
def analytics():
    df = storage.all_patients()
    fi = pd.read_csv(os.path.join(RES, "feature_importance_shap_perm.csv"), index_col=0)["mean_|SHAP|"]
    gfi = {LABELS.get(k, k): round(float(v), 4) for k, v in fi.items()}
    base = {"model_importance": gfi, "threshold": THR, "base_rate": BASE}
    if df.empty:
        return {"empty": True, **base}
    p = df.probability * 100
    bins = np.arange(0, 65, 5); hist, _ = np.histogram(p.clip(upper=59.9), bins=bins)
    ag = df.groupby(pd.cut(df.age, [0, 44, 54, 64, 120], labels=["<45", "45-54", "55-64", "65+"]), observed=False).probability.mean() * 100
    sx = df.groupby("sex").agg(avg=("probability", "mean"), flagged=("flagged", "mean"), n=("id", "count")) * [100, 100, 1]
    days = pd.date_range(end=pd.Timestamp.now().normalize(), periods=14)
    daily = df.groupby(df.created_at.dt.normalize()).size().reindex(days, fill_value=0)
    drivers = pd.Series([t["label"] for tf in df.top_factors.map(json.loads) for t in tf[:3]]).value_counts().head(8)
    bc = df.risk_band.value_counts()
    top = df.nlargest(5, "probability")
    return {**base, "empty": False,
            "summary": {"total": len(df), "high": int(bc.get("High", 0)), "elevated": int(bc.get("Elevated", 0)), "lower": int(bc.get("Lower", 0)),
                        "avg_risk": round(float(p.mean()), 1), "flagged": int(df.flagged.sum()), "max_risk": round(float(p.max()), 1)},
            "histogram": {"labels": [f"{a}-{a+5}%" for a in bins[:-1]], "counts": hist.tolist()},
            "bands": {"Lower": int(bc.get("Lower", 0)), "Elevated": int(bc.get("Elevated", 0)), "High": int(bc.get("High", 0))},
            "age_risk": {k: (None if pd.isna(v) else round(float(v), 1)) for k, v in ag.items()},
            "sex": {k: {"avg": round(float(v["avg"]), 1), "flagged": round(float(v["flagged"]), 1), "n": int(v["n"])} for k, v in sx.iterrows()},
            "scatter": [{"x": int(a), "y": round(float(b), 1)} for a, b in zip(df.age, p)],
            "daily": [{"date": d.strftime("%d %b"), "count": int(c)} for d, c in daily.items()],
            "drivers": {k: int(v) for k, v in drivers.items()},
            "top": [_row(r) for _, r in top.iterrows()]}


# ------------------------------------------------------------------ model results
def _csv(name):
    f = os.path.join(RES, name)
    return pd.read_csv(f).round(4).replace({np.nan: None}).to_dict("records") if os.path.exists(f) else []


@app.get("/api/model")
def model():
    rep = json.load(open(os.path.join(RES, "final_model_report.json")))
    return {"meta": {k: META[k] for k in ["model_name", "threshold", "cv_auc", "train_rows", "test_rows", "hyperparameters"]},
            "headline": rep["final_model_test_tuned_thr"], "default_thr": rep["final_model_test_default_thr"],
            "comparison": _csv("final_comparison.csv"), "tuning": _csv("tuning_comparison.csv"), "bootstrap": _csv("test_auc_bootstrap_ci.csv"),
            "leakage": pd.read_csv(os.path.join(RES, "leakage_results.csv"), index_col=0).round(4).reset_index().rename(columns={"index": "model"}).replace({np.nan: None}).to_dict("records"),
            "subgroup": _csv("subgroup_bias.csv"), "imbalance": _csv("imbalance_check.csv"), "overfitting": _csv("overfitting_check.csv"),
            "importance": pd.read_csv(os.path.join(RES, "feature_importance_shap_perm.csv"), index_col=0).round(4).reset_index().rename(columns={"index": "feature"}).to_dict("records")}


@app.get("/api/modelcard", response_class=PlainTextResponse)
def modelcard():
    return open(os.path.join(ROOT, "MODEL_CARD.md"), encoding="utf-8").read()
