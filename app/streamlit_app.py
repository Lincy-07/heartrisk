"""Streamlit dashboard.  Run from project root:  streamlit run app/streamlit_app.py"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from app.serving import predict_one, META, LABELS
from app import storage

st.set_page_config(page_title="Heart Risk Dashboard", page_icon="❤️", layout="wide")
THR = META["threshold"]; BASE = META["positive_rate_train"]
COL = {"Lower": "#54A24B", "Elevated": "#F58518", "High": "#E45756"}
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")


def band(p):
    return "Lower" if p < THR else ("Elevated" if p < 2 * THR else "High")


def load_csv(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def show_fig(name, caption=None):
    p = os.path.join(FIG, name)
    if os.path.exists(p): st.image(p, caption=caption, width="stretch")


def shap_chart(contribs, title="Top factors behind this prediction (SHAP)"):
    df = pd.DataFrame(contribs).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3))
    lab = [f"{l} = {v:.1f}" if v is not None else l for l, v in zip(df["label"], df["value"])]
    ax.barh(lab, df["shap_effect"], color=["#E45756" if v > 0 else "#4C78A8" for v in df["shap_effect"]])
    ax.axvline(0, color="k", lw=.7); ax.set_xlabel("Effect on risk (red = raises, blue = lowers)"); ax.set_title(title, fontsize=10)
    plt.tight_layout(); return fig


def gauge(p):
    fig, ax = plt.subplots(figsize=(6, 1.1))
    ax.barh([0], [THR], color=COL["Lower"], height=.5); ax.barh([0], [THR], left=THR, color=COL["Elevated"], height=.5)
    ax.barh([0], [max(0.6 - 2 * THR, .05)], left=2 * THR, color=COL["High"], height=.5)
    ax.axvline(p, color="black", lw=3); ax.text(p, .45, f"{p*100:.1f}%", ha="center", fontweight="bold")
    ax.axvline(BASE, color="gray", ls="--", lw=1); ax.text(BASE, -.55, "avg", ha="center", fontsize=7, color="gray")
    ax.set_xlim(0, .6); ax.set_ylim(-.7, .8); ax.set_yticks([]); ax.set_xlabel("Predicted 10-year risk")
    plt.tight_layout(); return fig


def compare_chart(feat):
    keys = ["age", "sysBP", "diaBP", "totChol", "BMI", "glucose", "heartRate", "cigsPerDay"]
    med = META["feature_medians"]
    ratio = [feat[k] / med[k] if med[k] else 1 for k in keys]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([LABELS[k].replace("Cigarettes/day", "Cigs/day") for k in keys], ratio, color=["#E45756" if r > 1.1 else "#4C78A8" if r < .9 else "#9ecae9" for r in ratio])
    ax.axhline(1, color="k", lw=.8, ls="--"); ax.set_ylabel("Value ÷ dataset median"); ax.set_title("This patient vs a typical patient", fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=8); plt.tight_layout(); return fig


st.title("❤️ Heart Disease Risk Dashboard")
st.caption("AutoML project • Framingham data • educational demo, **not a medical device**. Patient names entered here are stored only in a local file (data/patients.db).")
tabs = st.tabs(["🩺 New assessment", "🗂️ Patient records", "📊 Analytics", "🏆 Model performance", "📄 Model card"])

# ------------------------------------------------------------------ 1. assessment
with tabs[0]:
    with st.form("assess"):
        n1, n2 = st.columns([2, 3])
        name = n1.text_input("Patient name", placeholder="e.g. Priya Nair")
        notes = n2.text_input("Notes (optional)")
        c1, c2, c3 = st.columns(3)
        with c1:
            male = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x else "Female", index=1)
            age = st.slider("Age", 30, 80, 55)
            smoker = st.selectbox("Current smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cigs = st.slider("Cigarettes per day", 0, 70, 0)
            edu = st.selectbox("Education level (1-4)", [1, 2, 3, 4], index=1)
        with c2:
            sbp = st.slider("Systolic BP", 80, 250, 130)
            dbp = st.slider("Diastolic BP", 40, 140, 82)
            hyp = st.selectbox("Hypertensive", [0, 1], format_func=lambda x: "Yes" if x else "No")
            bpm = st.selectbox("On BP medication", [0, 1], format_func=lambda x: "Yes" if x else "No")
            stroke = st.selectbox("Prior stroke", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with c3:
            chol = st.slider("Total cholesterol", 100, 600, 235)
            bmi = st.slider("BMI", 15.0, 55.0, 26.5)
            hr = st.slider("Resting heart rate", 40, 140, 75)
            glu = st.slider("Glucose", 40, 350, 80)
            dia = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
        go = st.form_submit_button("Predict and save patient", type="primary")
    if go:
        feat = dict(male=male, age=age, education=edu, currentSmoker=smoker, cigsPerDay=cigs, BPMeds=bpm, prevalentStroke=stroke,
                    prevalentHyp=hyp, diabetes=dia, totChol=chol, sysBP=sbp, diaBP=dbp, BMI=bmi, heartRate=hr, glucose=glu)
        r = predict_one(feat); b = band(r["probability_10y_chd"])
        pid = storage.add_patient(name, feat, r, b, notes)
        st.session_state["last"] = (pid, name or "Anonymous", feat, r, b)
    if "last" in st.session_state:
        pid, nm, feat, r, b = st.session_state["last"]
        st.success(f"Saved as **P{pid:04d}** – {nm}")
        a, bb, c, d = st.columns(4)
        a.metric("10-year CHD probability", f"{r['probability_10y_chd']*100:.1f}%")
        bb.metric("Dataset average", f"{BASE*100:.1f}%")
        c.metric("Risk band", b)
        d.metric("Flagged (≥ %.2f)" % THR, "YES" if r["flagged_elevated_risk"] else "No")
        st.pyplot(gauge(r["probability_10y_chd"]))
        g1, g2 = st.columns(2)
        g1.pyplot(shap_chart(r["top_contributions"])); g2.pyplot(compare_chart(feat))
        st.caption("Bands are a display convention: Lower < threshold, Elevated < 2×threshold, High above. The threshold (%.2f) was chosen on training data to balance recall and precision." % THR)
        st.warning(r["disclaimer"])

# ------------------------------------------------------------------ 2. records
with tabs[1]:
    df = storage.all_patients()
    cA, cB, cC = st.columns([3, 1, 1])
    q = cA.text_input("Search by name or ID", key="search")
    if cB.button("Add 20 demo patients"):
        te = pd.read_csv(os.path.join(ROOT, "data", "framingham.csv")).sample(20, random_state=int(pd.Timestamp.now().timestamp()) % 1000)
        names = ["Arjun Sharma", "Priya Nair", "Rahul Verma", "Sunita Patel", "Karthik Raja", "Meena Krishnan", "Vijay Kumar", "Anitha Reddy", "Suresh Iyer", "Deepa Menon",
                 "Ravi Shankar", "Lakshmi Das", "Mohan Pillai", "Geetha Balan", "Arun Nair", "Kavitha Rao", "Siva Subramanian", "Uma Devi", "Prakash Naidu", "Radha Gopal"]
        for nm, (_, row) in zip(names, te.iterrows()):
            f = {k: (None if pd.isna(row[k]) else float(row[k])) for k in META["features"]}
            rr = predict_one(f); storage.add_patient(nm, {k: (v if v is not None else META["feature_medians"][k]) for k, v in f.items()}, rr, band(rr["probability_10y_chd"]), "demo record", "demo")
        st.rerun()
    if cC.button("Delete demo records"):
        storage.clear_all(only_demo=True); st.rerun()
    if df.empty:
        st.info("No patients yet. Add one in the first tab or click **Add 20 demo patients**.")
    else:
        view = df if not q else df[df.name.str.contains(q, case=False) | df.patient_code.str.contains(q, case=False)]
        show = view[["patient_code", "name", "created_at", "age", "sex", "probability", "risk_band", "flagged", "notes", "source"]].copy()
        show["probability"] = (show["probability"] * 100).round(1); show["flagged"] = show["flagged"].map({1: "Yes", 0: "No"})
        show.columns = ["ID", "Name", "Date", "Age", "Sex", "Risk %", "Band", "Flagged", "Notes", "Source"]
        st.dataframe(show, width="stretch", hide_index=True)
        st.download_button("⬇ Download records (CSV)", show.to_csv(index=False).encode(), "patients.csv", "text/csv")
        st.divider(); st.subheader("Open a patient")
        pick = st.selectbox("Patient", view.patient_code + " – " + view.name) if len(view) else None
        if pick:
            row = view[view.patient_code == pick.split(" – ")[0]].iloc[0]
            feats = json.loads(row.features); tf = json.loads(row.top_factors)
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk", f"{row.probability*100:.1f}%"); m2.metric("Band", row.risk_band); m3.metric("Recorded", row.created_at.strftime("%d %b %Y %H:%M"))
            p1, p2 = st.columns(2); p1.pyplot(shap_chart(tf)); p2.pyplot(compare_chart(feats))
            if st.button("🗑 Delete this patient"):
                storage.delete_patient(row.id); st.rerun()

# ------------------------------------------------------------------ 3. analytics
with tabs[2]:
    df = storage.all_patients()
    if df.empty:
        st.info("Add patients (or demo patients) to see analytics.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Patients", len(df)); k2.metric("Average risk", f"{df.probability.mean()*100:.1f}%")
        k3.metric("Flagged", f"{int(df.flagged.sum())} ({df.flagged.mean()*100:.0f}%)"); k4.metric("Highest risk", f"{df.probability.max()*100:.1f}%")
        r1, r2, r3 = st.columns(3)
        with r1:
            f, ax = plt.subplots(figsize=(4.4, 3.2)); ax.hist(df.probability * 100, bins=12, color="#4C78A8", edgecolor="white")
            ax.axvline(THR * 100, color="red", ls="--", label="flag threshold"); ax.set_xlabel("Predicted risk %"); ax.set_ylabel("Patients"); ax.set_title("Risk distribution"); ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(f)
        with r2:
            bc = df.risk_band.value_counts().reindex(["Lower", "Elevated", "High"]).fillna(0)
            f, ax = plt.subplots(figsize=(4.4, 3.2)); ax.pie([v for v in bc if v > 0], labels=[f"{k} ({int(v)})" for k, v in bc.items() if v > 0], colors=[COL[k] for k, v in bc.items() if v > 0], autopct="%1.0f%%", startangle=90)
            ax.set_title("Risk bands"); plt.tight_layout(); st.pyplot(f)
        with r3:
            ag = df.groupby(pd.cut(df.age, [0, 44, 54, 64, 120], labels=["<45", "45-54", "55-64", "65+"]), observed=True).probability.mean() * 100
            f, ax = plt.subplots(figsize=(4.4, 3.2)); ax.bar(ag.index.astype(str), ag.values, color="#54A24B"); ax.set_ylabel("Mean risk %"); ax.set_title("Average risk by age group"); plt.tight_layout(); st.pyplot(f)
        r4, r5, r6 = st.columns(3)
        with r4:
            f, ax = plt.subplots(figsize=(4.4, 3.2)); sc = ax.scatter(df.age, df.probability * 100, c=df.probability, cmap="RdYlGn_r", edgecolor="k", s=45)
            ax.set_xlabel("Age"); ax.set_ylabel("Risk %"); ax.set_title("Age vs risk"); plt.tight_layout(); st.pyplot(f)
        with r5:
            sx = df.groupby("sex").agg(avg=("probability", "mean"), flagged=("flagged", "mean")) * 100
            f, ax = plt.subplots(figsize=(4.4, 3.2)); x = np.arange(len(sx)); ax.bar(x - .2, sx.avg, .4, label="avg risk %", color="#4C78A8"); ax.bar(x + .2, sx.flagged, .4, label="% flagged", color="#E45756")
            ax.set_xticks(x); ax.set_xticklabels(sx.index); ax.legend(fontsize=8); ax.set_title("Risk by sex"); plt.tight_layout(); st.pyplot(f)
        with r6:
            top = pd.Series([t["label"] for tf in df.top_factors.map(json.loads) for t in tf[:3]]).value_counts().head(7).iloc[::-1]
            f, ax = plt.subplots(figsize=(4.4, 3.2)); ax.barh(top.index, top.values, color="#F58518"); ax.set_title("Most common top-3 risk drivers"); ax.set_xlabel("Patients"); plt.tight_layout(); st.pyplot(f)
        daily = df.groupby(df.created_at.dt.date).size()
        f, ax = plt.subplots(figsize=(10, 2.6)); ax.plot(daily.index.astype(str), daily.values, marker="o", color="#4C78A8"); ax.set_title("Assessments per day"); ax.set_ylabel("Count"); plt.xticks(rotation=30, fontsize=8); plt.tight_layout(); st.pyplot(f)
        st.subheader("Highest-risk patients")
        st.dataframe(df.nlargest(5, "probability")[["patient_code", "name", "age", "sex", "probability", "risk_band"]].assign(probability=lambda d: (d.probability * 100).round(1)), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 4. model performance
with tabs[3]:
    st.subheader("How every approach performed (same test set)")
    fc = load_csv("final_comparison.csv")
    if fc is not None: st.dataframe(fc.rename(columns={"model": "Model", "cv_auc": "CV AUC", "test_auc": "Test AUC", "time_s": "Train time (s)"}), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a: show_fig("final_comparison.png", "Test ROC-AUC of all approaches")
    with b: show_fig("roc_all_models.png", "ROC curves")
    st.subheader("Hyper-parameter tuning: Grid vs Random vs Optuna")
    a, b = st.columns(2)
    with a: show_fig("tuning_comparison.png")
    with b: show_fig("optuna_convergence.png", "Optuna convergence")
    tc = load_csv("tuning_comparison.csv")
    if tc is not None: st.dataframe(tc, hide_index=True, width="stretch")
    st.subheader("Leakage experiment: AutoML exploits a leaky feature")
    show_fig("leakage_experiment.png"); lk = load_csv("leakage_results.csv")
    if lk is not None: st.dataframe(lk.round(3), hide_index=True, width="stretch")
    st.subheader("Explainability, calibration and fairness")
    a, b = st.columns(2)
    with a: show_fig("shap_beeswarm.png", "SHAP: what drives predictions")
    with b: show_fig("calibration.png", "Calibration")
    sb = load_csv("subgroup_bias.csv")
    if sb is not None: st.dataframe(sb, hide_index=True, width="stretch")
    ib = load_csv("imbalance_check.csv")
    if ib is not None: st.dataframe(ib, hide_index=True, width="stretch")
    st.subheader("Workflow"); show_fig("architecture.png"); show_fig("eda_overview.png")

# ------------------------------------------------------------------ 5. model card
with tabs[4]:
    mc = os.path.join(ROOT, "MODEL_CARD.md")
    st.markdown(open(mc, encoding="utf-8").read() if os.path.exists(mc) else "MODEL_CARD.md not found")
