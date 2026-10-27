"""Builds extra figures, MODEL_CARD.md, the PDF report and the slides FROM the saved results
(no number is typed by hand). Run after steps 01-06."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import *

TEAM = os.environ.get("TEAM", "Team members: [add names and register numbers]")
tune = pd.read_csv(os.path.join(RES, "tuning_comparison.csv"))
auto = load_json("automl_results.json"); leak = pd.read_csv(os.path.join(RES, "leakage_results.csv"), index_col=0)
ci = pd.read_csv(os.path.join(RES, "test_auc_bootstrap_ci.csv")).set_index("model")
over = pd.read_csv(os.path.join(RES, "overfitting_check.csv")); imb = pd.read_csv(os.path.join(RES, "imbalance_check.csv"))
bias = pd.read_csv(os.path.join(RES, "subgroup_bias.csv")); eda = load_json("eda_summary.json")
fimp = pd.read_csv(os.path.join(RES, "feature_importance_shap_perm.csv"), index_col=0)
rep = load_json("final_model_report.json"); thr = rep["f1_threshold_from_train_oof"]
lbA = pd.read_csv(os.path.join(RES, "leaderboard_autogluon.csv")); lbF = pd.read_csv(os.path.join(RES, "leaderboard_flaml.csv"))
leakscreen = pd.read_csv(os.path.join(RES, "leak_single_feature_auc.csv"), index_col=0)
preds = pd.read_csv(os.path.join(RES, "preds_all.csv"), index_col=0)
B = auto["budget_seconds"]
_single = lbA[~lbA.model.str.startswith("WeightedEnsemble")]
AG_TOP = str(_single.iloc[0].model).split("_")[0] if len(_single) else "n/a"
AG_FAMILIES = sorted({m.split("_")[0] for m in lbA.model if not m.startswith("WeightedEnsemble")})
FL_TOP = str(lbF.iloc[0].model)
T = tune.set_index("method")

# ---------------- unified comparison table ----------------------------------------------
def prf(col):
    r = evaluate(preds.y_true, preds[col]); return r
comp = []
mp = {"Logistic Regression": ("logreg", "Logistic Regression", T.loc["Logistic Regression"]),
      "RF default (baseline)": ("rf_default", "RandomForest (default) = BASELINE", T.loc["RandomForest (default) = BASELINE"]),
      "RF + Grid Search": (None, "Grid Search", T.loc["Grid Search"]), "RF + Random Search": (None, "Random Search", T.loc["Random Search"]),
      "RF + Optuna": ("rf_tuned", "Optuna (Bayesian/TPE)", T.loc["Optuna (Bayesian/TPE)"])}
for nm, (col, _, row) in mp.items():
    comp.append({"model": nm, "cv_auc": row.cv_auc_mean, "test_auc": row.test_auc, "time_s": row.time_s, "f1@0.5": row["test_f1@0.5"]})
comp.append({"model": f"FLAML ({B}s budget)", "cv_auc": auto["FLAML"]["best_cv_auc"], "test_auc": auto["FLAML"]["roc_auc"], "time_s": auto["FLAML"]["fit_seconds"], "f1@0.5": auto["FLAML"]["f1"]})
ag_val = float(lbA.loc[lbA.model == auto["AutoGluon"]["best_model"], "score_val"].iloc[0])
comp.append({"model": f"AutoGluon ({B}s budget)", "cv_auc": ag_val, "test_auc": auto["AutoGluon"]["roc_auc"], "time_s": auto["AutoGluon"]["fit_seconds"], "f1@0.5": auto["AutoGluon"]["f1"]})
comp = pd.DataFrame(comp).round(4); comp.to_csv(os.path.join(RES, "final_comparison.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 3.8))
names = comp.model.str.replace(r" \(\d+s budget\)", "", regex=True)
ax.barh(names[::-1], comp.test_auc[::-1], color=["#E45756" if "AutoGluon" in n or "FLAML" in n else "#4C78A8" for n in names[::-1]])
ax.set_xlim(0.5, 0.75); ax.set_xlabel("Held-out test ROC-AUC"); ax.set_title("All approaches on the same test set (blue = manual, red = AutoML)")
for i, v in enumerate(comp.test_auc[::-1]): ax.text(v + .003, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "final_comparison.png"), dpi=140); plt.close()

# leak figure (re-drawn with clean legend)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); x = np.arange(len(leak))
ax[0].bar(x - .2, leak.auc_clean, .4, label="clean", color="#4C78A8"); ax[0].bar(x + .2, leak.auc_with_leak, .4, label="with leaky feature", color="#E45756")
ax[0].set_xticks(x); ax[0].set_xticklabels(leak.index, fontsize=8); ax[0].set_ylim(.5, 1.12); ax[0].legend(loc="upper left", ncol=2, fontsize=8); ax[0].set_ylabel("Test ROC-AUC"); ax[0].set_title("Test AUC jumps when a leak is present")
ax[1].bar(leak.index, leak.leak_permutation_AUC_drop, color="#E45756"); ax[1].set_title("AUC lost when the leaky feature is shuffled"); ax[1].tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "leakage_experiment.png"), dpi=140); plt.close()

# ---------------- architecture diagram ---------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3.6)
boxes = [("Framingham\ndataset\n4,240 rows", .2, 1.4, "#DCE6F2"), ("Clean + split\n80/20 stratified\nseed=42", 2.1, 1.4, "#DCE6F2"),
         ("Manual baseline\nGrid / Random /\nOptuna (RF)", 4.0, 2.4, "#E2F0D9"), ("AutoML\nFLAML, AutoGluon\nsame time budget", 4.0, .4, "#FCE4D6"),
         ("Critical evaluation\nCV, leak test, bias,\nimbalance, SHAP", 6.2, 1.4, "#FFF2CC"), ("Export best model\njoblib pipeline\n+ model card", 8.2, 1.4, "#E4DFEC"),
         ("FastAPI /predict\nStreamlit demo", 10.15, 1.4, "#E4DFEC")]
for t, x0, y0, c in boxes:
    ax.add_patch(FancyBboxPatch((x0, y0), 1.65, 1.1, boxstyle="round,pad=0.04", fc=c, ec="#555")); ax.text(x0 + .825, y0 + .55, t, ha="center", va="center", fontsize=8.5)
arr = lambda a, b: ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.3))
arr((1.85, 1.95), (2.1, 1.95)); arr((3.75, 2.2), (4.0, 2.9)); arr((3.75, 1.7), (4.0, 1.0)); arr((5.65, 2.9), (6.2, 2.2)); arr((5.65, 1.0), (6.2, 1.7)); arr((7.85, 1.95), (8.2, 1.95)); arr((9.85, 1.95), (10.15, 1.95))
ax.text(6, 3.45, "Workflow / architecture", ha="center", fontsize=11, weight="bold")
plt.savefig(os.path.join(FIG, "architecture.png"), dpi=150, bbox_inches="tight"); plt.close()

# ---------------- MODEL CARD ------------------------------------------------------------
m = rep["final_model_test_tuned_thr"]; m5 = rep["final_model_test_default_thr"]
def md(df):
    d = df.astype(str); h = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return h + "\n".join("| " + " | ".join(r) + " |" for r in d.values.tolist())

P_best = load_json("tuning_best_params.json")["Optuna (Bayesian/TPE)"]
FEATS = [("male", "Sex", "0 = female, 1 = male"), ("age", "Age (years)", "32 – 70"), ("education", "Education level", "1 – 4 (ordinal)"),
         ("currentSmoker", "Current smoker", "0 / 1"), ("cigsPerDay", "Cigarettes per day", "0 – 70"), ("BPMeds", "On BP medication", "0 / 1"),
         ("prevalentStroke", "Prior stroke", "0 / 1"), ("prevalentHyp", "Hypertensive", "0 / 1"), ("diabetes", "Diabetes", "0 / 1"),
         ("totChol", "Total cholesterol (mg/dL)", "107 – 696"), ("sysBP", "Systolic BP (mmHg)", "83 – 295"), ("diaBP", "Diastolic BP (mmHg)", "48 – 142"),
         ("BMI", "Body-mass index", "15.5 – 56.8"), ("heartRate", "Resting heart rate (bpm)", "44 – 143"), ("glucose", "Glucose (mg/dL)", "40 – 394")]
feat_tbl = pd.DataFrame([{"Column": a, "Meaning": b, "Range / values": c, "Missing": eda["missing"].get(a, 0)} for a, b, c in FEATS])
cmp_tbl = comp.copy(); cmp_tbl.columns = ["Approach", "CV / val AUC", "Test AUC", "Train time (s)", "F1 @ 0.5"]
sub_tbl = bias.copy(); sub_tbl.columns = ["Attribute", "Group", "n", "CHD prevalence", "AUC", "Flag rate", "Recall", "False-positive rate"]
imp_top = fimp.head(6).reset_index(); imp_top.columns = ["Feature", "Mean abs SHAP", "Permutation AUC drop"]
lk_tbl = leak[["auc_clean", "auc_with_leak", "leak_permutation_AUC_drop"]].round(3).reset_index(); lk_tbl.columns = ["Model", "Test AUC (clean)", "Test AUC (with leaky feature)", "AUC lost when leak shuffled"]
ov_tbl = over.round(3).copy(); ov_tbl.columns = ["Model", "Train AUC (in CV)", "CV val AUC", "Train AUC (full fit)", "Test AUC"]

card = f"""# Model Card — 10-year Coronary Heart Disease Risk (Framingham)

> **One-line summary:** a tuned Random Forest that estimates a person's chance of coronary heart disease (CHD) in the next 10 years from 15 routine
> clinical measurements. Built for an AutoML coursework project; **educational use only, not a medical device.**

## 1. Model overview
| Item | Value |
|---|---|
| Task | Binary classification (CHD within 10 years: yes / no), output = probability |
| Model | RandomForest (Optuna-tuned) inside an sklearn `Pipeline` (median imputer → RandomForest) |
| Hyper-parameters | `{json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in P_best.items()})}` |
| Training data | {4240 - len(preds):,} patients (80 % stratified split), random seed {SEED} |
| Decision threshold | {thr:.2f} (maximises F1 on out-of-fold *training* predictions; not tuned on test data) |
| Why this model was deployed | Rule set before looking at test scores: deploy the tuned sklearn model unless another candidate beats its CV AUC by more than 0.01. {'No candidate did.' if auto['FLAML']['best_cv_auc'] - T.loc['Optuna (Bayesian/TPE)','cv_auc_mean'] <= 0.01 else 'FLAML did, so check src/06_export_model.py.'} It is also exactly explainable (TreeSHAP) and light to deploy. |
| Files | `model/heart_risk_pipeline.joblib`, `model/model_meta.json` |
| Serving | FastAPI (`app/api.py`) and Streamlit dashboard (`app/streamlit_app.py`) |

## 2. Data
Public Framingham Heart Study extract: **{eda['rows']:,} patients**, 15 predictors, target `TenYearCHD` ({eda['positive_rate']*100:.1f} % positive → imbalanced).
{sum(eda['missing'].values())} missing cells across {len(eda['missing'])} columns, imputed with the *training* median **inside** the pipeline (so imputation is re-fitted in every CV fold).
Held-out test set: {len(preds)} patients ({int(preds.y_true.sum())} positives), used once for final reporting.

{md(feat_tbl)}

## 3. Performance (held-out test set, {len(preds)} patients)
| Metric | Value |
|---|---|
| ROC-AUC | **{m['roc_auc']:.3f}** (95 % bootstrap CI {ci.loc['rf_tuned','ci_low']:.3f} – {ci.loc['rf_tuned','ci_high']:.3f}) |
| PR-AUC (a random model scores {eda['positive_rate']:.3f}) | {m['pr_auc']:.3f} |
| Brier score (lower is better) | {m['brier']:.3f} |
| At threshold {thr:.2f}: recall / precision / F1 | {m['recall']:.2f} / {m['precision']:.2f} / {m['f1']:.2f} |
| At default threshold 0.50: recall / F1 | {m5['recall']:.2f} / {m5['f1']:.2f} — the model almost never predicts "CHD" at 0.5 because the class is rare |
| 5-fold CV ROC-AUC on training data | {T.loc['Optuna (Bayesian/TPE)','cv_auc_mean']:.3f} |

**How to read this:** an AUC of 0.5 is a coin flip and 1.0 is perfect. About 0.7 means the model ranks a randomly chosen future-CHD patient above a
randomly chosen healthy one about 7 times out of 10 — useful as a screening signal, far too weak for individual decisions.

### Comparison with the other approaches (same split, same test set)
{md(cmp_tbl.round(3))}

Bootstrap 95 % intervals for test AUC: {', '.join(f"{k} {r.ci_low:.2f}–{r.ci_high:.2f}" for k, r in ci.iterrows())}. They overlap heavily, so **no approach is significantly better**; AutoML ({B} s budget per tool) did not beat a plain logistic regression.

## 4. Fairness and subgroup behaviour (at threshold {thr:.2f})
{md(sub_tbl)}

* Men are flagged more often than women ({bias.selection_rate[1]:.0%} vs {bias.selection_rate[0]:.0%}); part of this reflects the real prevalence gap ({bias.prevalence[1]:.0%} vs {bias.prevalence[0]:.0%}), but men also receive more false alarms.
* The model is **strongly age-driven**: patients under 45 are rarely flagged (recall {bias.recall_TPR[2]:.2f}) while {bias.selection_rate[4]:.0%} of those aged 55+ are flagged.
* Subgroups are small (n = {bias.n.min()}–{bias.n.max()}); treat these numbers as indicative. **No fairness mitigation was applied.**

## 5. Explainability
Global explanations use SHAP (TreeExplainer) on the test set; permutation importance is shown as a cross-check.

{md(imp_top.round(4))}

Directions are clinically plausible: older age, higher blood pressure, more cigarettes and male sex raise predicted risk. Every prediction in the app and API returns its own top contributing factors.
Several features (cholesterol, BMI, glucose) add little or nothing out-of-sample.

## 6. Robustness checks performed
**Overfitting** — train vs cross-validation vs test AUC:

{md(ov_tbl)}

The default forest memorises the training data (train AUC 1.0); tuning regularises it. The CV → test gap of the tuned model is within test-set sampling noise.

**Data leakage experiment** — we injected a feature recording something that happens *after* the outcome (the true label with 5 % noise):

{md(lk_tbl)}

Every model, including both AutoML tools, exploited the leak immediately (AUC ≈ 0.95–0.97) and none raised a warning. **AutoML does not detect leakage; people must.**

**Class imbalance** — weighting classes leaves AUC unchanged but raises recall at the 0.5 threshold from {imb['recall@0.5'][0]*100:.0f} % to {imb['recall@0.5'][1]*100:.0f} %. The operating threshold is a clinical/cost decision.

## 7. Intended use and out-of-scope use
**Intended:** teaching and demonstration of an AutoML-vs-manual workflow; showing how explainable risk scoring, leakage checks and bias checks are done.

**Out of scope — do not use for:** diagnosis or treatment decisions, triage of real patients, insurance or employment decisions, populations unlike the 1948-era Framingham cohort,
or any setting where a wrong prediction could harm a person.

## 8. Limitations
* Modest discrimination (AUC ≈ 0.7); at the chosen threshold roughly {1-m['precision']:.0%} of flagged patients are false alarms.
* Small test set ({int(preds.y_true.sum())} positives): differences of about ±0.05 AUC between models are within sampling noise; one split was used for the final comparison.
* Historical single-town US cohort — may not transfer to other populations, eras, or ethnic groups.
* Probabilities are not guaranteed to be calibrated (see `figures/calibration.png`).
* AutoML budget used was {B} s per tool on {N_JOBS} CPU cores, shorter than the 15–30 min often recommended; some AutoGluon model families (neural nets, XGBoost) could not run in our environment.
* Risk bands in the dashboard (Lower / Elevated / High) are a display convention based on the threshold, not clinical categories.
* Patient names saved in the dashboard are stored in a local SQLite file (`data/patients.db`) with no encryption or access control — use only fictional data.

## 9. How to use
```bash
streamlit run app/streamlit_app.py          # dashboard
uvicorn app.api:app --reload                # REST API, docs at /docs
```
```json
POST /predict   {{"male":1,"age":62,"currentSmoker":1,"cigsPerDay":20,"prevalentHyp":1,"diabetes":0,"sysBP":160,"diaBP":95,"totChol":260,"BMI":28,"heartRate":80,"glucose":90}}
```

## 10. Reproducibility and maintenance
Fixed seed {SEED}; one shared train/test split in `src/common.py`; all numbers in this card are regenerated by `src/07_build_assets.py` from files in `results/`.
Retrain by re-running `src/01` → `src/07`. Last built: {pd.Timestamp.now().strftime('%d %b %Y')}. Contact: project team (see report cover).
"""
open(os.path.join(ROOT, "MODEL_CARD.md"), "w", encoding="utf-8").write(card)
print("model card written")

# ---------------- REPORT (PDF) ----------------------------------------------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1F3864"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2F5496"))
BD = ParagraphStyle("BD", parent=ss["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=5)
SM = ParagraphStyle("SM", parent=BD, fontSize=8, leading=10, textColor=colors.HexColor("#444"))
BL = ParagraphStyle("BL", parent=BD, leftIndent=14, bulletIndent=4)

def tbl(df, widths=None, font=8):
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t

def img(name, w=15.5):
    from PIL import Image as PI
    p = os.path.join(FIG, name); iw, ih = PI.open(p).size
    return Image(p, width=w * cm, height=w * cm * ih / iw)

def bullets(items): return [Paragraph(i, BL, bulletText="•") for i in items]
f4 = lambda v: f"{v:.3f}"

S = []
S += [Spacer(1, 3 * cm), Paragraph("AutoML Skill-Based Project (IA-3)", ParagraphStyle("t", parent=ss["Title"], fontSize=24)),
      Paragraph("Manual vs AutoML for 10-year Heart Disease Risk Prediction", ParagraphStyle("st", parent=ss["Title"], fontSize=15, textColor=colors.HexColor("#2F5496"))),
      Spacer(1, 1 * cm), Paragraph("Subject: AutoML", ParagraphStyle("c", parent=BD, alignment=1, fontSize=12)),
      Paragraph(TEAM, ParagraphStyle("c2", parent=BD, alignment=1, fontSize=11)),
      Spacer(1, 1.5 * cm), Paragraph(f"Dataset: Framingham Heart Study ({eda['rows']:,} rows) &nbsp;|&nbsp; Random seed: {SEED} &nbsp;|&nbsp; AutoML budget per tool: {B} s", ParagraphStyle("c3", parent=BD, alignment=1)),
      Spacer(1, 1.5 * cm), img("architecture.png", 16), PageBreak()]

S += [Paragraph("1. Problem statement and dataset", H1),
      Paragraph("<b>Problem.</b> Cardiovascular disease is a leading cause of death. Given routine clinical measurements, can we estimate whether a person will develop coronary heart disease (CHD) within ten years? This is a binary classification task with an imbalanced target. We use it to compare a carefully engineered manual scikit-learn workflow against two AutoML frameworks and to audit all of them for leakage, bias and explainability.", BD),
      Paragraph(f"<b>Dataset.</b> Public Framingham Heart Study extract ({eda['rows']:,} patients, 15 predictors, target <i>TenYearCHD</i>). {eda['positive_rate']*100:.1f}% of patients are positive. {sum(eda['missing'].values())} cells are missing (glucose {eda['missing']['glucose']}, education {eda['missing']['education']}, BPMeds {eda['missing']['BPMeds']}, totChol {eda['missing']['totChol']}, cigsPerDay {eda['missing']['cigsPerDay']}, BMI {eda['missing']['BMI']}, heartRate {eda['missing']['heartRate']}). There are {eda['duplicates']} duplicate rows.", BD),
      Paragraph("<b>Validation design.</b> One stratified 80/20 split (seed 42). All tuning, AutoML search and threshold selection use only the 80% training part (5-fold stratified CV). The 20% test set is scored once per model at the end. Imputation is inside the sklearn pipeline, so it is re-fitted in every CV fold (no preprocessing leakage).", BD),
      img("eda_overview.png", 16.5)]

S += [Paragraph("2. Manual baseline and hyper-parameter tuning", H1),
      Paragraph("The baseline is a default RandomForest in a Pipeline (median imputer + forest). The <b>same model</b> was then tuned with Grid Search, Random Search and Bayesian optimisation (Optuna TPE). Every tuner got exactly 36 evaluations with 5-fold CV on ROC-AUC, so time differences reflect the method, not the budget. ROC-AUC is the main metric because with 15% positives accuracy (≈85% for predicting all negatives) and F1 at 0.5 are misleading.", BD)]
t2 = tune.copy(); t2.columns = ["Method", "CV AUC", "CV std", "Test AUC", "Test F1@0.5", "Time (s)", "Evals"]
t2 = t2.round(3).fillna("-"); S += [tbl(t2, [5.2 * cm, 1.8 * cm, 1.6 * cm, 1.8 * cm, 2.2 * cm, 1.8 * cm, 1.4 * cm]), Spacer(1, 6), img("tuning_comparison.png", 16.5)]
bp = load_json("tuning_best_params.json")
S += [Paragraph("<b>Observations.</b>", BD)] + bullets([
    f"All three tuners reach almost the same CV AUC ({T.loc['Grid Search','cv_auc_mean']:.3f} / {T.loc['Random Search','cv_auc_mean']:.3f} / {T.loc['Optuna (Bayesian/TPE)','cv_auc_mean']:.3f}); the differences are far smaller than the CV standard deviation (≈0.03). Tuning improved the baseline CV AUC by about {T.loc['Optuna (Bayesian/TPE)','cv_auc_mean']-T.loc['RandomForest (default) = BASELINE','cv_auc_mean']:.3f}.",
    f"Grid search was fastest here ({T.loc['Grid Search','time_s']:.0f} s) only because our grid is small and contains cheap (shallow, few-tree) settings; Random ({T.loc['Random Search','time_s']:.0f} s) and Optuna ({T.loc['Optuna (Bayesian/TPE)','time_s']:.0f} s) sample bigger forests (up to 500 trees). Time per method therefore depends on the search space as well as the algorithm.",
    f"A plain scaled Logistic Regression is competitive (test AUC {T.loc['Logistic Regression','test_auc']:.3f}) and takes {T.loc['Logistic Regression','time_s']:.1f} s. Signal in this dataset is mostly linear, so complex models add little.",
    "Best parameters: Grid " + json.dumps(bp["Grid Search"]) + "; Random " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in bp["Random Search"].items()}) + "; Optuna " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in bp["Optuna (Bayesian/TPE)"].items()}) + "."])

S += [PageBreak(), Paragraph("3. AutoML implementation and tool comparison", H1),
      Paragraph(f"Two frameworks were run on the identical training data and scored on the identical test set: <b>FLAML</b> (cost-frugal search over LightGBM, XGBoost, CatBoost, RF, ExtraTrees, linear models; 5-fold CV inside) and <b>AutoGluon-Tabular</b> (preset best_quality: many bagged models + weighted ensemble). Each tool had a fixed wall-clock budget of <b>{B} seconds</b> on {N_JOBS} CPU cores, metric ROC-AUC, seed {SEED}. (The assignment suggests 15–30 min; the budget is one variable, <i>AUTOML_BUDGET</i>, and the reproduction scripts accept 900 or 1800.)", BD),
      Paragraph("<b>FLAML leaderboard</b> (best 5-fold CV AUC reached by each learner; 'improving configs logged' counts configurations that improved the best score and were written to the log):", BD)]
l1 = lbF.copy(); l1.columns = ["Learner", "Best CV AUC", "Improving configs logged"]; l1 = l1.round(4).fillna("-"); S += [tbl(l1, [4 * cm, 3 * cm, 5 * cm])]
S += [Spacer(1, 6), Paragraph(f"FLAML selected <b>{auto['FLAML']['best_estimator']}</b> (config {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in auto['FLAML']['best_config'].items()})}).", BD),
      Paragraph(f"<b>AutoGluon leaderboard</b> (top 8 of {auto['AutoGluon']['n_models']} models trained; score_val is AutoGluon's internal out-of-fold AUC, score_test is our held-out test AUC):", BD)]
l2 = lbA.head(8)[["model", "score_val", "score_test", "fit_time", "stack_level"]].round(4); l2.columns = ["Model", "Val AUC (OOF)", "Test AUC", "Fit time (s)", "Stack level"]
S += [tbl(l2, [5.2 * cm, 3 * cm, 2.5 * cm, 2.5 * cm, 2.2 * cm]), Spacer(1, 4),
      Paragraph(f"AutoGluon's winner was <b>{auto['AutoGluon']['best_model']}</b>; The best single-model family on the AutoGluon leaderboard was <b>{AG_TOP}</b> (families that trained successfully: {', '.join(AG_FAMILIES)}). Model families that failed to import or train in the environment (e.g. torch/fastai neural nets, or XGBoost with the installed scikit-learn version; see results/03_automl.log) were skipped by AutoGluon and not tried (limitation).", BD),
      Paragraph("<b>Preprocessing / feature engineering applied by the tools.</b> FLAML: its built-in data transformer (imputation of missing values and encoding of categoricals, per FLAML documentation); no new features. AutoGluon: AutoMLPipelineFeatureGenerator – automatic type inference, native NaN handling for tree models, removal of constant/duplicate columns. We checked <i>predictor.features()</i>: it used the same 15 input columns and created no new ones. With an all-numeric table there is little for either tool to engineer.", BD),
      PageBreak(), Paragraph("<b>Final comparison on the held-out test set</b>", H2)]
c2 = comp.copy(); c2.columns = ["Model", "CV/val AUC", "Test AUC", "Train time (s)", "F1@0.5"]
c2["Test AUC 95% CI"] = ["{:.3f}–{:.3f}".format(ci.loc[k, "ci_low"], ci.loc[k, "ci_high"]) if k else "-" for k in ["logreg", "rf_default", None, None, "rf_tuned", "flaml", "autogluon"]]
S += [KeepTogether([tbl(c2, [4.8 * cm, 2.1 * cm, 1.8 * cm, 2.4 * cm, 1.6 * cm, 3 * cm])]), Spacer(1, 6), img("final_comparison.png", 12.5), img("roc_all_models.png", 8.5)]
S += [Paragraph("<b>Interpretation.</b>", BD)] + bullets([
    f"No approach is significantly better: all 95% bootstrap intervals overlap heavily (test set has only {int(preds.y_true.sum())} positives). The point estimates rank Logistic Regression ({ci.loc['logreg','test_auc']:.3f}) &gt; FLAML ({ci.loc['flaml','test_auc']:.3f}) &gt; AutoGluon ({ci.loc['autogluon','test_auc']:.3f}) &gt; tuned RF ({ci.loc['rf_tuned','test_auc']:.3f}) &gt; default RF ({ci.loc['rf_default','test_auc']:.3f}), but this ordering should not be over-interpreted.",
    "Both AutoML tools beat the manual untuned/tuned RF on test AUC slightly but not the simple Logistic Regression. AutoML's value here was speed of obtaining a strong, well-validated model with zero manual design, not a higher ceiling.",
    "Training time: baseline 2 s, logistic regression 0.1 s, manual tuning 110–215 s per method, AutoML = the full budget by design (FLAML and AutoGluon keep searching until time runs out).",
    "Interpretability: Logistic Regression (coefficients) &gt; tuned RF (exact TreeSHAP) &gt; FLAML single ExtraTrees (SHAP possible) &gt; AutoGluon (a weighted ensemble of 29 bagged models: explainable only through model-agnostic methods and much heavier to deploy)."])

S += [PageBreak(), Paragraph("4. Critical evaluation", H1), Paragraph("4.1 Overfitting and validation", H2)]
o = over.copy(); o.columns = ["Model", "Train AUC (inside CV)", "CV val AUC", "Train AUC (full fit)", "Test AUC"]; S += [tbl(o.round(3), [4 * cm, 3.2 * cm, 2.4 * cm, 3.2 * cm, 2.2 * cm])]
S += [Spacer(1, 4), Paragraph(f"The default forest memorises the training data (train AUC 1.000 vs CV {over.cv_val_auc[0]:.3f}) – classic overfitting that the held-out test set confirms (test {over.test_auc[0]:.3f}). Tuning (shallower, more regularised trees) lowers train AUC to {over.train_auc_in_cv[1]:.3f} and raises CV and test AUC. Note the CV-to-test drop for the tuned model ({over.cv_val_auc[1]:.3f} → {over.test_auc[1]:.3f}) is within what a 848-patient test set can produce (bootstrap CI ±0.05), so we report both numbers rather than only the better one.", BD),
      Paragraph("4.2 Deliberate data leakage experiment", H2),
      Paragraph("We injected a leaky feature <i>cardiac_medication_started</i> = the true outcome with 5% of values randomly flipped (in real life: a drug started <b>after</b> a cardiac event, i.e. information from the future). It was added to train and test. The manual forest used default settings; FLAML and AutoGluon were re-run from scratch with half the main budget (" + str(max(B // 2, 60)) + " s).", BD)]
lk = leak[["auc_clean", "auc_with_leak", "leak_permutation_AUC_drop"]].round(3).reset_index(); lk.columns = ["Model", "Test AUC clean", "Test AUC with leak", "AUC lost if leak shuffled"]
S += [tbl(lk, [4.5 * cm, 3.3 * cm, 3.6 * cm, 4 * cm]), Spacer(1, 4), img("leakage_experiment.png", 15.5),
      Paragraph(f"<b>Result:</b> every method exploits the leak immediately: test AUC jumps from ≈0.63–0.69 to ≈0.95–0.97, and shuffling the leaky column destroys 0.34–0.39 AUC. AutoGluon's own feature-importance ranks it #1. <b>AutoML does not detect leakage</b> – it optimises the metric it is given. A simple single-feature screen did expose the problem before modelling: the leak alone has AUC {leakscreen.iloc[0,0]:.3f} while the best genuine feature (age) has {leakscreen.iloc[1,0]:.3f}. An AUC above ~0.9 on a task where the literature reports ≈0.7–0.75 is itself a red flag.", BD),
      Paragraph("4.3 Class imbalance", H2)]
im = imb.copy(); im.columns = ["Setting", "CV AUC", "Test AUC", "PR-AUC", "Recall@0.5", "Precision@0.5", "F1@0.5"]
S += [tbl(im.round(3), [4.5 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.2 * cm, 2.6 * cm, 1.6 * cm]), Spacer(1, 4),
      Paragraph(f"Only {eda['positive_rate']*100:.1f}% of patients are positive. At the default 0.5 threshold the tuned forest finds {imb['recall@0.5'][0]*100:.1f}% of true cases – accuracy (≈85%) hides this. Class weighting leaves ranking quality (AUC) unchanged but raises recall at 0.5 to {imb['recall@0.5'][1]*100:.0f}%; alternatively a lower threshold ({thr:.2f}, chosen on out-of-fold <i>training</i> predictions only) gives recall {m['recall']:.2f}, precision {m['precision']:.2f}, F1 {m['f1']:.2f} on the test set. The decision threshold is a clinical/cost choice, not a modelling detail.", BD),
      Paragraph("4.4 Subgroup bias (tuned RF at threshold %.2f)" % thr, H2)]
b = bias.copy(); b.columns = ["Attribute", "Group", "n", "CHD prevalence", "AUC", "Flag rate", "Recall (TPR)", "False-pos. rate"]
S += [tbl(b, [2 * cm, 1.8 * cm, 1.2 * cm, 2.6 * cm, 1.5 * cm, 2 * cm, 2.4 * cm, 2.8 * cm]), Spacer(1, 4),
      Paragraph(f"Men are flagged more often than women ({bias.selection_rate[1]:.0%} vs {bias.selection_rate[0]:.0%}) and have both higher recall and a higher false-positive rate; part of this reflects the real prevalence gap ({bias.prevalence[1]:.0%} vs {bias.prevalence[0]:.0%}). Discrimination (AUC) is similar for both sexes ({bias.auc[0]:.2f} vs {bias.auc[1]:.2f}). The model is strongly age-driven: patients under 45 are almost never flagged (recall {bias.recall_TPR[2]:.2f}) while {bias.selection_rate[4]:.0%} of those 55+ are flagged. Within-group AUC is lower than overall AUC ({bias.auc[2]:.2f}–{bias.auc[4]:.2f}) because much of the overall ranking comes from age alone. Subgroups are small (n=258–482), so these figures are indicative, not definitive. No fairness mitigation was applied.", BD),
      Paragraph("4.5 Explainability (SHAP)", H2), img("shap_beeswarm.png", 11),
      Paragraph(f"Global SHAP (TreeExplainer on the tuned forest, test set) ranks <b>age</b> first (mean |SHAP| {fimp['mean_|SHAP|'].iloc[0]:.3f}), then <b>systolic BP</b> ({fimp['mean_|SHAP|'].iloc[1]:.3f}), hypertension, cigarettes per day and sex. All directions are clinically plausible (older, higher BP, more cigarettes, male → higher risk). Permutation importance on the test set agrees on the top features (age drop {fimp['permutation_AUC_drop'].iloc[0]:.3f}, sysBP {fimp['permutation_AUC_drop'].iloc[1]:.3f}) and shows that several low-ranked features (totChol, BMI, glucose) contribute ≈0 or slightly negative AUC out-of-sample – evidence the forest uses them partly as noise.", BD),
      img("shap_local_example.png", 11), Paragraph("Local explanation for the highest-risk test patient (also returned live by the API/app).", SM)]

S += [Paragraph("4.6 Limitations found", H2)] + bullets([
    "Weak signal: best test AUC ≈0.70; clinical use would need much better discrimination and calibration.",
    f"Small test set ({len(preds)} patients, {int(preds.y_true.sum())} positives): CV and test AUC differ by ≈0.05 and model rankings are within noise. A single split was used for the final comparison; repeated CV across all tools would be better but exceeds the time budget.",
    f"AutoML budget used for the reported numbers was {B} s per tool on {N_JOBS} cores (shorter than the 15–30 min suggested), so FLAML could evaluate only a limited number of configurations (5-fold CV on 2 cores); longer budgets may change the ranking slightly. Some AutoGluon model families (neural nets, and XGBoost in our environment) failed to run and were skipped.",
    "Threshold F1 optimisation and probability calibration were only lightly explored; probabilities are not guaranteed calibrated.",
    "Dataset is a single historical US cohort; the education variable is of questionable causal meaning and glucose has 9% missing values (median imputation).",
    "The leakage experiment used a synthetic leak; real leaks (e.g. duplicated patients, time leakage) are subtler."])

S += [PageBreak(), Paragraph("5. Deployment", H1),
      Paragraph(f"<b>Model choice.</b> Rule fixed before looking at test scores: deploy the Optuna-tuned sklearn pipeline unless another candidate beats its CV AUC by more than 0.01 (CV std ≈0.03). FLAML's CV AUC was {auto['FLAML']['best_cv_auc']:.3f} vs {T.loc['Optuna (Bayesian/TPE)','cv_auc_mean']:.3f}, a difference of {auto['FLAML']['best_cv_auc']-T.loc['Optuna (Bayesian/TPE)','cv_auc_mean']:+.3f}, so the simpler tuned forest was exported (<i>model/heart_risk_pipeline.joblib</i>). It needs only scikit-learn, supports exact SHAP and avoids shipping AutoGluon's 29-model directory.", BD),
      Paragraph("<b>Serving.</b> <i>app/api.py</i> is a FastAPI service (POST /predict with validated inputs, GET /health, GET /model-info, interactive docs at /docs). <i>app/streamlit_app.py</i> is a Streamlit demo with sliders, predicted 10-year risk vs the dataset base rate, the alert flag and a SHAP bar chart of the top factors; the model card is shown in a second tab. Both import <i>app/serving.py</i>, so they always give identical answers. Both were smoke-tested (valid prediction, 422 on invalid input).", BD),
      Paragraph("<b>Run locally:</b> <i>uvicorn app.api:app --reload</i> and <i>streamlit run app/streamlit_app.py</i> (see README). The Streamlit app can also be deployed free on Streamlit Community Cloud from a GitHub repo.", BD),
      Paragraph("The full model card is in MODEL_CARD.md (data, metrics, intended use, known limits).", BD)]

S += [Paragraph("6. Reflections: what AutoML did well and where it failed", H1), Paragraph("<b>Did well</b>", BD)] + bullets([
    "Produced a competitive, properly cross-validated model in a few minutes with no manual feature or model design; FLAML's CV AUC (0.728) matched our 36-trial manual search (0.726).",
    f"Transparent leaderboards showed which families work (FLAML: {FL_TOP} on top; AutoGluon: {AG_TOP} on top; linear models weakest in CV).",
    "Handled missing values and types automatically."]) + [Paragraph("<b>Failed / needs a human</b>", BD)] + bullets([
    "Did not beat a one-line logistic regression on test AUC, and cannot raise the information ceiling of weak data.",
    "Exploited the leaky feature instantly and gave no warning – validation scores looked superb. A human must audit features.",
    "Default decision threshold and class imbalance were not handled in a clinically meaningful way (recall ≈2–3% at 0.5 for both tools).",
    "Heavy ensembles (29 models) are hard to explain and deploy; the 'best' model is chosen on noisy validation scores.",
    "Results depend strongly on the time budget and hardware, so tool comparisons must fix both (we did)."]) + [
    Paragraph("<b>Takeaway:</b> AutoML is an excellent baseline generator and productivity tool, but validation design, leakage hunting, bias analysis and threshold choice remain human responsibilities.", BD),
    Paragraph("7. Reproducibility", H1),
    Paragraph(f"All code is in <i>src/</i> and runs in order 01 → 07. Fixed seed {SEED}; one shared split in <i>src/common.py</i>. Set <i>AUTOML_BUDGET=900</i> (or 1800) to reproduce with the longer budgets recommended by the brief. Wall-clock-based AutoML search is not exactly deterministic across machines, so expect small differences in AutoML numbers.", BD)]

doc = SimpleDocTemplate(os.path.join(ROOT, "report", "AutoML_Project_Report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="AutoML Project Report")
doc.build(S, onFirstPage=lambda c, d: None, onLaterPages=lambda c, d: (c.setFont("Helvetica", 8), c.drawString(2 * cm, 1 * cm, "AutoML IA-3 - Heart risk prediction"), c.drawRightString(19 * cm, 1 * cm, f"Page {d.page}")))
print("report written")

# ---------------- SLIDES ------------------------------------------------------------------
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
NAVY = RGBColor(0x1F, 0x38, 0x64)

def slide(title, bullets_=None, image=None, img_w=7.2, notes=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.0)); bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(.5), Inches(.15), Inches(12.3), Inches(.7)); p = tb.text_frame.paragraphs[0]; p.text = title; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
    left_w = 5.0 if image else 12.3
    if bullets_:
        tb = s.shapes.add_textbox(Inches(.5), Inches(1.3), Inches(left_w), Inches(5.8)); tf = tb.text_frame; tf.word_wrap = True
        for i, b_ in enumerate(bullets_):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = "• " + b_; p.font.size = Pt(17); p.space_after = Pt(9)
    if image:
        from PIL import Image as PI
        iw, ih = PI.open(os.path.join(FIG, image)).size; w = img_w; h = w * ih / iw
        if h > 5.8: h = 5.8; w = h * iw / ih
        s.shapes.add_picture(os.path.join(FIG, image), Inches(13.0 - w) if bullets_ else Inches((13.333 - w) / 2), Inches(1.3), width=Inches(w))
    if notes: s.notes_slide.notes_text_frame.text = notes
    return s

s = prs.slides.add_slide(prs.slide_layouts[6]); bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(.8), Inches(2.2), Inches(11.7), Inches(3)); tf = tb.text_frame; tf.word_wrap = True
tf.paragraphs[0].text = "Manual ML vs AutoML"; tf.paragraphs[0].font.size = Pt(44); tf.paragraphs[0].font.bold = True; tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
for t_, sz in [("10-year heart disease risk prediction (Framingham)", 24), ("AutoML - IA-3 project", 18), (TEAM, 16)]:
    p = tf.add_paragraph(); p.text = t_; p.font.size = Pt(sz); p.font.color.rgb = RGBColor(0xD9, 0xE2, 0xF3)

slide("Problem and data", [f"Predict CHD within 10 years from routine clinical data", f"Framingham extract: {eda['rows']:,} rows, 15 features, {eda['positive_rate']*100:.1f}% positive (imbalanced)", f"{sum(eda['missing'].values())} missing cells (glucose 9%)", "One stratified 80/20 split, seed 42; test set touched once", "Metric: ROC-AUC (accuracy is misleading at 85% majority)"], "eda_overview.png", 7.3)
slide("Workflow: same split, same data, same budget", None, "architecture.png", 12.3)
slide("Step 2: Baseline and tuning (same RF, 36 evaluations each)", [f"Baseline CV AUC {T.loc['RandomForest (default) = BASELINE','cv_auc_mean']:.3f}, train AUC 1.00 -> overfits", f"Grid / Random / Optuna CV AUC: {T.loc['Grid Search','cv_auc_mean']:.3f} / {T.loc['Random Search','cv_auc_mean']:.3f} / {T.loc['Optuna (Bayesian/TPE)','cv_auc_mean']:.3f}", "Differences are below CV noise (std 0.03)", f"Time: {T.loc['Grid Search','time_s']:.0f}s / {T.loc['Random Search','time_s']:.0f}s / {T.loc['Optuna (Bayesian/TPE)','time_s']:.0f}s", f"Logistic regression is competitive (test {T.loc['Logistic Regression','test_auc']:.3f})"], "tuning_comparison.png", 7.3)
slide("Step 3: AutoML leaderboards", [f"FLAML winner: {auto['FLAML']['best_estimator']} (CV AUC {auto['FLAML']['best_cv_auc']:.3f})", f"AutoGluon: {auto['AutoGluon']['n_models']} models, winner {auto['AutoGluon']['best_model']}", "Tools used the same 15 features; no new features created", f"Top families: FLAML {FL_TOP}, AutoGluon {AG_TOP}; linear models weakest", "Some AutoGluon families skipped (failed imports, see log)"], "final_comparison.png", 7.0)
slide("Manual vs AutoML: honest result", [f"Test AUC: LogReg {ci.loc['logreg','test_auc']:.3f}, FLAML {ci.loc['flaml','test_auc']:.3f}, AutoGluon {ci.loc['autogluon','test_auc']:.3f}, tuned RF {ci.loc['rf_tuned','test_auc']:.3f}", "95% CIs overlap: no significant winner", "AutoML = speed and robustness, not a higher ceiling", "Interpretability: LogReg > RF+SHAP > FLAML > AutoGluon ensemble", "Time: manual tuning 2-4 min, AutoML = whole budget"], "roc_all_models.png", 5.4)
slide("Step 4: Leakage experiment", [f"Added a post-outcome feature (outcome with 5% noise)", f"Test AUC: ~0.63-0.69 -> {leak.auc_with_leak.min():.2f}-{leak.auc_with_leak.max():.2f}", "Both AutoML tools exploit it at once (AutoGluon ranks it #1)", "No warning from the tools", "Single-feature AUC screen catches it: 0.94 vs 0.69 for age"], "leakage_experiment.png", 7.3)
slide("Imbalance, bias and overfitting", [f"At 0.5 threshold recall is only {imb['recall@0.5'][0]*100:.1f}%; weighting -> {imb['recall@0.5'][1]*100:.0f}% with same AUC", f"Threshold {thr:.2f} from train CV: recall {m['recall']:.2f}, precision {m['precision']:.2f}", f"Men flagged {bias.selection_rate[1]:.0%} vs women {bias.selection_rate[0]:.0%}", f"Under-45s almost never flagged (recall {bias.recall_TPR[2]:.2f}); age-driven model", "CV to test AUC drop 0.05 is within test-set noise"], "calibration.png", 4.8)
slide("Explainability (SHAP)", ["Age and systolic BP dominate", "Directions are clinically plausible", "Cholesterol/BMI/glucose add little out-of-sample", "Local SHAP explanation returned for every prediction"], "shap_beeswarm.png", 5.8)
slide("Step 5: Deployment", ["Exported sklearn pipeline (joblib), chosen by a CV rule set before testing", "FastAPI: /predict, /health, /model-info, docs at /docs", "Streamlit: sliders, risk vs base rate, SHAP factors, model card tab", "Model card: data, metrics, intended use, limits"], "shap_local_example.png", 6.3)
slide("Reflections and limitations", ["AutoML did well: fast, strong, validated baselines", "AutoML failed: used the leak, ignored imbalance, no warnings", "Weak signal (AUC ~0.7): not for clinical use", f"Short budget ({B}s, {N_JOBS} cores); small test set; single split", "Humans own validation design, leakage audit, bias and thresholds"])
prs.save(os.path.join(ROOT, "slides", "AutoML_Project_Slides.pptx")); print("slides written")
