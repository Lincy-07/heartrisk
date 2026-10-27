# Model Card — 10-year Coronary Heart Disease Risk (Framingham)

> **One-line summary:** a tuned Random Forest that estimates a person's chance of coronary heart disease (CHD) in the next 10 years from 15 routine
> clinical measurements. Built for an AutoML coursework project; **educational use only, not a medical device.**

## 1. Model overview
| Item | Value |
|---|---|
| Task | Binary classification (CHD within 10 years: yes / no), output = probability |
| Model | RandomForest (Optuna-tuned) inside an sklearn `Pipeline` (median imputer → RandomForest) |
| Hyper-parameters | `{"n_estimators": 424, "max_depth": 8, "min_samples_leaf": 3, "max_features": 0.236}` |
| Training data | 3,392 patients (80 % stratified split), random seed 42 |
| Decision threshold | 0.18 (maximises F1 on out-of-fold *training* predictions; not tuned on test data) |
| Why this model was deployed | Rule set before looking at test scores: deploy the tuned sklearn model unless another candidate beats its CV AUC by more than 0.01. No candidate did. It is also exactly explainable (TreeSHAP) and light to deploy. |
| Files | `model/heart_risk_pipeline.joblib`, `model/model_meta.json` |
| Serving | FastAPI (`app/api.py`) and Streamlit dashboard (`app/streamlit_app.py`) |

## 2. Data
Public Framingham Heart Study extract: **4,240 patients**, 15 predictors, target `TenYearCHD` (15.2 % positive → imbalanced).
645 missing cells across 7 columns, imputed with the *training* median **inside** the pipeline (so imputation is re-fitted in every CV fold).
Held-out test set: 848 patients (129 positives), used once for final reporting.

| Column | Meaning | Range / values | Missing |
|---|---|---|---|
| male | Sex | 0 = female, 1 = male | 0 |
| age | Age (years) | 32 – 70 | 0 |
| education | Education level | 1 – 4 (ordinal) | 105 |
| currentSmoker | Current smoker | 0 / 1 | 0 |
| cigsPerDay | Cigarettes per day | 0 – 70 | 29 |
| BPMeds | On BP medication | 0 / 1 | 53 |
| prevalentStroke | Prior stroke | 0 / 1 | 0 |
| prevalentHyp | Hypertensive | 0 / 1 | 0 |
| diabetes | Diabetes | 0 / 1 | 0 |
| totChol | Total cholesterol (mg/dL) | 107 – 696 | 50 |
| sysBP | Systolic BP (mmHg) | 83 – 295 | 0 |
| diaBP | Diastolic BP (mmHg) | 48 – 142 | 0 |
| BMI | Body-mass index | 15.5 – 56.8 | 19 |
| heartRate | Resting heart rate (bpm) | 44 – 143 | 1 |
| glucose | Glucose (mg/dL) | 40 – 394 | 388 |

## 3. Performance (held-out test set, 848 patients)
| Metric | Value |
|---|---|
| ROC-AUC | **0.674** (95 % bootstrap CI 0.628 – 0.726) |
| PR-AUC (a random model scores 0.152) | 0.277 |
| Brier score (lower is better) | 0.123 |
| At threshold 0.18: recall / precision / F1 | 0.48 / 0.25 / 0.33 |
| At default threshold 0.50: recall / F1 | 0.02 / 0.03 — the model almost never predicts "CHD" at 0.5 because the class is rare |
| 5-fold CV ROC-AUC on training data | 0.726 |

**How to read this:** an AUC of 0.5 is a coin flip and 1.0 is perfect. About 0.7 means the model ranks a randomly chosen future-CHD patient above a
randomly chosen healthy one about 7 times out of 10 — useful as a screening signal, far too weak for individual decisions.

### Comparison with the other approaches (same split, same test set)
| Approach | CV / val AUC | Test AUC | Train time (s) | F1 @ 0.5 |
|---|---|---|---|---|
| Logistic Regression | 0.731 | 0.702 | 0.166 | 0.096 |
| RF default (baseline) | 0.711 | 0.632 | 2.181 | 0.071 |
| RF + Grid Search | 0.727 | 0.677 | 105.764 | 0.03 |
| RF + Random Search | 0.726 | 0.658 | 200.3 | 0.071 |
| RF + Optuna | 0.726 | 0.674 | 174.104 | 0.03 |
| FLAML (240s budget) | 0.728 | 0.688 | 240.417 | 0.044 |
| AutoGluon (240s budget) | 0.75 | 0.662 | 240.389 | 0.099 |

Bootstrap 95 % intervals for test AUC: logreg 0.66–0.75, rf_default 0.58–0.69, rf_tuned 0.63–0.73, flaml 0.64–0.74, autogluon 0.61–0.72. They overlap heavily, so **no approach is significantly better**; AutoML (240 s budget per tool) did not beat a plain logistic regression.

## 4. Fairness and subgroup behaviour (at threshold 0.18)
| Attribute | Group | n | CHD prevalence | AUC | Flag rate | Recall | False-positive rate |
|---|---|---|---|---|---|---|---|
| sex | Female | 482 | 0.108 | 0.659 | 0.237 | 0.404 | 0.216 |
| sex | Male | 366 | 0.21 | 0.66 | 0.369 | 0.532 | 0.325 |
| age | <45 | 299 | 0.09 | 0.626 | 0.04 | 0.111 | 0.033 |
| age | 45-54 | 291 | 0.141 | 0.626 | 0.268 | 0.39 | 0.248 |
| age | 55+ | 258 | 0.236 | 0.605 | 0.616 | 0.705 | 0.589 |

* Men are flagged more often than women (37% vs 24%); part of this reflects the real prevalence gap (21% vs 11%), but men also receive more false alarms.
* The model is **strongly age-driven**: patients under 45 are rarely flagged (recall 0.11) while 62% of those aged 55+ are flagged.
* Subgroups are small (n = 258–482); treat these numbers as indicative. **No fairness mitigation was applied.**

## 5. Explainability
Global explanations use SHAP (TreeExplainer) on the test set; permutation importance is shown as a cross-check.

| Feature | Mean abs SHAP | Permutation AUC drop |
|---|---|---|
| age | 0.0453 | 0.0602 |
| sysBP | 0.023 | 0.0162 |
| prevalentHyp | 0.0152 | 0.0035 |
| cigsPerDay | 0.0122 | 0.0072 |
| male | 0.0117 | 0.0093 |
| diaBP | 0.0102 | 0.0024 |

Directions are clinically plausible: older age, higher blood pressure, more cigarettes and male sex raise predicted risk. Every prediction in the app and API returns its own top contributing factors.
Several features (cholesterol, BMI, glucose) add little or nothing out-of-sample.

## 6. Robustness checks performed
**Overfitting** — train vs cross-validation vs test AUC:

| Model | Train AUC (in CV) | CV val AUC | Train AUC (full fit) | Test AUC |
|---|---|---|---|---|
| RF default (baseline) | 1.0 | 0.711 | 1.0 | 0.632 |
| RF tuned (Optuna) | 0.906 | 0.726 | 0.891 | 0.674 |

The default forest memorises the training data (train AUC 1.0); tuning regularises it. The CV → test gap of the tuned model is within test-set sampling noise.

**Data leakage experiment** — we injected a feature recording something that happens *after* the outcome (the true label with 5 % noise):

| Model | Test AUC (clean) | Test AUC (with leaky feature) | AUC lost when leak shuffled |
|---|---|---|---|
| RandomForest (manual) | 0.632 | 0.952 | 0.386 |
| FLAML | 0.688 | 0.966 | 0.34 |
| AutoGluon | 0.678 | 0.959 | 0.336 |

Every model, including both AutoML tools, exploited the leak immediately (AUC ≈ 0.95–0.97) and none raised a warning. **AutoML does not detect leakage; people must.**

**Class imbalance** — weighting classes leaves AUC unchanged but raises recall at the 0.5 threshold from 2 % to 51 %. The operating threshold is a clinical/cost decision.

## 7. Intended use and out-of-scope use
**Intended:** teaching and demonstration of an AutoML-vs-manual workflow; showing how explainable risk scoring, leakage checks and bias checks are done.

**Out of scope — do not use for:** diagnosis or treatment decisions, triage of real patients, insurance or employment decisions, populations unlike the 1948-era Framingham cohort,
or any setting where a wrong prediction could harm a person.

## 8. Limitations
* Modest discrimination (AUC ≈ 0.7); at the chosen threshold roughly 75% of flagged patients are false alarms.
* Small test set (129 positives): differences of about ±0.05 AUC between models are within sampling noise; one split was used for the final comparison.
* Historical single-town US cohort — may not transfer to other populations, eras, or ethnic groups.
* Probabilities are not guaranteed to be calibrated (see `figures/calibration.png`).
* AutoML budget used was 240 s per tool on 2 CPU cores, shorter than the 15–30 min often recommended; some AutoGluon model families (neural nets, XGBoost) could not run in our environment.
* Risk bands in the dashboard (Lower / Elevated / High) are a display convention based on the threshold, not clinical categories.
* Patient names saved in the dashboard are stored in a local SQLite file (`data/patients.db`) with no encryption or access control — use only fictional data.

## 9. How to use
```bash
streamlit run app/streamlit_app.py          # dashboard
uvicorn app.api:app --reload                # REST API, docs at /docs
```
```json
POST /predict   {"male":1,"age":62,"currentSmoker":1,"cigsPerDay":20,"prevalentHyp":1,"diabetes":0,"sysBP":160,"diaBP":95,"totChol":260,"BMI":28,"heartRate":80,"glucose":90}
```

## 10. Reproducibility and maintenance
Fixed seed 42; one shared train/test split in `src/common.py`; all numbers in this card are regenerated by `src/07_build_assets.py` from files in `results/`.
Retrain by re-running `src/01` → `src/07`. Last built: 24 Oct 2026. Contact: project team (see report cover).
