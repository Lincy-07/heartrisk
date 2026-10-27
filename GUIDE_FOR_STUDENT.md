# Your guide: finishing the AutoML IA-3 project in VS Code

## 1. Is the old SmartHealth project suitable?
**No — create a different project (this folder).** Checked against the PDF:

| Requirement in the PDF | Old SmartHealth project | Verdict |
|---|---|---|
| Real dataset, "at least a few thousand rows" | `heart_data.csv` has 920 rows; `train_model.py` trains on **randomly generated synthetic data** whose label is a hand-written formula | Fails |
| Baseline + Grid/Random/Optuna with time + score table | none | Missing |
| ≥2 AutoML tools, same time budget, leaderboards | only a TPOT script (never run, no budget, no comparison); TPOT is not even in requirements.txt | Missing |
| Leakage test, CV + hold-out, bias, SHAP | none | Missing |
| Deployment with FastAPI/Streamlit + model card | Flask + HTML; `app.py` shows **random "confidence"**, **hard-coded metrics (87.3% accuracy)** and made-up feature importance | Not honest reporting |
| Reproducible, honest reporting | label says "Cloud Computing" course (23DC2043), not AutoML | Wrong course |

The brief says marks depend on *honest reporting*, so the fake numbers in the old app would hurt you. The topic (heart-risk) is still reasonable, which is why this
project keeps it but with a real, larger dataset (Framingham, 4,240 rows) and a correct pipeline.

## 2. What is already done in this folder
Everything in Part B except things only you can do (see §3). All numbers in the report were produced by the code here
(AutoML budget used: **240 s per tool, 2 CPU cores** — shorter than the 15–30 min the brief suggests; see §4).

Main findings (held-out test AUC): Logistic Regression 0.702, FLAML 0.688, AutoGluon 0.678, tuned RF 0.674, default RF 0.632.
95 % bootstrap intervals overlap, so there is no significant winner. Both AutoML tools exploited the injected leaky feature
(AUC ≈ 0.63–0.69 → ≈ 0.95–0.97). The model is strongly age-driven, and almost never flags patients under 45 at the default threshold.

## 3. What only YOU can/must do
1. **Part A (10 marks): the online certification** — enrol, finish 100 %, submit the certificate to your instructor (details "will update soon" in the PDF).
2. **Get the topic approved by your instructor** and make sure no teammate uses the same topic. If they reject heart-risk, tell me the new topic and dataset; the pipeline in `src/` is generic (change `DATA`, `TARGET`, feature names in `app/`).
3. **Team of 3–4**: write team names in the report (set `TEAM="Names..."` env var before running `07_build_assets.py`, or edit the cover page).
4. **Understand it.** The brief penalises code you can't explain. Read §6 and each script's docstring; you must be able to answer questions on any part.
5. **Re-run on your laptop** (§4) so the results are yours and use the full time budget.
6. **Record a demo video or deploy online** (§5) — the deliverable list asks for a link or recording.
7. Put code on **GitHub** (or submit the notebook/scripts).

## 4. Steps in VS Code
1. Unzip, `File → Open Folder…` → `automl-heart-risk`.
2. Terminal (`Ctrl+\``):
   ```
   python -m venv .venv
   .venv\Scripts\activate            (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
   If `autogluon.tabular` fails to install, use Python 3.11 or 3.12 (`py -3.12 -m venv .venv`).
3. Run the steps one by one (so you see each output), or use `run_all.bat` / `run_all.sh`:
   ```
   python src/01_eda.py
   python src/02_baseline_tuning.py    # ~10 min
   set AUTOML_BUDGET=900               (Linux/macOS: export AUTOML_BUDGET=900)
   python src/03_automl.py             # 2 x 15 min
   python src/04_evaluation.py
   python src/05_leakage.py            # 2 x 7.5 min
   python src/06_export_model.py
   python src/07_build_assets.py       # report PDF, slides, model card
   ```
   Your numbers will differ slightly from mine (different hardware/time-based search). **That is fine — the report text and tables are regenerated from your results.**
   Read the new numbers and check the sentences in the report still match (e.g. "no significant winner"); edit `src/07_build_assets.py` wording if your results change the story. Honest reporting > high accuracy.
4. Quick test (~25 min total): `set AUTOML_BUDGET=240`.

## 5. Demo / deployment
* Local: `streamlit run app/streamlit_app.py` and `uvicorn app.api:app --reload` (open `/docs`).
* Online (free): GitHub repo → https://share.streamlit.io → main file `app/streamlit_app.py`. Pin your sklearn version in `app/requirements.txt`.
* Or record a 1–2 min screen video of the Streamlit app (prediction + SHAP chart) and the `/docs` page.

## 6. Likely viva questions (know these answers)
* *Why ROC-AUC, not accuracy?* 85 % of patients are negative; predicting all-negative gives 85 % accuracy.
* *How did you avoid leakage?* Split first; imputer inside the Pipeline (fit per fold); tuning only on train CV; test used once; threshold chosen on out-of-fold train predictions.
* *Why did the tuned RF score lower on test than in CV?* Test set has only 129 positives; AUC standard error ≈ 0.025 (bootstrap CI ±0.05). It is noise-level, not necessarily overfitting. Default RF though genuinely overfit (train AUC 1.0).
* *Grid vs Random vs Bayesian?* Grid: exhaustive on a fixed small grid; Random: samples wide ranges; Optuna/TPE: uses past trials to pick the next. Same 36 evaluations each; all within CV noise.
* *What did AutoML do for you?* Model/hyper-parameter search, ensembling, imputation; it did not engineer new features here and did not detect the leak.
* *Why not deploy the AutoGluon model?* Heavy 29-model ensemble, large dependency, harder to explain; CV rule (set before testing) picked the simple tuned RF.
* *What does SHAP show?* Age and systolic BP dominate; directions are clinically plausible.
* *Limits?* Weak signal (AUC ≈ 0.7), single-town 1948 cohort, small test set, short AutoML budget, not calibrated, not for clinical use.

## 7. Things I could not verify
* I could not access OpenML/UCI/Kaggle from my workspace, so the dataset came from a public GitHub copy of the Framingham CSV. If your instructor wants a specific source (Kaggle "Framingham"), swap `data/framingham.csv` (same columns) and re-run.
* AutoGluon's neural-network models were unavailable in my environment; on your laptop they may run (if torch installs), which may change the leaderboard.
* No Part A certificate, instructor approval, team names or live cloud URL — those need you.
