"""Step 5a: choose and export the model to deploy.

Selection rule (made BEFORE looking at the test set, using train-only CV scores):
  - candidates: Optuna-tuned RandomForest pipeline, FLAML best model
  - deploy the tuned sklearn pipeline UNLESS another model beats its CV AUC by > 0.01
    (CV std is ~0.03, so smaller differences are noise; we then prefer the simpler,
    exactly-explainable sklearn model with no extra runtime dependencies).
AutoGluon is not deployed: it is a multi-model directory that needs the whole autogluon stack at run time.
"""
import warnings; warnings.filterwarnings("ignore")
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from common import *

Xtr, Xte, ytr, yte = get_split()
P = load_json("tuning_best_params.json")["Optuna (Bayesian/TPE)"]
thr = load_json("threshold.json")["f1_threshold_from_train_oof"]
tune = pd.read_csv(os.path.join(RES, "tuning_comparison.csv")).set_index("method")
rf_cv = float(tune.loc["Optuna (Bayesian/TPE)", "cv_auc_mean"])
fl_cv = load_json("automl_results.json")["FLAML"]["best_cv_auc"]
print(f"CV AUC  tuned RF={rf_cv:.4f}  FLAML={fl_cv:.4f}")
if fl_cv - rf_cv > 0.01:
    print("NOTE: FLAML beats tuned RF by > 0.01 CV AUC; selection rule says deploy FLAML. "
          "Adapt this script if that happens on your machine.")

# final model is refit on the TRAIN split (same data the test score was measured on)
model = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("rf", RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **P))]).fit(Xtr, ytr)
joblib.dump(model, os.path.join(MODEL_DIR, "heart_risk_pipeline.joblib"))
rep = load_json("final_model_report.json")
meta = {
    "model_name": "RandomForest (Optuna-tuned) in sklearn Pipeline (median imputer + RF)",
    "features": list(Xtr.columns), "threshold": thr, "hyperparameters": P, "seed": SEED,
    "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "positive_rate_train": float(ytr.mean()),
    "cv_auc": rf_cv, "test_metrics_default_threshold": rep["final_model_test_default_thr"],
    "test_metrics_tuned_threshold": rep["final_model_test_tuned_thr"],
    "feature_ranges": {c: [float(Xtr[c].min()), float(Xtr[c].max())] for c in Xtr.columns},
    "feature_medians": {c: float(Xtr[c].median()) for c in Xtr.columns},
}
save_json(meta, "../model/model_meta.json") if False else json.dump(meta, open(os.path.join(MODEL_DIR, "model_meta.json"), "w"), indent=2)
print("exported ->", MODEL_DIR)
