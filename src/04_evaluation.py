"""Step 4a: overfitting check, bootstrap CIs, class imbalance, subgroup bias, SHAP.
Final model checked here = Optuna-tuned RandomForest (the sklearn pipeline we deploy candidates from)."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from common import *

Xtr, Xte, ytr, yte = get_split()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
P = load_json("tuning_best_params.json")["Optuna (Bayesian/TPE)"]
thr = load_json("threshold.json")["f1_threshold_from_train_oof"]


def rf(**p):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **p))])

report = {}
# ---- 1. overfitting: train vs CV vs test ------------------------------------
over = []
for name, m in {"RF default (baseline)": rf(), "RF tuned (Optuna)": rf(**P)}.items():
    cvr = cross_validate(m, Xtr, ytr, cv=cv, scoring="roc_auc", return_train_score=True)
    m.fit(Xtr, ytr)
    over.append({"model": name, "train_auc_in_cv": cvr["train_score"].mean(), "cv_val_auc": cvr["test_score"].mean(),
                 "train_auc_full_fit": roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]),
                 "test_auc": roc_auc_score(yte, m.predict_proba(Xte)[:, 1])})
over = pd.DataFrame(over).round(4); over.to_csv(os.path.join(RES, "overfitting_check.csv"), index=False); print(over.to_string(index=False))

final = rf(**P).fit(Xtr, ytr)
pt = final.predict_proba(Xte)[:, 1]
report["final_model_test_default_thr"] = evaluate(yte, pt, 0.5)
report["final_model_test_tuned_thr"] = evaluate(yte, pt, thr)
report["f1_threshold_from_train_oof"] = thr
print(report)

# ---- 2. bootstrap CI of test AUC for every model ------------------------------
preds = pd.read_csv(os.path.join(RES, "preds_automl.csv"), index_col=0)
preds["rf_tuned"] = pt
preds["rf_default"] = rf().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
preds["logreg"] = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
rng = np.random.RandomState(SEED); y = yte.values; ci = []
idx = [rng.randint(0, len(y), len(y)) for _ in range(1000)]
for c in ["logreg", "rf_default", "rf_tuned", "flaml", "autogluon"]:
    s = [roc_auc_score(y[i], preds[c].values[i]) for i in idx]
    ci.append({"model": c, "test_auc": roc_auc_score(y, preds[c]), "ci_low": np.percentile(s, 2.5), "ci_high": np.percentile(s, 97.5)})
ci = pd.DataFrame(ci).round(4); ci.to_csv(os.path.join(RES, "test_auc_bootstrap_ci.csv"), index=False); print(ci.to_string(index=False))
preds.to_csv(os.path.join(RES, "preds_all.csv"))

plt.figure(figsize=(5.8, 5))
for c, lab in [("logreg", "LogReg"), ("rf_default", "RF default"), ("rf_tuned", "RF tuned"), ("flaml", "FLAML"), ("autogluon", "AutoGluon")]:
    f, t, _ = roc_curve(y, preds[c]); plt.plot(f, t, label=f"{lab} ({roc_auc_score(y, preds[c]):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title("ROC on held-out test set")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "roc_all_models.png"), dpi=140)

# ---- 3. class imbalance -------------------------------------------------------
imb = []
for name, kw in {"no weighting": {}, "class_weight='balanced'": {"class_weight": "balanced"}}.items():
    m = rf(**P, **kw) if False else Pipeline([("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **P, **kw))])
    cvauc = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean(); m.fit(Xtr, ytr)
    r = evaluate(yte, m.predict_proba(Xte)[:, 1], 0.5)
    imb.append({"setting": name, "cv_auc": cvauc, "test_auc": r["roc_auc"], "pr_auc": r["pr_auc"], "recall@0.5": r["recall"], "precision@0.5": r["precision"], "f1@0.5": r["f1"]})
imb = pd.DataFrame(imb).round(4); imb.to_csv(os.path.join(RES, "imbalance_check.csv"), index=False); print(imb.to_string(index=False))

# ---- 4. subgroup bias at the chosen operating threshold -----------------------
grp = pd.DataFrame({"y": y, "p": pt, "sex": np.where(Xte["male"].values == 1, "Male", "Female"),
                    "age": pd.cut(Xte["age"].values, [0, 44, 54, 100], labels=["<45", "45-54", "55+"])})
rowsb = []
for gname in ["sex", "age"]:
    for lv, g in grp.groupby(gname, observed=True):
        pred = (g.p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(g.y, pred, labels=[0, 1]).ravel()
        rowsb.append({"attribute": gname, "group": str(lv), "n": len(g), "prevalence": g.y.mean(),
                      "auc": roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan,
                      "selection_rate": pred.mean(), "recall_TPR": tp / max(tp + fn, 1), "FPR": fp / max(fp + tn, 1)})
bias = pd.DataFrame(rowsb).round(3); bias.to_csv(os.path.join(RES, "subgroup_bias.csv"), index=False); print(bias.to_string(index=False))

# ---- 5. SHAP ------------------------------------------------------------------
imp = final.named_steps["imp"]; Xte_i = pd.DataFrame(imp.transform(Xte), columns=Xte.columns, index=Xte.index)
expl = shap.TreeExplainer(final.named_steps["rf"])
sv = expl(Xte_i); sv1 = sv[:, :, 1] if sv.values.ndim == 3 else sv
plt.figure(); shap.plots.beeswarm(sv1, show=False, max_display=15); plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_beeswarm.png"), dpi=140, bbox_inches="tight"); plt.close()
plt.figure(); shap.plots.bar(sv1, show=False, max_display=15); plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_bar.png"), dpi=140, bbox_inches="tight"); plt.close()
hi = int(np.argmax(pt)); plt.figure(); shap.plots.waterfall(sv1[hi], show=False, max_display=10); plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_local_example.png"), dpi=140, bbox_inches="tight"); plt.close()
mean_abs = pd.Series(np.abs(sv1.values).mean(0), index=Xte.columns).sort_values(ascending=False)
pi = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=1)
pim = pd.Series(pi.importances_mean, index=Xte.columns)
imp_tab = pd.DataFrame({"mean_|SHAP|": mean_abs, "permutation_AUC_drop": pim.reindex(mean_abs.index)}).round(4)
imp_tab.to_csv(os.path.join(RES, "feature_importance_shap_perm.csv")); print(imp_tab)

# ---- 6. calibration -----------------------------------------------------------
fp_, mp_ = calibration_curve(y, pt, n_bins=8, strategy="quantile")
plt.figure(figsize=(4.8, 4.6)); plt.plot(mp_, fp_, "o-", label="tuned RF"); plt.plot([0, .6], [0, .6], "k--"); plt.xlabel("Mean predicted risk"); plt.ylabel("Observed CHD rate"); plt.title("Calibration (test)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG, "calibration.png"), dpi=140)
save_json(report, "final_model_report.json")
