"""Step 2: manual baseline + Grid / Random / Optuna tuning of the SAME model.

Model: RandomForestClassifier inside a Pipeline (median imputation is fitted
inside every CV fold -> no preprocessing leakage). All three tuners get the
same number of evaluations (36) and the same 5-fold stratified CV on TRAIN only.
The held-out test set is used once, for the final score of each tuned model.
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_val_predict)
from scipy.stats import randint, uniform
from common import *

optuna.logging.set_verbosity(optuna.logging.WARNING)
Xtr, Xte, ytr, yte = get_split()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
N_EVALS = 36


def make_rf(**p):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **p))])


rows, params_out = [], {}

# ---- reference models -------------------------------------------------------
refs = {
    "Dummy (prior)": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", DummyClassifier(strategy="prior"))]),
    "Logistic Regression": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
    "RandomForest (default) = BASELINE": make_rf(),
}
for name, m in refs.items():
    with Timer() as t:
        cvs = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
        m.fit(Xtr, ytr)
    pr = m.predict_proba(Xte)[:, 1]
    r = evaluate(yte, pr)
    rows.append({"method": name, "cv_auc_mean": cvs.mean(), "cv_auc_std": cvs.std(),
                 "test_auc": r["roc_auc"], "test_f1@0.5": r["f1"], "time_s": t.seconds, "evals": 5})
    print(rows[-1])

# ---- 1. Grid search ---------------------------------------------------------
grid = {"rf__n_estimators": [100, 300], "rf__max_depth": [4, 8, 12],
        "rf__min_samples_leaf": [1, 5, 10], "rf__max_features": ["sqrt", 0.5]}  # 2*3*3*2 = 36
with Timer() as t:
    gs = GridSearchCV(make_rf(), grid, cv=cv, scoring="roc_auc", n_jobs=1).fit(Xtr, ytr)
r = evaluate(yte, gs.predict_proba(Xte)[:, 1])
rows.append({"method": "Grid Search", "cv_auc_mean": gs.best_score_, "cv_auc_std": gs.cv_results_["std_test_score"][gs.best_index_],
             "test_auc": r["roc_auc"], "test_f1@0.5": r["f1"], "time_s": t.seconds, "evals": 36})
params_out["Grid Search"] = {k.replace("rf__", ""): v for k, v in gs.best_params_.items()}
print(rows[-1])

# ---- 2. Random search -------------------------------------------------------
dist = {"rf__n_estimators": randint(100, 500), "rf__max_depth": randint(3, 20),
        "rf__min_samples_leaf": randint(1, 20), "rf__max_features": uniform(0.2, 0.7)}
with Timer() as t:
    rs = RandomizedSearchCV(make_rf(), dist, n_iter=N_EVALS, cv=cv, scoring="roc_auc",
                            n_jobs=1, random_state=SEED).fit(Xtr, ytr)
r = evaluate(yte, rs.predict_proba(Xte)[:, 1])
rows.append({"method": "Random Search", "cv_auc_mean": rs.best_score_, "cv_auc_std": rs.cv_results_["std_test_score"][rs.best_index_],
             "test_auc": r["roc_auc"], "test_f1@0.5": r["f1"], "time_s": t.seconds, "evals": N_EVALS})
params_out["Random Search"] = {k.replace("rf__", ""): (float(v) if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
print(rows[-1])

# ---- 3. Bayesian optimisation (Optuna TPE) ---------------------------------
def objective(trial):
    p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 500),
             max_depth=trial.suggest_int("max_depth", 3, 20),
             min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
             max_features=trial.suggest_float("max_features", 0.2, 0.9))
    return cross_val_score(make_rf(**p), Xtr, ytr, cv=cv, scoring="roc_auc").mean()

with Timer() as t:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_EVALS)
best = make_rf(**study.best_params).fit(Xtr, ytr)
r = evaluate(yte, best.predict_proba(Xte)[:, 1])
rows.append({"method": "Optuna (Bayesian/TPE)", "cv_auc_mean": study.best_value, "cv_auc_std": float("nan"),
             "test_auc": r["roc_auc"], "test_f1@0.5": r["f1"], "time_s": t.seconds, "evals": N_EVALS})
params_out["Optuna (Bayesian/TPE)"] = study.best_params
print(rows[-1])
trace = [max(tr.value for tr in study.trials[:i + 1]) for i in range(len(study.trials))]

# ---- save -------------------------------------------------------------------
tab = pd.DataFrame(rows).round(4)
tab.to_csv(os.path.join(RES, "tuning_comparison.csv"), index=False)
save_json(params_out, "tuning_best_params.json")
save_json({"optuna_trace": trace}, "optuna_trace.json")
print("\n", tab.to_string(index=False))

# threshold for F1 chosen from out-of-fold predictions on TRAIN only
final_p = params_out["Optuna (Bayesian/TPE)"]
oof = cross_val_predict(make_rf(**final_p), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
save_json({"f1_threshold_from_train_oof": best_f1_threshold(ytr, oof)}, "threshold.json")

# ---- figure -----------------------------------------------------------------
t4 = tab[tab.method.isin(["RandomForest (default) = BASELINE", "Grid Search", "Random Search", "Optuna (Bayesian/TPE)"])]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
lab = ["Baseline", "Grid", "Random", "Optuna"]; col = ["#9ecae9", "#4C78A8", "#F58518", "#54A24B"]
ax[0].bar(lab, t4.cv_auc_mean, color=col); ax[0].set_ylim(0.6, 0.75); ax[0].set_title("5-fold CV ROC-AUC (train)")
ax[1].bar(lab, t4.test_auc, color=col); ax[1].set_ylim(0.6, 0.75); ax[1].set_title("Held-out test ROC-AUC")
ax[2].bar(lab, t4.time_s, color=col); ax[2].set_title("Wall-clock time (s)")
for a in ax:
    for p in a.patches: a.annotate(f"{p.get_height():.3f}" if p.get_height() < 5 else f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "tuning_comparison.png"), dpi=140)
plt.figure(figsize=(5.5, 3.5)); plt.plot(trace, marker="o", ms=3); plt.xlabel("Optuna trial"); plt.ylabel("best CV AUC so far"); plt.title("Optuna convergence"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "optuna_convergence.png"), dpi=140)
