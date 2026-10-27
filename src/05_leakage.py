"""Step 4b: deliberately inject a leaky feature and test whether the AutoML tools exploit it.
Leaky feature 'cardiac_medication_started' = the true outcome with 5% of values flipped.
In real life this is information recorded AFTER the outcome (e.g. a statin/cardiac drug started
after a heart event), so it would NOT exist when we need to predict risk."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from automl_runners import *

Xtr, Xte, ytr, yte = get_split()
rng = np.random.RandomState(SEED)
flip = lambda y: np.where(rng.rand(len(y)) < 0.05, 1 - y.values, y.values)
LEAK = "cardiac_medication_started"
Xtr_l, Xte_l = Xtr.copy(), Xte.copy()
Xtr_l[LEAK] = flip(ytr); Xte_l[LEAK] = flip(yte)
half = max(BUDGET // 2, 60)

# detection screen: AUC of every single feature on TRAIN data
screen = pd.Series({c: roc_auc_score(ytr, Xtr_l[c].fillna(Xtr_l[c].median())) for c in Xtr_l.columns})
screen = (0.5 + (screen - 0.5).abs()).sort_values(ascending=False)
screen.round(4).to_csv(os.path.join(RES, "leak_single_feature_auc.csv"), header=["single_feature_auc"])
print(screen.head(5))


def perm_drop(predict, X, y, col, reps=10):
    base = roc_auc_score(y, predict(X)); d = []
    for r in range(reps):
        Xp = X.copy(); Xp[col] = np.random.RandomState(r).permutation(Xp[col].values)
        d.append(base - roc_auc_score(y, predict(Xp)))
    return float(np.mean(d))

res = {}
def record(name, predict_clean, predict_leak, extra=None):
    res[name] = {"auc_clean": roc_auc_score(yte, predict_clean(Xte)), "auc_with_leak": roc_auc_score(yte, predict_leak(Xte_l)),
                 "leak_permutation_AUC_drop": perm_drop(predict_leak, Xte_l, yte, LEAK), **(extra or {})}
    print(name, res[name], flush=True)

# RF default (manual)
mk = lambda: Pipeline([("i", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS))])
c, l = mk().fit(Xtr, ytr), mk().fit(Xtr_l, ytr)
record("RandomForest (manual)", lambda X: c.predict_proba(X)[:, 1], lambda X: l.predict_proba(X)[:, 1])

# FLAML (clean model from step 3 is re-used for the clean number)
import joblib
fc = joblib.load(os.path.join(MODEL_DIR, "flaml_automl.joblib"))
fl, pr, lb, info = run_flaml(Xtr_l, ytr, Xte_l, half, "leak")
record("FLAML", lambda X: fc.predict_proba(X[Xtr.columns])[:, 1], lambda X: fl.predict_proba(X)[:, 1], {"best_estimator_with_leak": info["best_estimator"], "budget_s": half})

# AutoGluon
agc = os.path.join(RES, "ag_clean")
from autogluon.tabular import TabularPredictor
pc = TabularPredictor.load(agc)
pl, pr, lb, info = run_autogluon(Xtr_l, ytr, Xte_l, yte, half, "leak")
record("AutoGluon", lambda X: pc.predict_proba(X[Xtr.columns])[1].values, lambda X: pl.predict_proba(X)[1].values, {"best_model_with_leak": info["best_model"], "budget_s": half})
try:
    fi = pl.feature_importance(Xte_l.assign(**{TARGET: yte.values}), silent=True)
    res["AutoGluon"]["native_importance_rank_of_leak"] = int(list(fi.index).index(LEAK) + 1)
except Exception as e:
    print("AG native importance failed", e)

save_json(res, "leakage_results.json")
t = pd.DataFrame(res).T; t.to_csv(os.path.join(RES, "leakage_results.csv")); print(t)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
x = np.arange(len(t)); ax[0].bar(x - .2, t.auc_clean, .4, label="clean", color="#4C78A8"); ax[0].bar(x + .2, t.auc_with_leak, .4, label="with leaky feature", color="#E45756")
ax[0].set_xticks(x); ax[0].set_xticklabels(t.index, fontsize=8); ax[0].set_ylim(.5, 1); ax[0].set_ylabel("Test ROC-AUC"); ax[0].legend(); ax[0].set_title("Test AUC jumps when a leak is present")
ax[1].bar(t.index, t.leak_permutation_AUC_drop, color="#E45756"); ax[1].set_title("AUC lost when the leaky feature is shuffled"); ax[1].tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "leakage_experiment.png"), dpi=140)
