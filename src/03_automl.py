"""Step 3: FLAML and AutoGluon on the same train/test split, same time budget."""
import warnings; warnings.filterwarnings("ignore")
import sys, joblib
from automl_runners import *

Xtr, Xte, ytr, yte = get_split()
print(f"Budget per tool: {BUDGET}s", flush=True)
out = {}
preds = pd.DataFrame({"y_true": yte.values}, index=yte.index)

a, pr, lb, info = run_flaml(Xtr, ytr, Xte, BUDGET, "clean")
lb.to_csv(os.path.join(RES, "leaderboard_flaml.csv"), index=False)
preds["flaml"] = pr
out["FLAML"] = {**info, **evaluate(yte, pr)}
joblib.dump(a, os.path.join(MODEL_DIR, "flaml_automl.joblib"))
print("FLAML", out["FLAML"], flush=True); print(lb, flush=True)

p, pr, lb, info = run_autogluon(Xtr, ytr, Xte, yte, BUDGET, "clean")
lb.to_csv(os.path.join(RES, "leaderboard_autogluon.csv"), index=False)
preds["autogluon"] = pr
out["AutoGluon"] = {**info, **evaluate(yte, pr)}
print("AutoGluon", out["AutoGluon"], flush=True); print(lb, flush=True)

out["budget_seconds"] = BUDGET
save_json(out, "automl_results.json")
preds.to_csv(os.path.join(RES, "preds_automl.csv"))
