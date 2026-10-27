"""Wrappers so Step 3 (clean data) and Step 4 (leaky data) use identical settings."""
import os, shutil, pickle
import numpy as np, pandas as pd
from common import *


def run_flaml(Xtr, ytr, Xte, budget, tag):
    from flaml import AutoML
    a = AutoML()
    log = os.path.join(RES, f"flaml_{tag}.log")
    if os.path.exists(log): os.remove(log)
    with Timer() as t:
        a.fit(Xtr, ytr, task="classification", metric="roc_auc", time_budget=budget,
              eval_method="cv", n_splits=5, seed=SEED, n_jobs=N_JOBS, log_file_name=log, verbose=0)
    proba = a.predict_proba(Xte)[:, 1]
    # leaderboard: best validation AUC reached by each learner family
    lb = pd.DataFrame([{"model": k, "cv_auc": 1 - v if v is not None else np.nan}
                       for k, v in a.best_loss_per_estimator.items()]).sort_values("cv_auc", ascending=False)
    # how many configurations of each learner were tried (from the search log)
    import json
    tried = {}
    if os.path.exists(log):
        for line in open(log):
            try:
                rec = json.loads(line); l = rec.get("learner"); tried[l] = tried.get(l, 0) + 1
            except Exception:
                pass
    lb["improving_configs_logged"] = lb.model.map(tried)
    info = {"best_estimator": a.best_estimator, "best_config": a.best_config,
            "best_cv_auc": 1 - a.best_loss, "fit_seconds": t.seconds, "n_improving_configs_logged": int(sum(tried.values()))}
    return a, proba, lb, info


def run_autogluon(Xtr, ytr, Xte, yte, budget, tag):
    from autogluon.tabular import TabularPredictor
    path = os.path.join(RES, f"ag_{tag}")
    shutil.rmtree(path, ignore_errors=True)
    train = Xtr.copy(); train[TARGET] = ytr.values
    with Timer() as t:
        p = TabularPredictor(label=TARGET, eval_metric="roc_auc", path=path, verbosity=0).fit(
            train, time_limit=budget, presets="best_quality", num_cpus=N_JOBS)
    proba = p.predict_proba(Xte)[1].values
    test = Xte.copy(); test[TARGET] = yte.values
    lb = p.leaderboard(test, silent=True)[["model", "score_test", "score_val", "fit_time", "pred_time_val", "stack_level"]]
    info = {"best_model": p.model_best, "n_models": int(len(lb)), "fit_seconds": t.seconds,
            "features_used": list(p.features()), "problem_type": p.problem_type,
            "feature_generator": str(type(p._learner.feature_generator).__name__)}
    return p, proba, lb, info
