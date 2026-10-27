#!/usr/bin/env bash
# Reproduce everything (Linux/macOS/Git-Bash). Windows: use run_all.bat
# AUTOML_BUDGET = seconds per AutoML tool. Brief suggests 900-1800. Default here: 900.
set -e
export AUTOML_BUDGET=${AUTOML_BUDGET:-900}
python src/01_eda.py
python src/02_baseline_tuning.py      # ~9 min on 2 cores
python src/03_automl.py               # 2 x AUTOML_BUDGET
python src/04_evaluation.py
python src/05_leakage.py              # 2 x AUTOML_BUDGET/2
python src/06_export_model.py
python src/07_build_assets.py         # figures, MODEL_CARD.md, report PDF, slides
