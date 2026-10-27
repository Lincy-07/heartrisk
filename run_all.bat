@echo off
REM Reproduce everything on Windows (run inside the activated venv).
REM Change 900 to 1800 for a 30-minute budget per AutoML tool.
if "%AUTOML_BUDGET%"=="" set AUTOML_BUDGET=900
python src/01_eda.py || exit /b 1
python src/02_baseline_tuning.py || exit /b 1
python src/03_automl.py || exit /b 1
python src/04_evaluation.py || exit /b 1
python src/05_leakage.py || exit /b 1
python src/06_export_model.py || exit /b 1
python src/07_build_assets.py || exit /b 1
