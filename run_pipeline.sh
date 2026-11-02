#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
MODE="${1:-all}"

build_thesis() {
  "$PYTHON" src/thesis_tables.py
  (
    cd thesis/tum_thesis_v2
    latexmk -pdf -silent -interaction=nonstopmode -halt-on-error main.tex
  )
}

case "$MODE" in
  all)
    echo "[1/10] Synthetic checks"
    "$PYTHON" -m unittest discover -s tests -v
    echo "[2/10] Full-cohort preprocessing"
    "$PYTHON" src/data_process.py
    echo "[3/10] Nine patient-separated predictive models and evaluation"
    "$PYTHON" -u src/processor.py
    echo "[4/10] Adjusted and sensitivity regressions"
    "$RSCRIPT" src/regression_analysis.R
    echo "[5/10] Supported model-based scenarios"
    "$RSCRIPT" src/scenario_analysis.R
    echo "[6/10] Regression and scenario verification"
    "$RSCRIPT" src/verify_regression.R
    echo "[7/10] Revised descriptive analysis"
    "$PYTHON" src/descriptive_analysis.py
    echo "[8/10] Figures from saved results"
    "$PYTHON" src/result_figures.py
    echo "[9/10] Prediction and cohort verification"
    "$PYTHON" src/verify_pipeline.py
    echo "[10/10] Aggregate thesis tables and TUM PDF"
    build_thesis
    ;;
  figures)
    "$PYTHON" src/result_figures.py
    build_thesis
    ;;
  thesis)
    build_thesis
    ;;
  verify)
    "$PYTHON" -m unittest discover -s tests -v
    "$PYTHON" src/verify_pipeline.py
    "$RSCRIPT" src/verify_regression.R
    ;;
  *)
    echo "Usage: bash run_pipeline.sh [all|figures|thesis|verify]" >&2
    exit 2
    ;;
esac
echo "Finished: $MODE"
echo "Canonical thesis PDF: $PROJECT_DIR/thesis/tum_thesis_v2/main.pdf"
