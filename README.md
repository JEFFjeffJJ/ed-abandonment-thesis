# ED Abandonment Thesis Source Code

Source-code snapshot for **Modeling and Predicting Patient Abandonment in
Emergency Department Service Systems**, prepared on 2026-11-02.

The main outcome is broad abandonment: `LEFT WITHOUT BEING SEEN`, `ELOPED`, or
`LEFT AGAINST MEDICAL ADVICE`.

This repository contains Python and R implementation files, synthetic tests,
the pipeline runner, and a historical EDA notebook with saved outputs removed.
It excludes MIMIC data, processed patient records, model bundles, predictions,
generated tables and figures, thesis documents, and local environments.

## Code Reference

| File | Purpose |
| --- | --- |
| `src/data_process.py` | Prepare the ED-stay analysis dataset and predictors. |
| `src/processor.py` | Patient-separated splits, predictive models, calibration, and evaluation. |
| `src/regression_analysis.R` | Adjusted and sensitivity regression analyses. |
| `src/scenario_analysis.R` | Model-based scenario analyses. |
| `src/descriptive_analysis.py` | Descriptive analysis of the processed cohort. |
| `src/result_figures.py` | Generate figures from saved analysis results. |
| `src/thesis_tables.py` | Generate aggregate LaTeX tables for the local thesis. |
| `src/verify_pipeline.py` | Verify cohort, predictions, and model evaluation consistency. |
| `src/verify_regression.R` | Verify regression and scenario results. |
| `tests/test_pipeline.py` | Synthetic processing and prediction checks. |
| `thesis/tum_thesis_v3/analysis/revision_analysis.py` | Additional return-timing analyses and revision tables. |
| `thesis/tum_thesis_v3/analysis/revision_figures.py` | Figures for the thesis revision. |
| `notebooks/01_eda_dataset_overview.ipynb` | Historical EDA; its outcome emphasis predates the current pipeline. |
| `run_pipeline.sh` | Original local pipeline runner, including thesis-build steps. |

File paths are preserved from the working project. The Python and R scripts and
runner are copied without implementation changes. Only the notebook's execution
outputs, execution counts, and output-related metadata are cleared in this copy.

## Local Use

Use an existing Python environment with pandas, NumPy, scikit-learn, Matplotlib,
and joblib. The R scripts use base R. No packages are installed by these files.

Run the synthetic checks without accessing MIMIC data:

```bash
python -m unittest discover -s tests -v
```

To run the analysis, place your separately authorized MIMIC-IV-ED 2.2 inputs
under `data/raw/mimic-iv-ed-2.2/ed/`, then run the individual scripts from the
repository root. The preprocessing step uses `edstays.csv.gz` and
`triage.csv.gz`.

```bash
python src/data_process.py
python src/processor.py
Rscript src/regression_analysis.R
Rscript src/scenario_analysis.R
```

The full runner also builds a local LaTeX thesis whose document sources are
excluded from this repository, so `bash run_pipeline.sh all` requires the full
local thesis project. The scenario code evaluates fitted-model scenarios; it
does not implement a discrete-event queue simulation.

For a stable reference, link to the relevant file at a specific commit using
GitHub's file permalink.
