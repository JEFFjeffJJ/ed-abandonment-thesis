"""Check consistency of this run without exposing visit-level records."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from processor import (predict_bundle, evaluate_model, CORE_PREDICTORS,
                       COHORTS, MODEL_NAMES, BOOTSTRAP_REPEATS, IMPORTANCE_REPEATS)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"


def main():
    data_file = DATA / "abandonment_modeling_data.csv.gz"
    data = pd.read_csv(data_file)
    raw = pd.read_csv(ROOT / "data/raw/mimic-iv-ed-2.2/ed/edstays.csv.gz",
                      usecols=["subject_id", "stay_id", "intime", "disposition"])
    triage = pd.read_csv(ROOT / "data/raw/mimic-iv-ed-2.2/ed/triage.csv.gz",
                         usecols=["subject_id", "stay_id"])
    assert raw.stay_id.is_unique and triage.stay_id.is_unique and data.stay_id.is_unique
    assert raw.stay_id.isin(triage.stay_id).all()
    assert len(raw) == len(data)
    np.testing.assert_array_equal(data.abandonment, data[["lwbs", "eloped", "ama"]].sum(axis=1))
    assert data.abandonment.isin([0, 1]).all()
    # Minimum rank within each patient's arrival times is exactly the strict count.
    expected = (pd.to_datetime(raw.intime).groupby(raw.subject_id)
                .rank(method="min").sub(1).astype(int))
    counts = raw[["stay_id"]].assign(expected=expected).merge(
        data[["stay_id", "previous_ed_visits"]], on="stay_id", validate="one_to_one")
    np.testing.assert_array_equal(counts.expected, counts.previous_ed_visits)
    tied = raw.duplicated(["subject_id", "intime"], keep=False).sum()
    membership = pd.read_csv(DATA / "patient_split_membership.csv.gz")
    assert membership.subject_id.is_unique
    assert data.subject_id.isin(membership.subject_id).all()
    assigned = data.merge(membership, on="subject_id", validate="many_to_one")
    training = pd.read_csv(DATA / "training_data.csv.gz")
    testing = pd.read_csv(DATA / "testing_data.csv.gz")
    assert not set(training.subject_id) & set(testing.subject_id)
    assert len(training) + len(testing) == len(data)
    manifest = json.loads((DATA / "run_manifest.json").read_text())
    assert manifest["data_sha256"] == hashlib.sha256(data_file.read_bytes()).hexdigest()
    assert manifest["outcome_definition"] == [
        "LEFT WITHOUT BEING SEEN", "ELOPED", "LEFT AGAINST MEDICAL ADVICE"]
    for filename in ["data_process.py", "processor.py"]:
        assert manifest["source_sha256"][filename] == hashlib.sha256(
            (ROOT / "src" / filename).read_bytes()).hexdigest()
    metrics = pd.read_csv(TABLES / "model_evaluation_metrics.csv")
    imports = pd.read_csv(TABLES / "importance_replicates.csv")
    bootstrap = pd.read_csv(TABLES / "evaluation_bootstrap_replicates.csv")
    max_prediction_error = 0.0
    for cohort, transport in COHORTS.items():
        for name in MODEL_NAMES:
            bundle = joblib.load(DATA / "models" / f"{cohort}_{name}.joblib")
            expected_predictors = CORE_PREDICTORS + (["arrival_transport"] if transport else [])
            assert bundle["preprocessing"]["input_predictors"] == expected_predictors
            assert bundle["manifest"]["data_sha256"] == manifest["data_sha256"]
            predictions = pd.read_csv(DATA / "predictions" / f"{cohort}_{name}.csv.gz")
            visits = predictions[["stay_id"]].merge(data, on="stay_id", validate="one_to_one")
            roles = visits.merge(membership, on="subject_id", validate="many_to_one")
            assert roles.split.eq("testing").all()
            assert len(visits) == len(testing if cohort == "A_full" else
                                      testing[testing.arrival_transport.notna()])
            recreated = predict_bundle(bundle, visits)
            difference = float(np.max(np.abs(recreated-predictions.probability)))
            max_prediction_error = max(max_prediction_error, difference)
            np.testing.assert_allclose(recreated, predictions.probability, atol=1e-12)
            assert predictions.abandonment.tolist() == visits.abandonment.tolist()
            saved = metrics[(metrics.cohort == cohort) & (metrics.model == name) &
                            (metrics.threshold_rule == "calibration_f1")].iloc[0]
            recalculated = evaluate_model(visits.abandonment, recreated, bundle["threshold"])
            for metric, value in recalculated.items():
                np.testing.assert_allclose(value, saved[metric], atol=1e-12)
            assert len(bootstrap[(bootstrap.cohort == cohort) &
                                 (bootstrap.model == name)]) == BOOTSTRAP_REPEATS
            groups = imports[(imports.cohort == cohort) &
                             (imports.model == name)].groupby("predictor").size()
            assert len(groups) == len(expected_predictors)
            assert groups.eq(IMPORTANCE_REPEATS).all()
    scenarios = pd.read_csv(TABLES / "scenario_probabilities.csv")
    supported = scenarios[scenarios.display_supported]
    assert supported.support_visits.ge(100).all() and supported.support_events.ge(5).all()
    assert scenarios.confidence_interval_lower.le(scenarios.adjusted_probability).all()
    assert scenarios.confidence_interval_upper.ge(scenarios.adjusted_probability).all()
    assert set(scenarios.arrival_transport) <= {"ALL", "WALK IN", "AMBULANCE"}
    primary_split = pd.read_csv(TABLES / "split_summary.csv")
    for _, row in primary_split.iterrows():
        sample = assigned[assigned.split == row.split]
        if row.cohort != "A_full":
            sample = sample[sample.arrival_transport.notna()]
        assert len(sample) == row.visits and sample.subject_id.nunique() == row.patients
        assert sample.abandonment.sum() == row.events
    report = {"status": "passed", "visits": len(data), "patients": data.subject_id.nunique(),
              "broad_events": int(data.abandonment.sum()), "tied_arrival_records": int(tied),
              "patient_overlap": 0, "models_checked": 9,
              "maximum_prediction_roundtrip_error": max_prediction_error,
              "supported_scenario_points": len(supported),
              "data_sha256": manifest["data_sha256"]}
    (TABLES / "verification_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
