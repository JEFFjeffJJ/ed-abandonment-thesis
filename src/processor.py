"""Patient-held-out prediction, calibration, evaluation and grouped importance."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
MODELING_DATA_FILE = PROCESSED_DATA_DIR / "abandonment_modeling_data.csv.gz"
RANDOM_STATE = 42
TESTING_SIZE = 0.30
CALIBRATION_SIZE = 0.20
IMPORTANCE_REPEATS = 10
BOOTSTRAP_REPEATS = 200
CORE_PREDICTORS = ["previous_ed_visits", "triage_acuity", "triage_pain"]
COHORTS = {"A_full": False, "B_known": False, "C_transport": True}
MODEL_NAMES = ["logistic", "rf_unweighted", "rf_balanced"]
RF_SETTINGS = dict(n_estimators=200, max_depth=12, min_samples_leaf=20,
                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)


def load_modeling_data():
    return pd.read_csv(MODELING_DATA_FILE)


def split_by_patient(modeling_data):
    """Return raw training/testing visits; no patient appears in both."""
    outcomes = modeling_data.groupby("subject_id").abandonment.max()
    train_ids, test_ids = train_test_split(
        outcomes.index, test_size=TESTING_SIZE, random_state=RANDOM_STATE,
        stratify=outcomes.to_numpy())
    return (modeling_data[modeling_data.subject_id.isin(train_ids)].copy(),
            modeling_data[modeling_data.subject_id.isin(test_ids)].copy())


def assign_patient_splits(data):
    training, testing = split_by_patient(data)
    outcomes = training.groupby("subject_id").abandonment.max()
    fit_ids, calibration_ids = train_test_split(
        outcomes.index, test_size=CALIBRATION_SIZE, random_state=RANDOM_STATE + 1,
        stratify=outcomes.to_numpy())
    membership = pd.concat([
        pd.DataFrame({"subject_id": fit_ids, "split": "training_fit"}),
        pd.DataFrame({"subject_id": calibration_ids, "split": "training_calibration"}),
        pd.DataFrame({"subject_id": testing.subject_id.unique(), "split": "testing"})])
    assert membership.subject_id.is_unique
    membership.sort_values("subject_id").to_csv(
        PROCESSED_DATA_DIR / "patient_split_membership.csv.gz", index=False)
    training.merge(membership, on="subject_id", validate="many_to_one").to_csv(
        PROCESSED_DATA_DIR / "training_data.csv.gz", index=False)
    testing.to_csv(PROCESSED_DATA_DIR / "testing_data.csv.gz", index=False)
    return data.merge(membership, on="subject_id", validate="many_to_one")


def fit_preprocessor(training_data, include_transport=False):
    """Learn only from training-fit patients, never calibration/test patients."""
    medians = training_data[["triage_acuity", "triage_pain"]].median().to_dict()
    values = pd.DataFrame({
        "history": np.log1p(training_data.previous_ed_visits),
        "pain": training_data.triage_pain.fillna(medians["triage_pain"])})
    settings = {"medians": medians, "means": values.mean().to_dict(),
                "scales": values.std(ddof=0).replace(0, 1).to_dict(),
                "include_transport": include_transport,
                "transport_levels": (sorted(training_data.arrival_transport.dropna().unique())
                                     if include_transport else [])}
    settings["input_predictors"] = CORE_PREDICTORS + (
        ["arrival_transport"] if include_transport else [])
    return settings


def encode_predictors(data, settings):
    """Apply saved values and a fixed column order to raw predictor columns."""
    x = pd.DataFrame(index=data.index)
    history = np.log1p(data.previous_ed_visits)
    pain = data.triage_pain.fillna(settings["medians"]["triage_pain"])
    x["history_scaled"] = (history - settings["means"]["history"]) / settings["scales"]["history"]
    x["pain_scaled"] = (pain - settings["means"]["pain"]) / settings["scales"]["pain"]
    x["pain_missing"] = data.triage_pain.isna().astype(int)
    x["acuity_missing"] = data.triage_acuity.isna().astype(int)
    acuity = data.triage_acuity.fillna(settings["medians"]["triage_acuity"])
    for level in range(1, 6):
        x[f"acuity_{level}"] = acuity.eq(level).astype(int)
    for level in settings["transport_levels"]:
        x[f"transport_{level}"] = data.arrival_transport.eq(level).astype(int)
    return x.astype(float)


def predictor_groups(columns):
    return {name: [c for c in columns if c.startswith(prefix)]
            for name, prefix in [("Previous ED visits", "history_"),
                                 ("Triage pain", "pain_"), ("Triage acuity", "acuity_"),
                                 ("Arrival transport", "transport_")]
            if any(c.startswith(prefix) for c in columns)}


def build_model(name):
    if name == "logistic":
        return LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    return RandomForestClassifier(
        **RF_SETTINGS, class_weight="balanced" if name == "rf_balanced" else None)


def log_odds(probability):
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def predict_bundle(bundle, data, calibrated=True):
    encoded = encode_predictors(data, bundle["preprocessing"])
    assert list(encoded.columns) == bundle["feature_names"]
    probability = bundle["model"].predict_proba(encoded)[:, 1]
    if calibrated:
        probability = bundle["calibrator"].predict_proba(log_odds(probability))[:, 1]
    return probability


def choose_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(scores)])


def evaluate_model(y, probability, threshold, weights=None):
    prediction = probability >= threshold
    kwargs = {"sample_weight": weights}
    return {
        "auroc": roc_auc_score(y, probability, **kwargs),
        "average_precision": average_precision_score(y, probability, **kwargs),
        "brier": brier_score_loss(y, probability, **kwargs),
        "precision": precision_score(y, prediction, zero_division=0, **kwargs),
        "recall": recall_score(y, prediction, zero_division=0, **kwargs),
        "f1": f1_score(y, prediction, zero_division=0, **kwargs),
        "observed_rate": np.average(y, weights=weights),
        "mean_probability": np.average(probability, weights=weights),
    }


def bootstrap_evaluation(data, probability, threshold):
    """Resample patients with all their visits; fitted models are held fixed."""
    codes, patients = pd.factorize(data.subject_id, sort=True)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for repeat in range(BOOTSTRAP_REPEATS):
        multiplicities = np.bincount(
            rng.integers(0, len(patients), size=len(patients)), minlength=len(patients))
        weights = multiplicities[codes]
        rows.append({"repeat": repeat, **evaluate_model(
            data.abandonment, probability, threshold, weights)})
    replicates = pd.DataFrame(rows)
    intervals = replicates.drop(columns="repeat").quantile([0.025, 0.975]).T
    intervals.columns = ["ci_lower", "ci_upper"]
    return intervals.rename_axis("metric").reset_index(), replicates


def calculate_predictor_importance(bundle, data):
    """Permute raw groups, keeping values and missingness indicators together."""
    y = data.abandonment
    baseline = predict_bundle(bundle, data)
    baseline_ap = average_precision_score(y, baseline)
    baseline_auc = roc_auc_score(y, baseline)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    group_names = {"previous_ed_visits": "Previous ED visits",
                   "triage_acuity": "Triage acuity", "triage_pain": "Triage pain",
                   "arrival_transport": "Arrival transport"}
    for predictor in bundle["preprocessing"]["input_predictors"]:
        for repeat in range(IMPORTANCE_REPEATS):
            shuffled = data.copy()
            shuffled[predictor] = data[predictor].to_numpy()[rng.permutation(len(data))]
            probability = predict_bundle(bundle, shuffled)
            rows.append({"predictor": group_names[predictor], "repeat": repeat,
                         "ap_decrease": baseline_ap - average_precision_score(y, probability),
                         "auroc_decrease": baseline_auc - roc_auc_score(y, probability)})
    return pd.DataFrame(rows)


def calibration_results(y, raw, calibrated):
    rows = []
    points = []
    for label, probability in [("raw", raw), ("calibrated", calibrated)]:
        diagnostic = LogisticRegression(penalty=None, max_iter=2000).fit(
            log_odds(probability), y)
        rows.append({"probability": label, "intercept": diagnostic.intercept_[0],
                     "slope": diagnostic.coef_[0, 0],
                     "brier": brier_score_loss(y, probability),
                     "mean_probability": np.mean(probability), "observed_rate": np.mean(y)})
        frame = pd.DataFrame({"probability": probability, "outcome": np.asarray(y)})
        frame["bin"] = pd.qcut(frame.probability, 10, duplicates="drop")
        bins = frame.groupby("bin", observed=True).agg(
            visits=("outcome", "size"), events=("outcome", "sum"),
            mean_probability=("probability", "mean"), observed_rate=("outcome", "mean"))
        bins = bins.reset_index(drop=True)
        bins["probability_type"] = label
        points.append(bins)
    return pd.DataFrame(rows), pd.concat(points, ignore_index=True)


def save_result(rows, filename):
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT_TABLES_DIR / filename, index=False)


def main():
    start = perf_counter()
    data = assign_patient_splits(load_modeling_data())
    model_dir = PROCESSED_DATA_DIR / "models"
    prediction_dir = PROCESSED_DATA_DIR / "predictions"
    model_dir.mkdir(exist_ok=True)
    prediction_dir.mkdir(exist_ok=True)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "MIMIC-IV-ED v2.2",
                "outcome_definition": ["LEFT WITHOUT BEING SEEN", "ELOPED",
                                       "LEFT AGAINST MEDICAL ADVICE"],
                "prediction_target": "Recorded broad abandonment before the end of the visit",
                "intended_prediction_time": "After initial triage; exact availability not verified",
                "data_sha256": hashlib.sha256(MODELING_DATA_FILE.read_bytes()).hexdigest(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((PROJECT_DIR / "src").glob("*.py"))},
                "versions": versions, "seed": RANDOM_STATE, "test_fraction": TESTING_SIZE,
                "calibration_fraction_of_training": CALIBRATION_SIZE,
                "rf_settings": RF_SETTINGS, "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "forest_prediction_jobs": 1,
                "importance_repeats": IMPORTANCE_REPEATS,
                "primary": "A_full/rf_unweighted",
                "threshold_rule": "Maximum calibration-subset F1; no test selection"}
    (PROCESSED_DATA_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    metrics, intervals, bootstraps, importances = [], [], [], []
    calibrations, calibration_bins, confusion, splits = [], [], [], []
    for cohort, transport in COHORTS.items():
        sample = data if cohort == "A_full" else data[data.arrival_transport.notna()]
        fit = sample[sample.split == "training_fit"]
        calibration = sample[sample.split == "training_calibration"]
        test = sample[sample.split == "testing"]
        settings = fit_preprocessor(fit, transport)
        encoded_fit = encode_predictors(fit, settings)
        encoded_calibration = encode_predictors(calibration, settings)
        for split, group in sample.groupby("split"):
            splits.append({"cohort": cohort, "split": split, "visits": len(group),
                           "patients": group.subject_id.nunique(),
                           "events": int(group.abandonment.sum())})
        for name in MODEL_NAMES:
            print(f"Fitting {cohort}/{name} ...", flush=True)
            model = build_model(name).fit(encoded_fit, fit.abandonment)
            # Fixed reduction order avoids floating-point threshold/tie changes
            # when parallel tree predictions are summed in different orders.
            if name != "logistic":
                model.set_params(n_jobs=1)
            raw_calibration = model.predict_proba(encoded_calibration)[:, 1]
            calibrator = LogisticRegression(penalty=None, max_iter=2000).fit(
                log_odds(raw_calibration), calibration.abandonment)
            calibration_probability = calibrator.predict_proba(log_odds(raw_calibration))[:, 1]
            threshold = choose_threshold(calibration.abandonment, calibration_probability)
            bundle = {"model": model, "calibrator": calibrator, "preprocessing": settings,
                      "feature_names": list(encoded_fit.columns), "threshold": threshold,
                      "cohort": cohort, "model_name": name, "manifest": manifest,
                      "predictor_groups": predictor_groups(encoded_fit.columns)}
            joblib.dump(bundle, model_dir / f"{cohort}_{name}.joblib", compress=3)
            if cohort == "A_full" and name == "rf_unweighted":
                joblib.dump(bundle, PROCESSED_DATA_DIR / "random_forest_model.joblib", compress=3)
            raw = predict_bundle(bundle, test, calibrated=False)
            probability = predict_bundle(bundle, test)
            predictions = test[["subject_id", "stay_id", "abandonment"]].copy()
            predictions["raw_probability"] = raw
            predictions["probability"] = probability
            predictions["threshold"] = threshold
            predictions.to_csv(prediction_dir / f"{cohort}_{name}.csv.gz", index=False)
            tags = {"cohort": cohort, "model": name}
            evaluation_rows = []
            for kind, rule, p, cutoff in [
                ("raw", "0.5", raw, 0.5),
                ("calibrated", "0.5", probability, 0.5),
                ("calibrated", "calibration_f1", probability, threshold)]:
                evaluation_rows.append({**tags, "probability": kind, "threshold_rule": rule,
                    "threshold": cutoff, **evaluate_model(test.abandonment, p, cutoff),
                    "constant_training_prevalence_brier": brier_score_loss(
                        test.abandonment, np.repeat(fit.abandonment.mean(), len(test)))})
                tn, fp, fn, tp = confusion_matrix(test.abandonment, p >= cutoff).ravel()
                confusion.append({**tags, "probability": kind, "threshold_rule": rule,
                                  "tn": tn, "fp": fp, "fn": fn, "tp": tp})
            metrics.append(pd.DataFrame(evaluation_rows))
            save_result(metrics, "model_evaluation_metrics.csv")
            cal, bins = calibration_results(test.abandonment, raw, probability)
            calibrations.append(cal.assign(**tags))
            calibration_bins.append(bins.assign(**tags))
            print(f"  Test AP {average_precision_score(test.abandonment, probability):.4f}; "
                  "patient bootstrap and grouped importance ...", flush=True)
            ci, reps = bootstrap_evaluation(test, probability, threshold)
            intervals.append(ci.assign(**tags))
            bootstraps.append(reps.assign(**tags))
            importance = calculate_predictor_importance(bundle, test).assign(**tags)
            importances.append(importance)
            save_result(intervals, "evaluation_confidence_intervals.csv")
            save_result(bootstraps, "evaluation_bootstrap_replicates.csv")
            save_result(importances, "importance_replicates.csv")
            save_result(calibrations, "calibration_diagnostics.csv")
            save_result(calibration_bins, "calibration_bins.csv")
    importance = pd.concat(importances).groupby(["cohort", "model", "predictor"]).agg(
        ap_decrease_mean=("ap_decrease", "mean"), ap_decrease_sd=("ap_decrease", "std"),
        auroc_decrease_mean=("auroc_decrease", "mean"), auroc_decrease_sd=("auroc_decrease", "std")
    ).reset_index()
    importance["ap_rank"] = importance.groupby(["cohort", "model"]).ap_decrease_mean.rank(
        ascending=False, method="min")
    importance["auroc_rank"] = importance.groupby(["cohort", "model"]).auroc_decrease_mean.rank(
        ascending=False, method="min")
    importance.to_csv(OUTPUT_TABLES_DIR / "predictor_importance.csv", index=False)
    pd.DataFrame(splits).to_csv(OUTPUT_TABLES_DIR / "split_summary.csv", index=False)
    pd.DataFrame(confusion).to_csv(OUTPUT_TABLES_DIR / "confusion_matrix.csv", index=False)
    print(f"All models and evaluation saved in {(perf_counter() - start) / 60:.1f} minutes.",
          flush=True)


if __name__ == "__main__":
    main()
