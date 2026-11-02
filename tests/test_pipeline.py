"""Small synthetic checks; no clinical records are printed or embedded."""
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_process
import processor


class ProcessingTests(unittest.TestCase):
    def test_strictly_previous_and_components(self):
        data = pd.DataFrame({
            "subject_id": [1, 1, 1, 1, 2], "stay_id": [10, 11, 12, 13, 14],
            "intime": ["2150-01-01", "2150-01-02", "2150-01-02", "2150-01-03", "2160-01-01"],
            "arrival_transport": ["UNKNOWN", "WALK IN", "AMBULANCE", "", "OTHER"],
            "disposition": ["HOME", "LEFT WITHOUT BEING SEEN", "ELOPED", "LEFT AGAINST MEDICAL ADVICE", None],
            "gender": ["F"] * 5})
        result = data_process.prepare_edstay_data(data).set_index("stay_id")
        self.assertEqual(result.previous_ed_visits.tolist(), [0, 1, 1, 3, 0])
        self.assertEqual(int(result.abandonment.sum()), 3)
        self.assertTrue(pd.isna(result.loc[14, "abandonment"]))
        self.assertTrue(pd.isna(result.loc[10, "arrival_transport"]))
        self.assertEqual(result.loc[11:13, ["lwbs", "eloped", "ama"]].sum().tolist(), [1, 1, 1])
        self.assertNotIn("intime", result)

    def test_pain_provenance(self):
        data = pd.DataFrame({"subject_id": range(5), "stay_id": range(5),
                             "acuity": [1, 2, 3, None, 9],
                             "pain": [None, "unable", "99", "0", "10"]})
        result = data_process.prepare_triage_features(data)
        self.assertEqual(result.pain_cleaning_status.tolist(),
                         ["original_missing", "non_numeric", "out_of_range", "valid_numeric", "valid_numeric"])
        self.assertEqual(result.triage_pain.isna().sum(), 3)
        self.assertEqual(result.triage_acuity.isna().sum(), 2)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.data = pd.DataFrame({"subject_id": np.repeat(np.arange(100), 2),
            "stay_id": np.arange(200), "previous_ed_visits": np.tile([0, 1], 100),
            "triage_acuity": rng.choice([1, 2, 3, 4, 5, np.nan], 200),
            "triage_pain": rng.choice([0, 3, 7, np.nan], 200),
            "arrival_transport": rng.choice(["WALK IN", "AMBULANCE"], 200),
            "abandonment": np.tile([0, 0, 0, 1], 50),
            "disposition": ["DO NOT USE AS PREDICTOR"] * 200})

    def test_patient_splits(self):
        old_dir = processor.PROCESSED_DATA_DIR
        with tempfile.TemporaryDirectory() as directory:
            processor.PROCESSED_DATA_DIR = Path(directory)
            try:
                result = processor.assign_patient_splits(self.data)
            finally:
                processor.PROCESSED_DATA_DIR = old_dir
        self.assertEqual(result.groupby("subject_id").split.nunique().max(), 1)
        counts = result.groupby("split").subject_id.nunique().to_dict()
        self.assertEqual(counts, {"testing": 30, "training_calibration": 14, "training_fit": 56})

    def test_preprocessor_and_bundle_roundtrip(self):
        settings = processor.fit_preprocessor(self.data, include_transport=True)
        x = processor.encode_predictors(self.data, settings)
        altered = self.data.copy()
        altered["triage_pain"] = 10
        processor.encode_predictors(altered, settings)
        self.assertEqual(settings["medians"]["triage_pain"], self.data.triage_pain.median())
        self.assertFalse(any(c in x for c in ["subject_id", "stay_id", "disposition", "abandonment"]))
        self.assertFalse(x.isna().any().any())
        self.assertIn("acuity_missing", processor.predictor_groups(x.columns)["Triage acuity"])
        self.assertIn("pain_missing", processor.predictor_groups(x.columns)["Triage pain"])
        model = processor.build_model("logistic").fit(x, self.data.abandonment)
        raw = model.predict_proba(x)[:, 1]
        calibrator = LogisticRegression().fit(processor.log_odds(raw), self.data.abandonment)
        bundle = {"model": model, "calibrator": calibrator, "preprocessing": settings,
                  "feature_names": list(x.columns)}
        expected = processor.predict_bundle(bundle, self.data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(bundle, path)
            actual = processor.predict_bundle(joblib.load(path), self.data)
        np.testing.assert_allclose(expected, actual)

    def test_evaluation_labels_and_threshold(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.1, .2, .7, .9])
        threshold = processor.choose_threshold(y, p)
        results = processor.evaluate_model(y, p, threshold)
        self.assertEqual(results["f1"], 1)
        self.assertEqual(results["average_precision"], 1)
        self.assertNotIn("auprc", results)


if __name__ == "__main__":
    unittest.main()
