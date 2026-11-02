"""Aggregate descriptions of the revised cohort; never export individual rows."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
OUTCOMES = ["abandonment", "lwbs", "eloped", "ama"]
HISTORY_LABELS = ["0", "1", "2-3", "4-5", "6-10", ">10"]


def summarize_categories(data, predictor):
    rows = []
    labels = data[predictor].astype("string").fillna("Missing/unknown")
    for label, group in data.groupby(labels, sort=False):
        row = {"predictor": predictor, "category": label, "visits": len(group),
               "patients": group.subject_id.nunique()}
        for outcome in OUTCOMES:
            row[f"{outcome}_events"] = int(group[outcome].sum())
            row[f"{outcome}_percent"] = group[outcome].mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    data = pd.read_csv(ROOT / "data" / "processed" / "abandonment_modeling_data.csv.gz")
    data["previous_visit_group"] = pd.cut(data.previous_ed_visits,
        [-1, 0, 1, 3, 5, 10, np.inf], labels=HISTORY_LABELS)
    categorical = []
    continuous = []
    outcome_counts = []
    for name, sample in [("A_full", data), ("B_known", data[data.arrival_transport.notna()])]:
        for predictor in ["previous_visit_group", "triage_acuity", "arrival_transport", "gender"]:
            categorical.append(summarize_categories(sample, predictor).assign(cohort=name))
        groups = {"All": sample, "Non-abandonment": sample[sample.abandonment == 0],
                  "Broad abandonment": sample[sample.abandonment == 1]}
        for outcome, group in groups.items():
            outcome_counts.append({"cohort": name, "outcome": outcome, "visits": len(group),
                                   "patients": group.subject_id.nunique()})
            for predictor in ["previous_ed_visits", "triage_pain"]:
                values = group[predictor]
                continuous.append({"cohort": name, "outcome": outcome, "predictor": predictor,
                    "observed": int(values.notna().sum()), "missing": int(values.isna().sum()),
                    "median": values.median(), "q1": values.quantile(.25),
                    "q3": values.quantile(.75), "maximum": values.max()})
        for outcome, group in groups.items():
            if outcome == "All":
                continue
            for predictor in ["triage_acuity", "arrival_transport", "gender"]:
                table = summarize_categories(group, predictor).assign(cohort=name, outcome=outcome)
                categorical.append(table)
    categories = pd.concat(categorical, ignore_index=True)
    categories["outcome"] = categories["outcome"].fillna("All")
    categories.to_csv(TABLES / "descriptive_categories.csv", index=False)
    pd.DataFrame(continuous).to_csv(TABLES / "descriptive_continuous.csv", index=False)
    pd.DataFrame(outcome_counts).to_csv(TABLES / "descriptive_outcome_counts.csv", index=False)
    categories[categories.predictor == "previous_visit_group"].to_csv(
        TABLES / "observed_previous_visit_rates.csv", index=False)
    missingness = data.groupby(
        [data.triage_acuity.isna().rename("acuity_missing"),
         data.triage_pain.isna().rename("pain_missing")]).agg(
        visits=("stay_id", "size"), events=("abandonment", "sum"),
        abandonment_rate=("abandonment", "mean")).reset_index()
    missingness.to_csv(TABLES / "missingness_patterns.csv", index=False)
    print("Revised EDA tables saved: cohorts, prior history, components, acuity, pain, transport, gender.")


if __name__ == "__main__":
    main()
