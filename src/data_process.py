from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
ED_DATA_DIR = PROJECT_DIR / "data" / "raw" / "mimic-iv-ed-2.2" / "ed"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"

EDSTAYS_FILE = ED_DATA_DIR / "edstays.csv.gz"
TRIAGE_FILE = ED_DATA_DIR / "triage.csv.gz"
OUTPUT_FILE = PROCESSED_DATA_DIR / "abandonment_modeling_data.csv.gz"

ABANDONMENT_DISPOSITIONS = {
    "LEFT WITHOUT BEING SEEN",
    "ELOPED",
    "LEFT AGAINST MEDICAL ADVICE",
}
DOCUMENTED_DISPOSITIONS = ABANDONMENT_DISPOSITIONS | {
    "HOME", "ADMITTED", "TRANSFER", "OTHER", "EXPIRED"
}


def load_raw_tables():
    edstays = pd.read_csv(
        EDSTAYS_FILE,
        usecols=[
            "subject_id",
            "stay_id",
            "intime",
            "arrival_transport",
            "disposition",
            "gender",
        ],
    )
    triage = pd.read_csv(
        TRIAGE_FILE,
        usecols=["subject_id", "stay_id", "pain", "acuity"],
    )

    return edstays, triage


def prepare_edstay_data(edstays):
    edstay_data = edstays.copy()
    edstay_data["disposition"] = (
        edstay_data["disposition"].astype("string").str.strip().str.upper()
    )
 
    edstay_data["arrival_transport"] = (
        edstay_data["arrival_transport"].astype("string").str.strip().str.upper()
    )
    edstay_data["arrival_transport"] = edstay_data["arrival_transport"].replace(
        {"": pd.NA, "UNKNOWN": pd.NA}
    )

    edstay_data["intime"] = pd.to_datetime(edstay_data["intime"])
    edstay_data = edstay_data.sort_values(
        ["subject_id", "intime", "stay_id"]
    )
    # Simultaneous records must not count one another as previous visits.
    arrivals = edstay_data.groupby(["subject_id", "intime"]).size().rename("n")
    history = arrivals.groupby(level=0).cumsum() - arrivals
    edstay_data = edstay_data.merge(
        history.rename("previous_ed_visits").reset_index(),
        on=["subject_id", "intime"], how="left", validate="many_to_one"
    )

    edstay_data["abandonment"] = (
        edstay_data["disposition"].isin(ABANDONMENT_DISPOSITIONS).astype("Int8")
    )
    documented = edstay_data["disposition"].isin(DOCUMENTED_DISPOSITIONS)
    edstay_data.loc[~documented, "abandonment"] = pd.NA
    components = {"lwbs": "LEFT WITHOUT BEING SEEN", "eloped": "ELOPED",
                  "ama": "LEFT AGAINST MEDICAL ADVICE"}
    for name, label in components.items():
        edstay_data[name] = edstay_data["disposition"].eq(label).astype("Int8")
        edstay_data.loc[~documented, name] = pd.NA

    return edstay_data[
        [
            "subject_id",
            "stay_id",
            "previous_ed_visits",
            "arrival_transport",
            "abandonment",
            "disposition", "lwbs", "eloped", "ama", "gender",
        ]
    ]


def prepare_triage_features(triage):
    features = triage.copy()

    features["triage_acuity"] = pd.to_numeric(features["acuity"], errors="coerce")
    features["triage_acuity"] = features["triage_acuity"].where(
        features["triage_acuity"].between(1, 5)
    )

    features["triage_pain"] = pd.to_numeric(features["pain"], errors="coerce")
    features["pain_cleaning_status"] = "valid_numeric"
    features.loc[features["pain"].isna(), "pain_cleaning_status"] = "original_missing"
    features.loc[
        features["pain"].notna() & features["triage_pain"].isna(),
        "pain_cleaning_status"
    ] = "non_numeric"
    features.loc[
        features["triage_pain"].notna() & ~features["triage_pain"].between(0, 10),
        "pain_cleaning_status"
    ] = "out_of_range"
    features["triage_pain"] = features["triage_pain"].where(
        features["triage_pain"].between(0, 10)
    )

    return features[
        [
            "subject_id",
            "stay_id",
            "triage_acuity",
            "triage_pain",
            "pain_cleaning_status",
        ]
    ]


def build_modeling_dataset():
    edstays, triage = load_raw_tables()
    edstay_data = prepare_edstay_data(edstays)
    features = prepare_triage_features(triage)

    modeling_data = edstay_data.merge(
        features,
        on=["subject_id", "stay_id"],
        how="left",
        validate="one_to_one",
    )
    # Unknown transport is not an exclusion in the primary cohort.
    export_processing_audit(modeling_data)
    return modeling_data.dropna(subset=["abandonment"]).copy()


def export_processing_audit(data):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    documented = data["abandonment"].notna()
    samples = {
        "source": data,
        "A_full": data.loc[documented],
        "B_C_known_transport": data.loc[documented & data.arrival_transport.notna()],
        "unknown_transport": data.loc[documented & data.arrival_transport.isna()],
        "undocumented_outcome": data.loc[~documented],
    }
    flow = []
    dispositions = []
    for name, sample in samples.items():
        flow.append({"cohort": name, "visits": len(sample),
                     "patients": sample.subject_id.nunique(),
                     "abandonment_visits": int(sample.abandonment.sum()),
                     "abandonment_rate": sample.abandonment.mean()})
        counts = sample.disposition.fillna("MISSING").value_counts().rename_axis(
            "disposition").reset_index(name="visits")
        counts.insert(0, "cohort", name)
        dispositions.append(counts)
    pd.DataFrame(flow).to_csv(TABLE_DIR / "cohort_flow.csv", index=False)
    pd.concat(dispositions).to_csv(TABLE_DIR / "cohort_dispositions.csv", index=False)
    audit = data.groupby("pain_cleaning_status", dropna=False).agg(
        visits=("stay_id", "size"), abandonment_visits=("abandonment", "sum")
    )
    audit.to_csv(TABLE_DIR / "pain_cleaning_audit.csv")


def save_modeling_dataset(modeling_data):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    modeling_data.to_csv(OUTPUT_FILE, index=False)
    return OUTPUT_FILE


def print_dataset_summary(modeling_data, output_file):
    abandonment_visits = int(modeling_data["abandonment"].sum())
    abandonment_rate = modeling_data["abandonment"].mean() * 100
    returning_patient_visits = int((modeling_data["previous_ed_visits"] > 0).sum())

    print(f"Processed visits: {len(modeling_data):,}")
    print(f"Unique patients: {modeling_data['subject_id'].nunique():,}")
    print(f"Visits with previous ED history: {returning_patient_visits:,}")
    print(
        f"Broad abandonment visits: {abandonment_visits:,} "
        f"({abandonment_rate:.2f}%)"
    )
    print("\nMissing predictor values:")
    print(modeling_data[["triage_acuity", "arrival_transport", "triage_pain"]].isna().sum())
    print(f"\nSaved to: {output_file}")


def main():
    modeling_data = build_modeling_dataset()
    output_file = save_modeling_dataset(modeling_data)
    print_dataset_summary(modeling_data, output_file)


if __name__ == "__main__":
    main()
