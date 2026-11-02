"""Descriptive return timing and publication tables for the separate v3 thesis."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parents[1]
PROJECT = REVISION.parents[1]
TABLES = REVISION / "tables"
GENERATED = REVISION / "generated"
FIGURES = REVISION / "figures"


def previous_visit_timing(arrivals):
    """Count only strictly earlier arrivals, including all records at earlier ties."""
    visits = arrivals.sort_values(["subject_id", "intime", "stay_id"]).copy()
    distinct = visits[["subject_id", "intime"]].drop_duplicates().copy()
    distinct["previous_intime"] = distinct.groupby("subject_id").intime.shift()
    visits = visits.merge(distinct, on=["subject_id", "intime"], validate="many_to_one")
    visits["days_since_previous"] = (
        visits.intime - visits.previous_intime).dt.total_seconds() / 86400
    first = visits.groupby("subject_id").intime.transform("min")
    visits["days_since_first_observed"] = (visits.intime - first).dt.total_seconds() / 86400
    times = visits.intime.to_numpy(dtype="datetime64[ns]")
    counts = {days: np.zeros(len(visits), dtype=int) for days in (30, 90, 365)}
    for positions in visits.groupby("subject_id", sort=False).indices.values():
        patient_times = times[positions]
        # Both boundaries are explicit: [current - window, current).
        right = np.searchsorted(patient_times, patient_times, side="left")
        for days, values in counts.items():
            left = np.searchsorted(patient_times, patient_times - np.timedelta64(days, "D"),
                                   side="left")
            values[positions] = right - left
    for days, values in counts.items():
        visits[f"previous_{days}d_visits"] = values
    return visits


def check_timing_logic():
    """Synthetic checks cover ties, first visits and exact window boundaries."""
    sample = pd.DataFrame({"subject_id": [1, 1, 1, 1, 2], "stay_id": range(5),
        "intime": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-31",
                                  "2020-02-01", "2020-01-31"])})
    result = previous_visit_timing(sample).set_index("stay_id")
    assert result.previous_30d_visits.tolist() == [0, 0, 2, 1, 0]
    assert result.loc[2, "days_since_previous"] == 30
    assert result.loc[3, "days_since_previous"] == 1
    assert result.loc[[0, 1, 4], "days_since_previous"].isna().all()


def write_table(filename, caption, label, headers, rows, alignment):
    """Export only aggregate values; no patient identifiers enter the thesis."""
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{" + caption + " (source: author's calculations).}",
             r"\label{" + label + "}", r"\begin{tabular}{" + alignment + "}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GENERATED / filename).write_text("\n".join(lines) + "\n")


def summarize_timing():
    arrivals = pd.read_csv(PROJECT / "data/raw/mimic-iv-ed-2.2/ed/edstays.csv.gz",
                          usecols=["subject_id", "stay_id", "intime"], parse_dates=["intime"])
    data = pd.read_csv(PROJECT / "data/processed/abandonment_modeling_data.csv.gz")
    timing = previous_visit_timing(arrivals)
    data = data.merge(timing, on=["subject_id", "stay_id"], validate="one_to_one")
    assert len(data) == 425087
    assert data.days_since_previous.isna().equals(data.previous_ed_visits.eq(0))
    assert (data.previous_30d_visits <= data.previous_90d_visits).all()
    assert (data.previous_90d_visits <= data.previous_365d_visits).all()
    assert (data.previous_365d_visits <= data.previous_ed_visits).all()
    groups = ["No earlier observed", "0-7 days", "8-30 days", "31-90 days", "91-365 days", ">365 days"]
    # Labels abbreviate continuous intervals: (0,7], (7,30], etc.
    data["recency_group"] = pd.cut(data.days_since_previous, [0, 7, 30, 90, 365, np.inf],
        labels=groups[1:]).cat.add_categories(groups[0]).fillna(groups[0])
    data["history_group"] = pd.cut(data.previous_ed_visits, [-1, 0, 1, 3, 5, 10, np.inf],
        labels=["0", "1", "2-3", "4-5", "6-10", ">10"])
    summary = data.groupby("recency_group", observed=True).agg(
        visits=("stay_id", "size"), patients=("subject_id", "nunique"),
        events=("abandonment", "sum"), median_history=("previous_ed_visits", "median"),
        lwbs=("lwbs", "sum"), eloped=("eloped", "sum"), ama=("ama", "sum")).reindex(groups)
    summary["rate_percent"] = 100 * summary.events / summary.visits
    summary.to_csv(TABLES / "return_interval_summary.csv")
    cross = data.groupby(["history_group", "recency_group"], observed=True).agg(
        visits=("stay_id", "size"), events=("abandonment", "sum"))
    cross["rate_percent"] = 100 * cross.events / cross.visits
    cross.to_csv(TABLES / "history_recency_cross_tab.csv")
    rows = []
    for name, group in [("All returning visits", data[data.previous_ed_visits > 0]),
                        ("History above ten", data[data.previous_ed_visits > 10])]:
        rows.append({"population": name, "visits": len(group),
            "patients": group.subject_id.nunique(),
            "gap_q25_days": group.days_since_previous.quantile(.25),
            "gap_median_days": group.days_since_previous.median(),
            "gap_q75_days": group.days_since_previous.quantile(.75),
            "span_median_days": group.days_since_first_observed.median(),
            "prior_30d_median": group.previous_30d_visits.median(),
            "prior_90d_median": group.previous_90d_visits.median(),
            "prior_365d_median": group.previous_365d_visits.median(),
            "return_within_30d_percent": 100 * group.days_since_previous.le(30).mean()})
    pd.DataFrame(rows).to_csv(TABLES / "return_interval_quantiles.csv", index=False)
    write_table("return_intervals.tex", "Broad abandonment by time since the previous arrival",
        "tab:return-intervals", ["Interval", "Visits", "Events", "Rate (\\%)", "Median history"],
        [(name, f"{row.visits:,.0f}", f"{row.events:,.0f}", f"{row.rate_percent:.2f}",
          f"{row.median_history:.0f}") for name, row in summary.iterrows()], "lrrrr")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["None", "(0,7]", "(7,30]", "(30,90]", "(90,365]", ">365"]
    axes[0].bar(labels, summary.rate_percent, color="#0072B2")
    axes[1].bar(labels, summary.visits / 1000, color="#009E73")
    for ax in axes:
        ax.set_xlabel("Days since previous arrival")
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Broad abandonment (%)")
    axes[1].set_ylabel("ED visits (thousands)")
    fig.tight_layout()
    fig.savefig(FIGURES / "return_interval_patterns.png", dpi=300)
    plt.close(fig)
    print(summary.to_string())
    print(pd.DataFrame(rows).to_string(index=False))


def coefficient_table():
    all_results = pd.read_csv(TABLES / "logistic_regression_results.csv")
    data = all_results[all_results.analysis.eq("A_full")]
    labels = ["Intercept", r"$\log(1+\mathrm{previous\ visits})$", "Acuity 1 vs. 3",
              "Acuity 2 vs. 3", "Acuity 4 vs. 3", "Acuity 5 vs. 3", "Pain (one point)",
              "Acuity missing", "Pain missing"]
    rows = []
    for label, row in zip(labels, data.itertuples()):
        p = "$<0.001$" if row.p_value < .001 else f"{row.p_value:.3f}"
        interval = f"{row.confidence_interval_lower:.3f}--{row.confidence_interval_upper:.3f}"
        rows.append((label, f"{row.coefficient:.3f}", f"{row.robust_standard_error:.3f}",
                     f"{row.odds_ratio:.3f}", interval, p))
    write_table("coefficients.tex", "Full-cohort adjusted logistic regression coefficients",
                "tab:coefficients", ["Term", r"$\beta$", "Robust SE", "OR", "95\\% CI", "$p$"],
                rows, "lrrrrr")


def record_preservation(mode):
    paths = []
    for folder in ("thesis/tum_thesis_v2", "src", "outputs/tables", "outputs/figures"):
        paths.extend(p for p in (PROJECT / folder).rglob("*") if p.is_file())
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths)}
    manifest = REVISION / "original_files_sha256.json"
    if mode == "snapshot":
        assert not manifest.exists(), "Preservation snapshot already exists."
        manifest.write_text(json.dumps(hashes, indent=2) + "\n")
    else:
        assert hashes == json.loads(manifest.read_text()), "An original file changed."
        print(f"Preservation verified: {len(hashes)} original files unchanged.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        record_preservation(sys.argv[1])
    else:
        check_timing_logic()
        summarize_timing()
        coefficient_table()
