"""Generate aggregate LaTeX tables and numeric macros from this run's CSV files."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
GENERATED = ROOT / "thesis" / "tum_thesis_v2" / "generated"


def read(name):
    return pd.read_csv(TABLES / name)


def count(value):
    return f"{int(value):,}"


def percent(value, digits=2):
    return f"{value:.{digits}f}" + r"\%"


def interval(row, digits=3):
    return (f"{row.odds_ratio:.{digits}f} "
            f"({row.confidence_interval_lower:.{digits}f}--{row.confidence_interval_upper:.{digits}f})")


def pvalue(value):
    return "$<0.001$" if value < .001 else f"{value:.3f}"


def save_table(filename, headers, rows, caption, label, alignment=None, note=""):
    alignment = alignment or "l" + "r" * (len(headers)-1)
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(str(v) for v in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\medskip\begin{minipage}{0.97\textwidth}\footnotesize " + note +
                  r"\end{minipage}"]
    lines += [r"\end{table}", ""]
    (GENERATED / filename).write_text("\n".join(lines))


def main():
    GENERATED.mkdir(parents=True, exist_ok=True)
    flow = read("cohort_flow.csv").set_index("cohort")
    rows = []
    for key, label in [("source", "Source ED visits"), ("A_full", "A: primary full cohort"),
                       ("B_C_known_transport", "B/C: known transport"),
                       ("unknown_transport", "Unknown transport (retained in A)")]:
        r = flow.loc[key]
        rows.append([label, count(r.visits), count(r.patients),
                     count(r.abandonment_visits), percent(r.abandonment_rate*100)])
    save_table("cohort_flow.tex", ["Population", "Visits", "Patients", "Events", "Rate"],
               rows, "Source population and analysis cohorts.", "tab:cohort-flow",
               note="All source visits have a documented disposition. B and C contain the same visits; "
                    "only C includes transport as a predictor. Rows are not mutually exclusive.")

    distributions = read("cohort_dispositions.csv")
    rows = []
    for outcome in ["LEFT WITHOUT BEING SEEN", "ELOPED", "LEFT AGAINST MEDICAL ADVICE"]:
        subset = distributions[distributions.disposition == outcome].set_index("cohort")
        all_n = subset.loc["A_full", "visits"]
        excluded = subset.loc["unknown_transport", "visits"]
        label = {"LEFT WITHOUT BEING SEEN": "LWBS", "ELOPED": "Eloped",
                 "LEFT AGAINST MEDICAL ADVICE": "Against medical advice"}[outcome]
        rows.append([label, count(all_n), count(subset.loc["B_C_known_transport", "visits"]),
                     count(excluded), percent(100*excluded/all_n)])
    save_table("outcome_components.tex", ["Component", "Full", "Known", "Unknown", r"\% in unknown"],
               rows, "Outcome components by transport documentation.", "tab:components",
               note="Unknown transport is excluded from B/C, not from A.")

    categories = read("descriptive_categories.csv")
    categories = categories[categories.cohort == "A_full"]
    continuous = read("descriptive_continuous.csv")
    continuous = continuous[continuous.cohort == "A_full"]
    sizes = read("descriptive_outcome_counts.csv")
    sizes = sizes[sizes.cohort == "A_full"].set_index("outcome")
    groups = ["All", "Non-abandonment", "Broad abandonment"]
    rows = [["Visits"] + [count(sizes.loc[g, "visits"]) for g in groups],
            ["Patients"] + [count(sizes.loc[g, "patients"]) for g in groups]]
    for predictor, label in [("previous_ed_visits", "Previous visits, median [IQR]"),
                             ("triage_pain", "Numeric pain, median [IQR]")]:
        values = continuous[continuous.predictor == predictor].set_index("outcome")
        rows.append([label] + [f"{values.loc[g, 'median']:g} "
                    f"[{values.loc[g, 'q1']:g}--{values.loc[g, 'q3']:g}]" for g in groups])
    levels = [("triage_acuity", str(float(i)), f"Acuity {i}") for i in range(1, 6)]
    levels += [("triage_acuity", "Missing/unknown", "Acuity missing")]
    levels += [("arrival_transport", level, label) for level, label in [
        ("WALK IN", "Walk-in"), ("AMBULANCE", "Ambulance"), ("OTHER", "Other transport"),
        ("HELICOPTER", "Helicopter"), ("Missing/unknown", "Transport unknown")]]
    levels += [("gender", "F", "Recorded gender F"), ("gender", "M", "Recorded gender M")]
    for predictor, level, label in levels:
        row = [label]
        for group in groups:
            values = categories[(categories.predictor == predictor) &
                                (categories.category == level) & (categories.outcome == group)]
            n = int(values.visits.sum())
            row.append(f"{n:,} ({100*n/sizes.loc[group, 'visits']:.1f}" + r"\%)")
        rows.append(row)
    pain = continuous[continuous.predictor == "triage_pain"].set_index("outcome")
    rows.append(["Pain missing after cleaning"] +
                [f"{int(pain.loc[g, 'missing']):,} "
                 f"({100*pain.loc[g, 'missing']/sizes.loc[g, 'visits']:.1f}" + r"\%)" for g in groups])
    save_table("sample_characteristics.tex",
               ["Characteristic", "All visits", "Other dispositions", "Abandonment"], rows,
               "Full-cohort characteristics by visit outcome.", "tab:characteristics",
               note="Percentages use column visit totals. A patient can contribute to both outcome columns, "
                    "so their patient counts are not additive. Pain summaries use valid numeric values only. "
                    "IQR: interquartile range. Acuity 1 is most urgent and 5 least urgent.")

    metrics = read("model_evaluation_metrics.csv")
    selected = metrics[metrics.threshold_rule == "calibration_f1"]
    cis = read("evaluation_confidence_intervals.csv")
    rows = []
    names = {"logistic": "Logistic", "rf_unweighted": "RF unweighted", "rf_balanced": "RF balanced"}
    for _, r in selected.iterrows():
        ci = cis[(cis.cohort == r.cohort) & (cis.model == r.model)].set_index("metric")
        def metric_ci(metric):
            return f"{r[metric]:.3f} ({ci.loc[metric, 'ci_lower']:.3f}--{ci.loc[metric, 'ci_upper']:.3f})"
        rows.append([r.cohort[0], names[r.model], metric_ci("auroc"),
                     metric_ci("average_precision"), f"{r.brier:.4f}"])
    save_table("model_comparison.tex",
               ["Cohort", "Model", "AUROC (95\\% CI)", "AP (95\\% CI)", "Brier"],
               rows, "Patient-held-out predictive comparison after sigmoid calibration.",
               "tab:models", alignment="llrrr",
               note="AP denotes average precision, not trapezoidal precision--recall area. "
                    "Intervals use 200 patient-cluster bootstrap replicates with the fitted model fixed. "
                    "A and B/C have different test prevalence; their AP values are not directly comparable "
                    "as measures of algorithm quality.")

    primary_rows = metrics[(metrics.cohort == "A_full") & (metrics.model == "rf_unweighted")]
    rows = []
    for _, r in primary_rows.iterrows():
        label = "Raw, 0.5" if r.probability == "raw" else (
            "Calibrated, 0.5" if r.threshold_rule == "0.5" else "Calibrated, training F1")
        rows.append([label, f"{r.threshold:.4f}", f"{r.precision:.3f}",
                     f"{r.recall:.3f}", f"{r.f1:.3f}"])
    save_table("primary_thresholds.tex", ["Rule", "Threshold", "Precision", "Recall", "F1"],
               rows, "Threshold-dependent performance of the primary forest.", "tab:thresholds",
               alignment="lrrrr",
               note="The F1 threshold is selected on the training calibration subset, never on the test set. "
                    "A zero precision/recall/F1 can mean that no positive prediction is made.")

    regression = read("logistic_regression_results.csv")
    primary_reg = regression[regression.analysis == "A_full"].set_index("term")
    labels = {"previous_ed_visits_log": r"$\log(1+\mathrm{previous\ visits})$",
              "triage_acuity_factor1": "Acuity 1 vs. 3", "triage_acuity_factor2": "Acuity 2 vs. 3",
              "triage_acuity_factor4": "Acuity 4 vs. 3", "triage_acuity_factor5": "Acuity 5 vs. 3",
              "triage_pain": "Pain (one-point increase)", "acuity_missing": "Acuity missing indicator",
              "pain_missing": "Pain missing indicator"}
    rows = [[label, interval(primary_reg.loc[term]), pvalue(primary_reg.loc[term, "p_value"])]
            for term, label in labels.items()]
    save_table("regression_main.tex", ["Term", "Adjusted OR (95\\% CI)", "$p$"],
               rows, "Full-cohort adjusted logistic regression.", "tab:regression", alignment="lrr",
               note="Patient-clustered standard errors; acuity 3 is the reference. Coefficients are "
                    "visit-level adjusted associations. Missingness indicators describe documentation "
                    "status and are not clinical severity measures.")

    history = regression[regression.term == "previous_ed_visits_log"]
    labels = {"A_full": "A: full cohort", "B_known": "B: known, no transport",
              "C_transport": "C: known, with transport", "A_complete_case": "A: complete cases",
              "lwbs": "LWBS vs. all other dispositions", "eloped": "Eloped vs. all other dispositions",
              "ama": "AMA vs. all other dispositions"}
    rows = []
    for _, r in history.iterrows():
        key = r.outcome if r.analysis == "A_component" else r.analysis
        rows.append([labels[key], interval(r)])
    save_table("history_sensitivity.tex", ["Analysis", "History OR (95\\% CI)"], rows,
               "Sensitivity of the adjusted history association.", "tab:sensitivity", alignment="lr",
               note="All rows estimate the OR for a one-unit increase in log(1 + previous visits). "
                    "Component outcomes use one-versus-rest comparators, not only non-abandonment visits.")

    scenarios = read("scenario_probabilities.csv")
    subset = scenarios[(scenarios.triage_acuity == 3) &
                       scenarios.previous_ed_visits.isin([0, 10])]
    rows = []
    for _, r in subset.iterrows():
        label = "A: all transport" if r.cohort == "A_full" else "C: " + r.arrival_transport.lower()
        rows.append([label, str(r.previous_ed_visits), count(r.support_visits),
                     count(r.support_events),
                     f"{r.adjusted_probability_percent:.2f} "
                     f"({r.confidence_interval_lower_percent:.2f}--{r.confidence_interval_upper_percent:.2f})"])
    save_table("scenario_selected.tex", ["Cohort/profile", "History", "Visits", "Events", "Probability, \\% (95\\% CI)"],
               rows, "Selected supported acuity-3 scenarios.", "tab:scenarios", alignment="lrrrr",
               note="Visits/events count the exact observed history/acuity/transport profile (transport "
                    "unrestricted in A). Probabilities are standardized over each cohort's pain distribution. "
                    "Intervals use clustered covariance and a logit-scale delta method, conditional on "
                    "that distribution and model. They do not measure intervention effects.")

    r = selected[(selected.cohort == "A_full") & (selected.model == "rf_unweighted")].iloc[0]
    ci = cis[(cis.cohort == "A_full") & (cis.model == "rf_unweighted")].set_index("metric")
    hist = primary_reg.loc["previous_ed_visits_log"]
    macros = {"FullVisits": count(flow.loc["A_full", "visits"]),
              "FullPatients": count(flow.loc["A_full", "patients"]),
              "FullEvents": count(flow.loc["A_full", "abandonment_visits"]),
              "FullRate": percent(flow.loc["A_full", "abandonment_rate"]*100),
              "PrimaryAUROC": f"{r.auroc:.3f}", "PrimaryAP": f"{r.average_precision:.3f}",
              "PrimaryBrier": f"{r.brier:.4f}", "PrimaryPrecision": f"{r.precision:.3f}",
              "PrimaryRecall": f"{r.recall:.3f}", "PrimaryFone": f"{r.f1:.3f}",
              "PrimaryThreshold": f"{r.threshold:.4f}",
              "PrimaryPrevalence": f"{r.observed_rate:.4f}",
              "PrimaryAUCInterval": f"{ci.loc['auroc', 'ci_lower']:.3f}--{ci.loc['auroc', 'ci_upper']:.3f}",
              "PrimaryAPInterval": f"{ci.loc['average_precision', 'ci_lower']:.3f}--{ci.loc['average_precision', 'ci_upper']:.3f}",
              "HistoryOR": f"{hist.odds_ratio:.3f}",
              "HistoryORInterval": f"{hist.confidence_interval_lower:.3f}--{hist.confidence_interval_upper:.3f}",
              "DoublingOR": f"{np.exp(hist.coefficient*np.log(2)):.3f}",
              "DoublingORInterval": f"{np.exp((hist.coefficient-1.96*hist.robust_standard_error)*np.log(2)):.3f}"
                                    f"--{np.exp((hist.coefficient+1.96*hist.robust_standard_error)*np.log(2)):.3f}"}
    (GENERATED / "results_macros.tex").write_text(
        "% Generated from aggregate outputs; rerun src/thesis_tables.py.\n" +
        "\n".join("\\newcommand{\\" + name + "}{" + value + "}"
                  for name, value in macros.items()) + "\n")
    print("Generated thesis tables and numeric macros.")


if __name__ == "__main__":
    main()
