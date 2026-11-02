"""Redraw two compact thesis figures from aggregate results, without training."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REVISION = Path(__file__).resolve().parents[1]
TABLES = REVISION / "tables"
FIGURES = REVISION / "figures"
COLORS = ["#0065BD", "#008566", "#D15C16", "#B51E45", "#756BB1"]


def plot_scenarios():
    """Keep the same supported profiles and numerical predictions as the old plot."""
    data = pd.read_csv(TABLES / "scenario_probabilities.csv")
    profiles = [("A_full", "ALL", "A: full cohort"),
                ("C_transport", "WALK IN", "C: walk-in"),
                ("C_transport", "AMBULANCE", "C: ambulance")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.05), sharey=True)
    maximum = data.loc[data.display_supported, "adjusted_probability_percent"].max() * 1.12
    for ax, (cohort, transport, title) in zip(axes, profiles):
        sample = data[(data.cohort == cohort) & (data.arrival_transport == transport)]
        for acuity, color in zip(range(1, 6), COLORS):
            rows = sample[sample.triage_acuity == acuity].sort_values("previous_ed_visits")
            # Missing plotted values break lines at unsupported profiles.
            probability = rows.adjusted_probability_percent.where(rows.display_supported)
            ax.plot(rows.previous_ed_visits, probability, "o-", color=color,
                    markersize=3, linewidth=1.2, label=f"Acuity {acuity}")
        ax.set(title=title, xlabel="Previous ED visits", xticks=[0, 1, 3, 5, 10],
               ylim=(0, maximum))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Adjusted abandonment (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", frameon=False,
               handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, .89), pad=.5)
    fig.savefig(FIGURES / "scenario_probabilities.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_processing_diagnostics():
    """Use saved aggregate counts; missingness is after cleaning, before imputation."""
    flow = pd.read_csv(TABLES / "cohort_flow.csv").set_index("cohort")
    missingness = pd.read_csv(TABLES / "missingness_patterns.csv")
    splits = pd.read_csv(TABLES / "split_summary.csv")
    full = flow.loc["A_full"]
    counts = [int(full.visits - full.abandonment_visits), int(full.abandonment_visits)]
    missing_counts = [missingness.loc[missingness.acuity_missing, "visits"].sum(),
                      missingness.loc[missingness.pain_missing, "visits"].sum(),
                      flow.loc["unknown_transport", "visits"]]
    rates = splits[splits.cohort == "A_full"].set_index("split").loc[
        ["training_fit", "training_calibration", "testing"]]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.15))
    axes[0].bar(["Other", "Abandonment"], counts, color=COLORS[:2])
    axes[0].set(title="Full-cohort\noutcomes", ylabel="Visits (log scale)",
                yscale="log", ylim=(1, 2_000_000))
    for index, count in enumerate(counts):
        axes[0].text(index, count, f"{count:,}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(["Acuity", "Pain", "Transport"],
                [100 * count / full.visits for count in missing_counts], color=COLORS[:3])
    axes[1].set(title="Missing/unknown\nbefore imputation", ylabel="Visits (%)", ylim=(0, 12))
    axes[2].bar(["Training\nfit", "Training\ncalibration", "Testing"],
                100 * rates.events / rates.visits, color=COLORS[:3])
    axes[2].set(title="Patient-separated\nsubsets", ylabel="Abandonment (%)", ylim=(0, 4.5))
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout(pad=.6)
    fig.savefig(FIGURES / "intermediate_pipeline_diagnostics.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10,
        "axes.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True})
    plot_scenarios()
    plot_processing_diagnostics()
    print("Saved two figures in the separate v3 thesis; original figures unchanged.")
