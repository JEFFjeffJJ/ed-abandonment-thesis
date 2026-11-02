"""Draw publication figures from saved results, without retraining models."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, precision_recall_curve
from processor import predict_bundle

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA = PROJECT_DIR / "data" / "processed"
TABLES = PROJECT_DIR / "outputs" / "tables"
FIGURES = PROJECT_DIR / "outputs" / "figures"
COLORS = ["#0065BD", "#008566", "#D15C16", "#B51E45", "#756BB1"]
MODEL_LABELS = {"logistic": "Logistic regression", "rf_unweighted": "Unweighted forest",
                "rf_balanced": "Balanced forest"}
HISTORY = ["0", "1", "2-3", "4-5", "6-10", ">10"]


def save_figure(figure, filename):
    figure.tight_layout()
    figure.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {filename}")


def style():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "legend.fontsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True})


def plot_observed_abandonment():
    rates = pd.read_csv(TABLES / "observed_previous_visit_rates.csv")
    rates = rates[rates.cohort == "A_full"].set_index("category").loc[HISTORY]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(HISTORY, rates.abandonment_percent, color=COLORS[0])
    for bar, (_, row) in zip(bars, rates.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .12,
                f"{row.abandonment_percent:.2f}%\nn={int(row.visits):,}",
                ha="center", va="bottom", fontsize=9)
    ax.set(xlabel="Strictly earlier recorded ED visits", ylabel="Broad abandonment (%)",
           title="Full cohort: observed broad abandonment", ylim=(0, 10))
    ax.grid(axis="y", alpha=.2)
    save_figure(fig, "observed_abandonment_previous_visits.png")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(rates))
    for column, label, color in zip(["lwbs", "eloped", "ama"],
                                    ["LWBS", "Eloped", "Against medical advice"], COLORS):
        ax.bar(HISTORY, rates[f"{column}_percent"], bottom=bottom, label=label, color=color)
        bottom += rates[f"{column}_percent"].to_numpy()
    ax.set(xlabel="Strictly earlier recorded ED visits", ylabel="Component rate (%)",
           title="Components of the broad outcome")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    save_figure(fig, "abandonment_components.png")


def plot_model_evaluation():
    metrics = pd.read_csv(TABLES / "model_evaluation_metrics.csv")
    metrics = metrics[(metrics.cohort == "A_full") &
                      (metrics.threshold_rule == "calibration_f1")].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for (model, label), color in zip(MODEL_LABELS.items(), COLORS):
        data = pd.read_csv(DATA / "predictions" / f"A_full_{model}.csv.gz")
        fpr, tpr, _ = roc_curve(data.abandonment, data.probability)
        precision, recall, _ = precision_recall_curve(data.abandonment, data.probability)
        axes[0].plot(fpr, tpr, color=color, label=f"{label}: {metrics.loc[model, 'auroc']:.3f}")
        axes[1].plot(recall, precision, color=color,
                     label=f"{label}: {metrics.loc[model, 'average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set(title="ROC curves (legend: AUROC)", xlabel="False-positive rate",
                ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    prevalence = data.abandonment.mean()
    axes[1].axhline(prevalence, linestyle="--", color="grey", label=f"Prevalence: {prevalence:.3f}")
    axes[1].set(title="Precision-recall (legend: average precision)", xlabel="Recall",
                ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
    save_figure(fig, "model_evaluation_curves.png")


def plot_calibration():
    points = pd.read_csv(TABLES / "calibration_bins.csv")
    points = points[points.cohort == "A_full"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, kind in zip(axes, ["raw", "calibrated"]):
        subset = points[points.probability_type == kind]
        for (model, label), color in zip(MODEL_LABELS.items(), COLORS):
            values = subset[subset.model == model]
            ax.plot(values.mean_probability, values.observed_rate, "o-", color=color, label=label)
        limit = 1 if kind == "raw" else max(subset.mean_probability.max(),
                                            subset.observed_rate.max()) * 1.2
        ax.plot([0, limit], [0, limit], "--", color="grey")
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Mean predicted probability",
               ylabel="Observed abandonment proportion",
               title="Raw scores: full scale" if kind == "raw" else "Calibrated scores: zoomed scale")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    save_figure(fig, "calibration_comparison.png")


def plot_predictor_importance():
    data = pd.read_csv(TABLES / "predictor_importance.csv")
    primary = data[(data.cohort == "A_full") & (data.model == "rf_unweighted")].sort_values(
        "ap_decrease_mean")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(primary.predictor, primary.ap_decrease_mean, xerr=primary.ap_decrease_sd,
            capsize=4, color=COLORS[0])
    ax.axvline(0, color="grey", linewidth=1)
    ax.set(title="Full cohort: unweighted forest", xlabel="Mean decrease in test average precision")
    ax.grid(axis="x", alpha=.2)
    save_figure(fig, "grouped_predictor_importance.png")
    data["analysis"] = data.cohort + " / " + data.model
    order = [f"{cohort} / {model}" for cohort in ["A_full", "B_known", "C_transport"]
             for model in MODEL_LABELS]
    matrix = data.pivot(index="analysis", columns="predictor", values="ap_rank").reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5.7))
    plot = ax.imshow(matrix, cmap="YlGnBu_r", vmin=1, vmax=4, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix)), matrix.index)
    for row in range(len(matrix)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            ax.text(column, row, str(int(value)) if pd.notna(value) else "Not included",
                    ha="center", va="center", color="white" if value == 1 else "black", fontsize=9)
    ax.set_title("AP-importance ranks within each fitted model (1 = highest)")
    fig.colorbar(plot, ax=ax, ticks=[1, 2, 3, 4], pad=.03)
    save_figure(fig, "importance_rank_stability.png")


def plot_scenario_probabilities():
    data = pd.read_csv(TABLES / "scenario_probabilities.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), sharey=True)
    profiles = [("A_full", "ALL", "A: full cohort"),
                ("C_transport", "WALK IN", "C: walk-in"),
                ("C_transport", "AMBULANCE", "C: ambulance")]
    supported = data[data.display_supported]
    maximum = supported.adjusted_probability_percent.max() * 1.12
    for ax, (cohort, transport, title) in zip(axes, profiles):
        sample = data[(data.cohort == cohort) & (data.arrival_transport == transport)]
        for acuity, color in zip(range(1, 6), COLORS):
            rows = sample[sample.triage_acuity == acuity].sort_values("previous_ed_visits")
            # NaN breaks the line, so it never bridges an unsupported profile.
            y = rows.adjusted_probability_percent.where(rows.display_supported)
            ax.plot(rows.previous_ed_visits, y, "o-", color=color, label=f"Acuity {acuity}")
        ax.set(title=title, xlabel="Previous recorded ED visits",
               xticks=[0, 1, 3, 5, 10], ylim=(0, maximum))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Adjusted broad-abandonment probability (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(.52, 1.04),
               frameon=False)
    save_figure(fig, "scenario_probabilities.png")


def plot_diagnostics():
    data = pd.read_csv(DATA / "abandonment_modeling_data.csv.gz")
    split = pd.read_csv(TABLES / "split_summary.csv")
    split = split[split.cohort == "A_full"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    counts = data.abandonment.value_counts().sort_index()
    axes[0].bar(["Other outcome", "Abandonment"], counts, color=COLORS[:2])
    axes[0].set(ylabel="Visits (log scale)", yscale="log", title="Full-cohort outcomes")
    for i, count in enumerate(counts):
        axes[0].text(i, count, f"{count:,}", ha="center", va="bottom", fontsize=9)
    missing = data[["triage_acuity", "triage_pain", "arrival_transport"]].isna().mean()*100
    axes[1].bar(["Acuity", "Pain", "Transport"], missing, color=COLORS[:3])
    axes[1].set(title="Missing/unknown before processing", ylabel="Visits (%)", ylim=(0, 12))
    rates = split.set_index("split").loc[["training_fit", "training_calibration", "testing"]]
    axes[2].bar(["Training\nfit", "Training\ncalibration", "Testing"],
                rates.events/rates.visits*100, color=COLORS[:3])
    axes[2].set(title="Patient-separated subsets", ylabel="Abandonment (%)", ylim=(0, 4.5))
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, "intermediate_pipeline_diagnostics.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    previous = data.previous_ed_visits
    limit = previous.quantile(.99)
    axes[0].hist(previous[previous <= limit], bins=30, color=COLORS[0])
    axes[0].set(title="Raw counts through 99th percentile", xlabel="Previous recorded visits")
    axes[1].hist(np.log1p(previous), bins=30, color=COLORS[1])
    axes[1].set(title="All counts after log transformation", xlabel="log(1 + previous visits)")
    for ax in axes:
        ax.set_ylabel("Visits")
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, "intermediate_previous_visit_transformation.png")
    bundle = joblib.load(DATA / "models" / "A_full_rf_unweighted.joblib")
    training = pd.read_csv(DATA / "training_data.csv.gz")
    calibration = training[training.split == "training_calibration"]
    probability = predict_bundle(bundle, calibration)
    precision, recall, thresholds = precision_recall_curve(calibration.abandonment, probability)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-15)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for values, label, color in zip([precision[:-1], recall[:-1], f1],
                                    ["Precision", "Recall", "F1"], COLORS):
        ax.plot(thresholds, values, label=label, color=color)
    ax.axvline(bundle["threshold"], color="grey", linestyle="--", label="Selected within training")
    ax.set(title="Training calibration subset: threshold choice",
           xlabel="Calibrated probability threshold", ylabel="Metric value", xlim=(0, .30), ylim=(0, 1))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save_figure(fig, "intermediate_threshold_performance.png")


def main():
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_observed_abandonment()
    plot_model_evaluation()
    plot_calibration()
    plot_predictor_importance()
    plot_scenario_probabilities()
    plot_diagnostics()


if __name__ == "__main__":
    main()
