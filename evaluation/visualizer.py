from pathlib import Path

import pandas as pd



def _load_plotting_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except Exception as exc:
        raise RuntimeError(
            "Matplotlib and seaborn are required to generate benchmark visualizations. Install requirements.txt before running the benchmark."
        ) from exc



def generate_leaderboard(summary: dict, output_path: str | Path) -> None:
    plt, _ = _load_plotting_libs()

    systems = list(summary.keys())
    values = [summary[system].get("hallucination_rate") for system in systems]
    dataframe = pd.DataFrame({"system": systems, "hallucination_rate": values}).dropna()
    dataframe = dataframe.sort_values("hallucination_rate", ascending=False)

    if dataframe.empty:
        raise RuntimeError("No benchmark summary data is available for leaderboard generation.")

    cmap = plt.get_cmap("RdYlGn_r")
    normalized = dataframe["hallucination_rate"] - dataframe["hallucination_rate"].min()
    denominator = dataframe["hallucination_rate"].max() - dataframe["hallucination_rate"].min()
    if denominator == 0:
        denominator = 1.0
    normalized = normalized / denominator
    colors = [cmap(value) for value in normalized]

    plt.figure(figsize=(10, 5))
    plt.barh(dataframe["system"], dataframe["hallucination_rate"], color=colors)
    plt.xlabel("Hallucination Rate (lower is better)")
    plt.ylabel("System")
    plt.title("ReflectScore Leaderboard")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()



def generate_iteration_curve(iteration_results: pd.DataFrame, output_path: str | Path) -> None:
    plt, _ = _load_plotting_libs()
    if iteration_results.empty:
        raise RuntimeError("No iteration snapshot data is available for iteration curve generation.")

    dataframe = iteration_results.copy()
    forced_only = dataframe[dataframe["system"] == "forced_reflection"].copy()
    if not forced_only.empty:
        dataframe = forced_only

    dataframe["hallucination_rate"] = 1.0 - dataframe["is_correct"].astype(float)
    curve = (
        dataframe.groupby(["category", "iteration"], as_index=False)["hallucination_rate"]
        .mean()
        .sort_values(["category", "iteration"])
    )

    plt.figure(figsize=(10, 5))
    for category, group in curve.groupby("category"):
        plt.plot(group["iteration"], group["hallucination_rate"], marker="o", label=category)
    plt.xlabel("Iteration")
    plt.ylabel("Hallucination Rate")
    plt.title("Forced Reflection Hallucination Rate by Iteration and Category")
    plt.xticks(sorted(curve["iteration"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()



def generate_failure_heatmap(final_results: pd.DataFrame, output_path: str | Path) -> None:
    plt, sns = _load_plotting_libs()
    if final_results.empty:
        raise RuntimeError("No final benchmark results are available for failure heatmap generation.")

    dataframe = final_results.copy()
    dataframe = dataframe[dataframe["initial_answer"].notna()]
    if dataframe.empty:
        raise RuntimeError("No reflective benchmark rows are available for failure heatmap generation.")

    dataframe["backfired"] = dataframe["backfired"].fillna(False).astype(float)
    heatmap = dataframe.pivot_table(
        index="system",
        columns="category",
        values="backfired",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4))
    sns.heatmap(heatmap, annot=True, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, fmt=".2f")
    plt.title("Backfire Rate by System and Category")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()



def generate_confidence_plot(final_results: pd.DataFrame, output_path: str | Path) -> None:
    plt, _ = _load_plotting_libs()

    dataframe = final_results.copy()
    dataframe = dataframe[dataframe["system"] == "confidence_reflection"]
    dataframe = dataframe[dataframe["confidence"].notna()]
    if dataframe.empty:
        raise RuntimeError("No confidence_reflection rows with confidence scores are available for confidence plot generation.")

    plt.figure(figsize=(6, 6))
    plt.scatter(
        dataframe["confidence"].astype(float),
        dataframe["is_correct"].astype(float),
        alpha=0.7,
        color="#1f77b4",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Actual accuracy")
    plt.title("Confidence vs Accuracy")
    plt.xlim(0, 1)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
