import json
from pathlib import Path

from config import REPORT_DIR, RESULTS_DIR, VISUALIZATIONS_DIR



def _format_metric(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_ci(value):
    if value is None:
        return "-"
    return str(value)



def generate_report(
    summary_path: str | Path = RESULTS_DIR / "summary.json",
    output_path: str | Path = REPORT_DIR / "benchmark_report.md",
    visualizations_dir: str | Path = VISUALIZATIONS_DIR,
) -> Path:
    summary_path = Path(summary_path)
    output_path = Path(output_path)
    visualizations_dir = Path(visualizations_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    lines = [
        "# ReflectScore Benchmark Report",
        "",
        "## Overview",
        "",
        "ReflectScore measures how different reflection strategies affect hallucination, grounding, refusal behavior, and latency.",
        "",
        "## Leaderboard",
        "",
        "| System | Hallucination Rate | Grounding | Refusal | Backfire | Mean Latency (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for system, metrics in summary.items():
        lines.append(
            "| {system} | {hall} | {ground} | {refusal} | {backfire} | {latency} |".format(
                system=system,
                hall=_format_metric(metrics.get("hallucination_rate")),
                ground=_format_metric(metrics.get("grounding_score")),
                refusal=_format_metric(metrics.get("refusal_accuracy")),
                backfire=_format_metric(metrics.get("backfire_rate")),
                latency=_format_metric(metrics.get("mean_response_time_seconds")),
            )
        )

    lines.extend(
        [
            "",
            "## Confidence Intervals",
            "",
        ]
    )
    for system, metrics in summary.items():
        lines.append(
            "- **{system}**: hallucination CI {hall_ci}, grounding CI {ground_ci}, refusal CI {refusal_ci}, correction CI {corr_ci}, backfire CI {back_ci}".format(
                system=system,
                hall_ci=_format_ci(metrics.get("hallucination_rate_ci")),
                ground_ci=_format_ci(metrics.get("grounding_score_ci")),
                refusal_ci=_format_ci(metrics.get("refusal_accuracy_ci")),
                corr_ci=_format_ci(metrics.get("correction_rate_ci")),
                back_ci=_format_ci(metrics.get("backfire_rate_ci")),
            )
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![Leaderboard]({(visualizations_dir / 'leaderboard.png').as_posix()})",
            f"![Iteration Curve]({(visualizations_dir / 'iteration_curve.png').as_posix()})",
            f"![Failure Heatmap]({(visualizations_dir / 'failure_heatmap.png').as_posix()})",
            f"![Confidence Plot]({(visualizations_dir / 'confidence_plot.png').as_posix()})",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    generate_report()
