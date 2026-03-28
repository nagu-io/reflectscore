import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import DATA_DIR, REPORT_DIR, VISUALIZATIONS_DIR, ensure_directories, set_global_seed, validate_runtime_config
from evaluation.auto_evaluator import AutoEvaluator
from evaluation.scorer import BenchmarkScorer
from evaluation.visualizer import (
    generate_confidence_plot,
    generate_failure_heatmap,
    generate_iteration_curve,
    generate_leaderboard,
)
from report.generate_report import generate_report
from retrieval import CodeRetriever
from systems.baseline import run as run_baseline
from systems.confidence_reflection import run as run_confidence_reflection
from systems.cross_agent_reflection import run as run_cross_agent_reflection
from systems.forced_reflection import run as run_forced_reflection
from systems.self_reflection import run as run_self_reflection
from systems.verifier_reflection import run as run_verifier_reflection

SYSTEMS = [
    run_baseline,
    run_forced_reflection,
    run_confidence_reflection,
    run_self_reflection,
    run_cross_agent_reflection,
    run_verifier_reflection,
]
SYSTEM_NAME_TO_RUNNER = {
    "baseline": run_baseline,
    "forced_reflection": run_forced_reflection,
    "confidence_reflection": run_confidence_reflection,
    "self_reflection": run_self_reflection,
    "cross_agent": run_cross_agent_reflection,
    "verifier_reflection": run_verifier_reflection,
}



def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))



def load_all_questions() -> list[dict]:
    factual = load_dataset(DATA_DIR / "factual.json")
    code = load_dataset(DATA_DIR / "code.json")
    unanswerable = load_dataset(DATA_DIR / "unanswerable.json")
    return factual + code + unanswerable


def load_smoke_questions(limit_per_dataset: int = 5) -> list[dict]:
    factual = load_dataset(DATA_DIR / "factual.json")[:limit_per_dataset]
    code = load_dataset(DATA_DIR / "code.json")[:limit_per_dataset]
    unanswerable = load_dataset(DATA_DIR / "unanswerable.json")[:limit_per_dataset]
    return factual + code + unanswerable


def resolve_system_runners(selected_systems: str | None) -> list:
    if not selected_systems:
        return SYSTEMS

    runners = []
    for raw_name in selected_systems.split(","):
        system_name = raw_name.strip()
        if not system_name:
            continue
        if system_name not in SYSTEM_NAME_TO_RUNNER:
            available = ", ".join(sorted(SYSTEM_NAME_TO_RUNNER))
            raise ValueError(f"Unknown system '{system_name}'. Available systems: {available}")
        runners.append(SYSTEM_NAME_TO_RUNNER[system_name])
    if not runners:
        raise ValueError("No valid systems were provided.")
    return runners



def prepare_question_context(question_row: dict, retriever) -> str:
    if question_row["category"] in {"code", "unanswerable"}:
        return retriever.format_context(question_row["question"])
    return ""



def summarize_findings(final_df: pd.DataFrame, iteration_df: pd.DataFrame, summary: dict) -> tuple[str, str, str, str]:
    reflective = final_df[final_df["system"].isin([
        "forced_reflection",
        "confidence_reflection",
        "self_reflection",
        "cross_agent",
        "verifier_reflection",
    ])]
    factual = reflective[reflective["category"] == "factual"]
    factual_backfire = factual.groupby("system")["backfired"].mean().dropna()
    x_value = float(factual_backfire.mean()) if not factual_backfire.empty else 0.0

    confidence_rows = iteration_df[iteration_df["system"] == "confidence_reflection"]
    if confidence_rows.empty:
        y_value = 0.0
    else:
        calls_per_question = confidence_rows.groupby("id").size()
        y_value = float(((4 - calls_per_question.mean()) / 4) * 100)

    cross = summary.get("cross_agent", {}).get("hallucination_rate")
    self_reflection = summary.get("self_reflection", {}).get("hallucination_rate")
    z_value = (self_reflection - cross) if cross is not None and self_reflection is not None else 0.0

    latency_parts = []
    for system, metrics in summary.items():
        latency = metrics.get("mean_response_time_seconds")
        if latency is not None:
            latency_parts.append(f"{system}: {latency:.2f}s")
    w_value = "; ".join(latency_parts)

    return (
        f"Reflection backfired in {x_value:.2%} of reflective factual cases on average.",
        f"Confidence-triggered reflection saved an estimated {y_value:.2f}% of answer-generation calls versus an always-on 3-round loop.",
        f"Cross-agent reflection improved hallucination rate over self-reflection by {z_value:.2%}.",
        f"Average response_time_seconds per system -> {w_value}",
    )



def print_leaderboard(summary: dict) -> None:
    print("=" * 40)
    print("     REFLECTSCORE BENCHMARK RESULTS")
    print("=" * 40)
    print()
    header = f"{'System':25} {'Hall.Rate':>9} {'Ground.':>8} {'Refusal':>8} {'Backfire':>9} {'Latency':>8}"
    print(header)
    print("-" * len(header))
    ordered = sorted(
        summary.items(),
        key=lambda item: item[1].get("hallucination_rate") if item[1].get("hallucination_rate") is not None else 1.0,
    )
    for system, metrics in ordered:
        hall = metrics.get("hallucination_rate")
        ground = metrics.get("grounding_score")
        refusal = metrics.get("refusal_accuracy")
        backfire = metrics.get("backfire_rate")
        latency = metrics.get("mean_response_time_seconds")
        print(
            f"{system:25} "
            f"{f'{hall:.2f}' if hall is not None else '-':>9} "
            f"{f'{ground:.2f}' if ground is not None else '-':>8} "
            f"{f'{refusal:.2f}' if refusal is not None else '-':>8} "
            f"{f'{backfire:.2f}' if backfire is not None else '-':>9} "
            f"{f'{latency:.2f}' if latency is not None else '-':>8}"
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ReflectScore benchmark.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first 5 questions from each dataset for a fast smoke test.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=5,
        help="When using --smoke, limit each dataset to the first N questions.",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default=None,
        help="Comma-separated list of systems to run. Example: baseline,forced_reflection",
    )
    return parser.parse_args()



def execute_benchmark(
    questions: list[dict] | None = None,
    system_runners: list | None = None,
    scorer: BenchmarkScorer | None = None,
    retriever=None,
    visualizations_dir: Path = VISUALIZATIONS_DIR,
    report_path: Path = REPORT_DIR / "benchmark_report.md",
    validate_api: bool = True,
    print_terminal: bool = True,
) -> dict:
    ensure_directories()
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    set_global_seed()
    if validate_api:
        validate_runtime_config()

    evaluator = scorer.evaluator if scorer is not None else AutoEvaluator()
    scorer = scorer or BenchmarkScorer(evaluator=evaluator)
    retriever = retriever or CodeRetriever(DATA_DIR / "code_context.txt")
    questions = questions or load_all_questions()
    system_runners = system_runners or SYSTEMS
    question_lookup = {row["id"]: row for row in questions}
    completed = scorer.load_completed_pairs()

    for question_row in tqdm(questions, desc="Running benchmark", disable=not print_terminal):
        context = prepare_question_context(question_row, retriever)
        for system_runner in system_runners:
            module_name = system_runner.__module__.split(".")[-1]
            final_system_name = "cross_agent" if module_name == "cross_agent_reflection" else module_name
            if (question_row["id"], final_system_name) in completed:
                continue
            if print_terminal:
                print(f"Running {final_system_name} on {question_row['id']}...", flush=True)
            system_result = system_runner(
                question_row["question"],
                question_row["answer"],
                context,
                question_row["unanswerable"],
            )
            scorer.persist_result(question_row, system_result)
            completed.add((question_row["id"], system_result["system"]))

    final_df = scorer.load_raw_results()
    iteration_df = scorer.load_iteration_results()
    final_records = final_df.to_dict(orient="records")
    summary = scorer.build_summary(final_records, question_lookup)
    scorer.save_summary(summary)

    generate_leaderboard(summary, visualizations_dir / "leaderboard.png")
    generate_iteration_curve(iteration_df, visualizations_dir / "iteration_curve.png")
    generate_failure_heatmap(final_df, visualizations_dir / "failure_heatmap.png")
    generate_confidence_plot(final_df, visualizations_dir / "confidence_plot.png")
    generate_report(
        summary_path=scorer.summary_path,
        output_path=report_path,
        visualizations_dir=visualizations_dir,
    )

    if print_terminal:
        print_leaderboard(summary)
        print()
        findings = summarize_findings(final_df=final_df, iteration_df=iteration_df, summary=summary)
        print("=" * 40)
        for finding in findings:
            print(finding)
        print("=" * 40)

    return summary



def main() -> None:
    args = parse_args()
    questions = load_smoke_questions(limit_per_dataset=args.limit_per_dataset) if args.smoke else None
    execute_benchmark(
        questions=questions,
        system_runners=resolve_system_runners(args.systems),
        validate_api=True,
    )


if __name__ == "__main__":
    main()
