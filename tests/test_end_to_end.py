import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluation.auto_evaluator import AutoEvaluator
from evaluation.scorer import BenchmarkScorer
from run_benchmark import execute_benchmark, load_all_questions, load_smoke_questions


class FakeRetriever:
    def format_context(self, query: str) -> str:
        if "authentication" in query.lower():
            return "# file: auth.py\ndef authenticate_user(username, password):\n    return None\n"
        if "password" in query.lower() or "salary" in query.lower():
            return "# file: auth.py\ndef authenticate_user(username, password):\n    return None\n"
        return ""



def make_stub_runner(system_name: str, answer_builder):
    def runner(question: str, ground_truth: str, context: str, is_unanswerable: bool) -> dict:
        answer = answer_builder(question, ground_truth, context, is_unanswerable)
        return {
            "system": system_name,
            "answer": answer,
            "iterations": 1,
            "confidence": 0.8 if system_name == "confidence_reflection" else None,
            "initial_answer": ground_truth if system_name != "baseline" else None,
            "response_time_seconds": 0.01,
            "snapshots": [
                {
                    "iteration": 1,
                    "answer": answer,
                    "confidence": 0.8 if system_name == "confidence_reflection" else None,
                    "response_time_seconds_cumulative": 0.01,
                }
            ],
        }

    runner.__module__ = f"tests.stub_{system_name}"
    return runner



def touch_png(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"png")


class EndToEndSmokeTests(unittest.TestCase):
    def test_load_smoke_questions_uses_first_five_from_each_dataset(self):
        questions = load_smoke_questions()
        self.assertEqual(len(questions), 15)
        self.assertEqual([row["id"] for row in questions[:5]], ["f1", "f2", "f3", "f4", "f5"])
        self.assertEqual([row["id"] for row in questions[5:10]], ["c1", "c2", "c3", "c4", "c5"])
        self.assertEqual([row["id"] for row in questions[10:15]], ["u1", "u2", "u3", "u4", "u5"])

    @patch("run_benchmark.generate_confidence_plot", side_effect=lambda *args, **kwargs: touch_png(args[1]))
    @patch("run_benchmark.generate_failure_heatmap", side_effect=lambda *args, **kwargs: touch_png(args[1]))
    @patch("run_benchmark.generate_iteration_curve", side_effect=lambda *args, **kwargs: touch_png(args[1]))
    @patch("run_benchmark.generate_leaderboard", side_effect=lambda *args, **kwargs: touch_png(args[1]))
    def test_execute_benchmark_generates_outputs_with_stub_systems(self, *_mocks):
        questions = [row for row in load_all_questions() if row["id"] in {"f1", "c1", "u1"}]

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            results_dir = base / "results"
            visualizations_dir = base / "visualizations"
            report_path = base / "report" / "benchmark_report.md"
            scorer = BenchmarkScorer(
                evaluator=AutoEvaluator(),
                raw_path=results_dir / "raw_results.csv",
                iteration_path=results_dir / "iteration_results.csv",
                summary_path=results_dir / "summary.json",
            )

            system_runners = [
                make_stub_runner("baseline", lambda question, ground_truth, context, is_unanswerable: ground_truth if not is_unanswerable else "I don't know. The answer is not in context."),
                make_stub_runner("forced_reflection", lambda question, ground_truth, context, is_unanswerable: ground_truth if not is_unanswerable else "I don't know. The answer is not in context."),
                make_stub_runner("confidence_reflection", lambda question, ground_truth, context, is_unanswerable: ground_truth if not is_unanswerable else "I don't know. The answer is not in context."),
            ]

            summary = execute_benchmark(
                questions=questions,
                scorer=scorer,
                retriever=FakeRetriever(),
                system_runners=system_runners,
                visualizations_dir=visualizations_dir,
                report_path=report_path,
                validate_api=False,
                print_terminal=False,
            )

            self.assertIn("baseline", summary)
            self.assertTrue((results_dir / "raw_results.csv").exists())
            self.assertTrue((results_dir / "iteration_results.csv").exists())
            self.assertTrue((results_dir / "summary.json").exists())
            self.assertTrue((visualizations_dir / "leaderboard.png").exists())
            self.assertTrue((visualizations_dir / "iteration_curve.png").exists())
            self.assertTrue((report_path).exists())


if __name__ == "__main__":
    unittest.main()
