import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.scorer import BenchmarkScorer


class FakeEvaluator:
    def is_correct(self, answer, ground_truth, keywords, unanswerable=False):
        if unanswerable:
            return "not in context" in answer.lower() or "not present" in answer.lower()
        return answer == ground_truth or any(keyword in answer for keyword in keywords)


class ScorerTests(unittest.TestCase):
    def test_build_iteration_rows_preserves_snapshot_shape(self):
        scorer = BenchmarkScorer(evaluator=FakeEvaluator())
        question = {
            "id": "q1",
            "question": "Which function handles auth?",
            "answer": "authenticate_user",
            "category": "code",
            "keywords": ["authenticate_user"],
            "unanswerable": False,
        }
        result = {
            "system": "baseline",
            "answer": "authenticate_user",
            "snapshots": [
                {
                    "iteration": 1,
                    "answer": "authenticate_user",
                    "confidence": None,
                    "response_time_seconds_cumulative": 0.3,
                }
            ],
            "response_time_seconds": 0.3,
        }
        rows = scorer.build_iteration_rows(question, result)
        self.assertEqual(rows[0]["iteration"], 1)
        self.assertTrue(rows[0]["is_correct"])

    def test_scorer_can_target_custom_output_paths(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            scorer = BenchmarkScorer(
                evaluator=FakeEvaluator(),
                raw_path=base / "raw.csv",
                iteration_path=base / "iteration.csv",
                summary_path=base / "summary.json",
            )
            self.assertEqual(scorer.raw_path, base / "raw.csv")
            self.assertEqual(scorer.iteration_path, base / "iteration.csv")
            self.assertEqual(scorer.summary_path, base / "summary.json")

    def test_baseline_summary_never_reports_reflection_only_metrics(self):
        scorer = BenchmarkScorer(evaluator=FakeEvaluator())
        summary = scorer.build_summary(
            final_results=[
                {
                    "id": "q1",
                    "system": "baseline",
                    "category": "factual",
                    "unanswerable": False,
                    "answer": "right",
                    "ground_truth": "right",
                    "is_correct": True,
                    "initial_answer": "wrong",
                    "initial_is_correct": False,
                    "backfired": False,
                    "response_time_seconds": 0.2,
                }
            ],
            question_lookup={
                "q1": {
                    "id": "q1",
                    "question": "Q",
                    "answer": "right",
                    "category": "factual",
                    "keywords": ["right"],
                    "unanswerable": False,
                }
            },
        )
        self.assertIsNone(summary["baseline"]["correction_rate"])
        self.assertIsNone(summary["baseline"]["correction_rate_ci"])
        self.assertIsNone(summary["baseline"]["backfire_rate"])
        self.assertIsNone(summary["baseline"]["backfire_rate_ci"])


if __name__ == "__main__":
    unittest.main()
