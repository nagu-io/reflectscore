import unittest

from evaluation import metrics


class MetricsTests(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic(self):
        rows = [{"is_correct": True}, {"is_correct": False}, {"is_correct": True}]
        first = metrics.bootstrap_ci(rows, metrics.hallucination_rate)
        second = metrics.bootstrap_ci(rows, metrics.hallucination_rate)
        self.assertEqual(first, second)

    def test_confidence_calibration_groups_predictions(self):
        rows = [
            {"system": "confidence_reflection", "confidence": 0.81, "is_correct": True},
            {"system": "confidence_reflection", "confidence": 0.79, "is_correct": False},
        ]
        calibration = metrics.confidence_calibration(rows)
        self.assertIn("0.8", calibration)
        self.assertAlmostEqual(calibration["0.8"], 0.5)

    def test_reflection_metrics_ignore_nan_initial_answers(self):
        rows = [
            {"initial_answer": float("nan"), "initial_is_correct": True, "is_correct": False},
            {"initial_answer": None, "initial_is_correct": False, "is_correct": True},
            {"initial_answer": "", "initial_is_correct": False, "is_correct": True},
        ]
        self.assertIsNone(metrics.correction_rate(rows))
        self.assertIsNone(metrics.backfire_rate(rows))


if __name__ == "__main__":
    unittest.main()
