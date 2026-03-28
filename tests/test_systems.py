import unittest
from unittest.mock import patch

from config import MAX_ITERATIONS
from systems import confidence_reflection, forced_reflection
from systems.llm import call_llm_json
from systems.verifier_reflection import verify


class SystemsTests(unittest.TestCase):
    @patch("systems.forced_reflection.call_llm")
    def test_forced_reflection_saves_initial_and_refinements(self, mock_call_llm):
        side_effect = ["initial answer"]
        for index in range(1, MAX_ITERATIONS + 1):
            side_effect.extend([f"critique {index}", f"answer {index}"])
        mock_call_llm.side_effect = side_effect
        result = forced_reflection.run("Q", "A", "", False)
        self.assertEqual(result["initial_answer"], "initial answer")
        self.assertEqual(len(result["snapshots"]), MAX_ITERATIONS + 1)
        self.assertEqual(result["snapshots"][-1]["answer"], f"answer {MAX_ITERATIONS}")

    @patch("systems.confidence_reflection.call_llm_json")
    def test_confidence_reflection_only_runs_until_threshold(self, mock_call_llm_json):
        mock_call_llm_json.side_effect = [
            {"answer": "first", "confidence": 0.2, "reason": "low"},
            {"answer": "second", "confidence": 0.9, "reason": "good"},
        ]
        result = confidence_reflection.run("Q", "A", "", False)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(len(result["snapshots"]), 2)
        self.assertEqual(result["answer"], "second")

    def test_verifier_accepts_real_function_name_answer(self):
        context = "# file: auth.py\ndef authenticate_user(username, password):\n    return None\n"
        passed, checks = verify(
            question="Which function handles user authentication in the codebase?",
            answer="authenticate_user",
            context=context,
            is_unanswerable=False,
        )
        self.assertTrue(passed)
        self.assertTrue(all(check["passed"] for check in checks))

    def test_verifier_flags_hallucinated_entities_from_context_grounded_answers(self):
        context = "# file: auth.py\ndef authenticate_user(username, password):\n    return None\n"
        passed, checks = verify(
            question="Which function handles user authentication in the codebase?",
            answer="authenticate_user in OAuthGateway",
            context=context,
            is_unanswerable=False,
        )
        self.assertFalse(passed)
        self.assertTrue(any(check["rule"] == "entity_hallucination_check" and not check["passed"] for check in checks))

    def test_verifier_flags_contradictory_answers(self):
        context = "# file: auth.py\ndef authenticate_user(username, password):\n    return None\n"
        passed, checks = verify(
            question="Which function handles user authentication in the codebase?",
            answer="authenticate_user",
            context=context,
            is_unanswerable=False,
            contradiction_detected=True,
            contradiction_response="YES",
        )
        self.assertFalse(passed)
        self.assertTrue(any(check["rule"] == "contradiction_check" and not check["passed"] for check in checks))

    @patch("systems.llm.call_llm", return_value="not-json")
    def test_call_llm_json_falls_back_when_json_is_invalid(self, mock_call_llm):
        result = call_llm_json([{"role": "user", "content": "hi"}])
        self.assertEqual(result["answer"], "not-json")
        self.assertEqual(result["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
