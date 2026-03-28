import time
from typing import Any

from systems.common import build_direct_answer_messages, format_context
from systems.llm import call_llm, call_llm_json


SYSTEM_NAME = "self_reflection"
MAX_SELF_CHECK_ROUNDS = 3



def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)



def _self_check_messages(question: str, answer: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a fact-checker. Analyze if the answer contains hallucinations.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}Question: {question}\n"
                f"Answer: {answer}\n"
                'Does this answer contain hallucinations or incorrect facts?\nRespond with JSON: {"hallucinated": true/false, "explanation": "..."} and keep the explanation brief.'
            ),
        },
    ]



def _regeneration_messages(question: str, answer: str, explanation: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Revise your answer to remove hallucinations and unsupported claims.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}Question: {question}\n"
                f"Previous answer: {answer}\n"
                f"Fact-check feedback: {explanation}\n"
                "Provide a concise corrected answer grounded in the available information."
            ),
        },
    ]



def run(question: str, ground_truth: str, context: str, is_unanswerable: bool) -> dict:
    del ground_truth, is_unanswerable

    start_time = time.perf_counter()
    initial_answer = call_llm(
        build_direct_answer_messages(
            question=question,
            context=context,
            system_prompt="You are a helpful assistant. Answer the question directly.",
        )
    )
    current_answer = initial_answer
    self_detected_hallucination = False
    self_checks = 0
    snapshots = [
        {
            "iteration": 1,
            "answer": current_answer,
            "confidence": None,
            "response_time_seconds_cumulative": time.perf_counter() - start_time,
        }
    ]

    while self_checks < MAX_SELF_CHECK_ROUNDS:
        self_checks += 1
        check = call_llm_json(_self_check_messages(question=question, answer=current_answer, context=context))
        hallucinated = _coerce_bool(check.get("hallucinated", False))
        explanation = str(check.get("explanation", "No explanation provided.")).strip()
        if not hallucinated:
            break

        self_detected_hallucination = True
        current_answer = call_llm(
            _regeneration_messages(
                question=question,
                answer=current_answer,
                explanation=explanation,
                context=context,
            )
        )
        snapshots.append(
            {
                "iteration": len(snapshots) + 1,
                "answer": current_answer,
                "confidence": None,
                "response_time_seconds_cumulative": time.perf_counter() - start_time,
            }
        )

    elapsed = time.perf_counter() - start_time
    return {
        "answer": current_answer,
        "system": SYSTEM_NAME,
        "iterations": self_checks,
        "confidence": None,
        "initial_answer": initial_answer,
        "self_detected_hallucination": self_detected_hallucination,
        "response_time_seconds": elapsed,
        "snapshots": snapshots,
    }
