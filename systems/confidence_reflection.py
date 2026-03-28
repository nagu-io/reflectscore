import time
from typing import Any

from config import CONFIDENCE_THRESHOLD
from systems.common import format_context
from systems.llm import call_llm_json


SYSTEM_NAME = "confidence_reflection"
MAX_REFLECTION_ROUNDS = 3



def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return max(0.0, min(1.0, confidence))



def _answer_and_confidence_messages(question: str, context: str, previous_answer: str | None = None, previous_reason: str | None = None) -> list[dict]:
    previous_block = ""
    if previous_answer:
        previous_block = f"Previous answer: {previous_answer}\nPrevious assessment: {previous_reason or 'No prior assessment.'}\n"
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question briefly and rate your confidence.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}{previous_block}Question: {question}\n"
                "Respond in this exact JSON format:\n"
                '{\n  "answer": "your answer here",\n  "confidence": 0.85,\n  "reason": "why you are confident or not"\n}'
            ),
        },
    ]



def run(question: str, ground_truth: str, context: str, is_unanswerable: bool) -> dict:
    del ground_truth, is_unanswerable

    start_time = time.perf_counter()
    parsed = call_llm_json(_answer_and_confidence_messages(question=question, context=context))
    initial_answer = str(parsed.get("answer", "")).strip()
    current_answer = initial_answer
    current_reason = str(parsed.get("reason", "")).strip()
    current_confidence = _coerce_confidence(parsed.get("confidence", 0.5))
    confidence_scores = [current_confidence]
    triggered = current_confidence < CONFIDENCE_THRESHOLD

    snapshots = [
        {
            "iteration": 1,
            "answer": current_answer,
            "confidence": current_confidence,
            "response_time_seconds_cumulative": time.perf_counter() - start_time,
        }
    ]

    reflection_rounds = 0
    while current_confidence < CONFIDENCE_THRESHOLD and reflection_rounds < MAX_REFLECTION_ROUNDS:
        reflection_rounds += 1
        parsed = call_llm_json(
            _answer_and_confidence_messages(
                question=question,
                context=context,
                previous_answer=current_answer,
                previous_reason=current_reason,
            )
        )
        current_answer = str(parsed.get("answer", current_answer)).strip()
        current_reason = str(parsed.get("reason", current_reason)).strip()
        current_confidence = _coerce_confidence(parsed.get("confidence", current_confidence))
        confidence_scores.append(current_confidence)
        snapshots.append(
            {
                "iteration": reflection_rounds + 1,
                "answer": current_answer,
                "confidence": current_confidence,
                "response_time_seconds_cumulative": time.perf_counter() - start_time,
            }
        )

    elapsed = time.perf_counter() - start_time
    return {
        "answer": current_answer,
        "system": SYSTEM_NAME,
        "iterations": reflection_rounds,
        "confidence": current_confidence,
        "confidence_scores": confidence_scores,
        "triggered": triggered,
        "initial_answer": initial_answer,
        "response_time_seconds": elapsed,
        "snapshots": snapshots,
    }
