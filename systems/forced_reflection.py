import time

from config import MAX_ITERATIONS
from systems.common import build_direct_answer_messages, format_context
from systems.llm import call_llm


SYSTEM_NAME = "forced_reflection"



def _critique_messages(question: str, answer: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a critical reviewer. Review this answer for factual errors, hallucinations, or unsupported claims.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}Question: {question}\n"
                f"Answer: {answer}\n"
                'List up to 3 specific problems briefly. If the answer is correct, say "No issues found."'
            ),
        },
    ]



def _refinement_messages(question: str, answer: str, critique: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Improve your answer based on critique.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}Question: {question}\n"
                f"Previous answer: {answer}\n"
                f"Critique: {critique}\n"
                "Provide a concise improved answer grounded in the available information:"
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

    critiques = []
    snapshots = [
        {
            "iteration": 1,
            "answer": initial_answer,
            "confidence": None,
            "response_time_seconds_cumulative": time.perf_counter() - start_time,
        }
    ]
    current_answer = initial_answer

    for round_number in range(1, MAX_ITERATIONS + 1):
        critique = call_llm(_critique_messages(question=question, answer=current_answer, context=context))
        critiques.append(critique)
        current_answer = call_llm(
            _refinement_messages(
                question=question,
                answer=current_answer,
                critique=critique,
                context=context,
            )
        )
        snapshots.append(
            {
                "iteration": round_number + 1,
                "answer": current_answer,
                "confidence": None,
                "response_time_seconds_cumulative": time.perf_counter() - start_time,
            }
        )

    elapsed = time.perf_counter() - start_time
    return {
        "answer": current_answer,
        "system": SYSTEM_NAME,
        "iterations": MAX_ITERATIONS,
        "confidence": None,
        "initial_answer": initial_answer,
        "critiques": critiques,
        "response_time_seconds": elapsed,
        "snapshots": snapshots,
    }
