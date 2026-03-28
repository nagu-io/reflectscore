import time

from systems.common import build_direct_answer_messages
from systems.llm import call_llm


SYSTEM_NAME = "baseline"



def run(question: str, ground_truth: str, context: str, is_unanswerable: bool) -> dict:
    del ground_truth, is_unanswerable

    start_time = time.perf_counter()
    answer = call_llm(
        build_direct_answer_messages(
            question=question,
            context=context,
            system_prompt="You are a helpful assistant. Answer the question directly.",
        )
    )
    elapsed = time.perf_counter() - start_time

    return {
        "answer": answer,
        "system": SYSTEM_NAME,
        "iterations": 1,
        "confidence": None,
        "initial_answer": None,
        "response_time_seconds": elapsed,
        "snapshots": [
            {
                "iteration": 1,
                "answer": answer,
                "confidence": None,
                "response_time_seconds_cumulative": elapsed,
            }
        ],
    }
