import time

from systems.common import build_direct_answer_messages, format_context
from systems.llm import call_llm


SYSTEM_NAME = "cross_agent"



def _peer_review_messages(question: str, answer: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a skeptical peer reviewer. Challenge this answer aggressively.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}Question: {question}\n"
                f"Answer from Agent A: {answer}\n"
                "Find up to 3 mistakes, unsupported claims, or hallucinations. Be direct and concise."
            ),
        },
    ]



def _agent_a_refinement_messages(question: str, original_answer: str, critique: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are Agent A. Your peer Agent B has reviewed your answer.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}Question: {question}\n"
                f"Your original answer: {original_answer}\n"
                f"Agent B critique: {critique}\n"
                "Produce your final concise improved answer:"
            ),
        },
    ]



def _verification_messages(question: str, answer: str, context: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a skeptical peer reviewer. Verify whether the final answer is grounded and note any remaining concerns.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}Question: {question}\n"
                f"Final answer from Agent A: {answer}\n"
                "Provide a one-sentence verification note."
            ),
        },
    ]



def run(question: str, ground_truth: str, context: str, is_unanswerable: bool) -> dict:
    del ground_truth, is_unanswerable

    start_time = time.perf_counter()
    agent_a_initial = call_llm(
        build_direct_answer_messages(
            question=question,
            context=context,
            system_prompt="You are Agent A. Give a careful, grounded answer.",
        ),
        temperature=0.3,
    )
    snapshots = [
        {
            "iteration": 1,
            "answer": agent_a_initial,
            "confidence": None,
            "response_time_seconds_cumulative": time.perf_counter() - start_time,
        }
    ]

    agent_b_critique = call_llm(
        _peer_review_messages(question=question, answer=agent_a_initial, context=context),
        temperature=0.7,
    )
    final_answer = call_llm(
        _agent_a_refinement_messages(
            question=question,
            original_answer=agent_a_initial,
            critique=agent_b_critique,
            context=context,
        ),
        temperature=0.3,
    )
    snapshots.append(
        {
            "iteration": 2,
            "answer": final_answer,
            "confidence": None,
            "response_time_seconds_cumulative": time.perf_counter() - start_time,
        }
    )
    agent_b_verification = call_llm(
        _verification_messages(question=question, answer=final_answer, context=context),
        temperature=0.7,
    )

    elapsed = time.perf_counter() - start_time
    return {
        "answer": final_answer,
        "system": SYSTEM_NAME,
        "iterations": 2,
        "confidence": None,
        "initial_answer": agent_a_initial,
        "agent_a_initial": agent_a_initial,
        "agent_b_critique": agent_b_critique,
        "agent_b_verification": agent_b_verification,
        "response_time_seconds": elapsed,
        "snapshots": snapshots,
    }
