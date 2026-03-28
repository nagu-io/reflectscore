import re
import time

from systems.common import build_direct_answer_messages, format_context
from systems.llm import call_llm


SYSTEM_NAME = "verifier_reflection"
MAX_RETRIES = 2
REFUSAL_PHRASES = [
    "not present",
    "not in context",
    "i don't know",
    "cannot find",
    "not available",
    "no information",
    "not mentioned",
    "unable to find",
    "insufficient context",
    "cannot determine",
]

FUNCTION_PATTERN = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CLASS_PATTERN = re.compile(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|:)")
FILE_PATTERN = re.compile(r"# file: ([^\n]+)")
SYMBOL_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*\.py|[A-Z][A-Za-z0-9_]+|[a-z_][a-z0-9_]+)\b")
MULTIWORD_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
CAMEL_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
UPPER_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
GENERIC_ENTITY_WORDS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Context",
    "Question",
    "Answer",
    "Function",
    "Class",
    "File",
}



def extract_function_names(context: str) -> list[str]:
    return sorted(set(FUNCTION_PATTERN.findall(context)))



def extract_known_entities(context: str) -> set[str]:
    functions = FUNCTION_PATTERN.findall(context)
    classes = CLASS_PATTERN.findall(context)
    files = FILE_PATTERN.findall(context)
    constants = UPPER_ENTITY_PATTERN.findall(context)
    return set(functions + classes + files + constants)


def extract_named_entities(text: str) -> set[str]:
    entities = set()
    for token in SYMBOL_PATTERN.findall(text):
        if len(token) > 2 and ("_" in token or token.endswith(".py") or token[:1].isupper()):
            entities.add(token)
    entities.update(MULTIWORD_ENTITY_PATTERN.findall(text))
    entities.update(CAMEL_ENTITY_PATTERN.findall(text))
    entities.update(UPPER_ENTITY_PATTERN.findall(text))
    return {entity.strip() for entity in entities if entity.strip() and entity.strip() not in GENERIC_ENTITY_WORDS}


def normalize_entities(values: set[str]) -> set[str]:
    return {value.strip().lower() for value in values if value and value.strip()}


def detect_contradiction(question: str, answer: str, context: str) -> tuple[bool, str]:
    response = call_llm(
        [
            {
                "role": "system",
                "content": "You are a strict verifier. Determine whether the answer contradicts itself internally. Reply YES or NO only.",
            },
            {
                "role": "user",
                "content": (
                    f"{format_context(context)}Question: {question}\n"
                    f"Answer: {answer}\n"
                    "Does this answer contradict itself? Reply YES or NO only."
                ),
            },
        ],
        temperature=0.0,
        max_tokens=8,
    ).strip()
    return response.upper().startswith("YES"), response



def verify(
    question: str,
    answer: str,
    context: str,
    is_unanswerable: bool,
    contradiction_detected: bool = False,
    contradiction_response: str | None = None,
) -> tuple[bool, list[dict]]:
    checks = []
    answer_lower = answer.lower()
    known_entities = extract_known_entities(context)

    if context.strip():
        mentioned_symbols = {
            token
            for token in SYMBOL_PATTERN.findall(answer)
            if len(token) > 2 and ("_" in token or token.endswith(".py") or token[:1].isupper())
        }
        unknown_entities = sorted(symbol for symbol in mentioned_symbols if symbol not in known_entities)
        entity_passed = len(unknown_entities) == 0
        checks.append(
            {
                "rule": "entity_grounding_check",
                "passed": entity_passed,
                "details": unknown_entities,
            }
        )

        allowed_entities = normalize_entities(
            known_entities | extract_named_entities(question) | extract_named_entities(context)
        )
        answer_entities = extract_named_entities(answer)
        hallucinated_entities = sorted(
            entity for entity in answer_entities if entity.strip().lower() not in allowed_entities
        )
        checks.append(
            {
                "rule": "entity_hallucination_check",
                "passed": len(hallucinated_entities) == 0,
                "details": hallucinated_entities,
            }
        )

    if is_unanswerable:
        refusal_passed = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)
        checks.append({"rule": "refusal_check", "passed": refusal_passed})

    if "function" in question.lower() or "code" in question.lower():
        functions = extract_function_names(context)
        grounded = any(function_name in answer for function_name in functions)
        checks.append({"rule": "grounding_check", "passed": grounded})

    if context.strip():
        length_passed = len(answer.split()) >= 5 or any(entity in answer for entity in known_entities)
    else:
        length_passed = len(answer.strip()) >= 2
    checks.append({"rule": "length_check", "passed": length_passed})
    checks.append(
        {
            "rule": "contradiction_check",
            "passed": not contradiction_detected,
            "details": contradiction_response,
        }
    )

    all_passed = all(check["passed"] for check in checks)
    return all_passed, checks



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
                'List up to 3 specific problems briefly. If answer is correct, say "No issues found."'
            ),
        },
    ]



def _refinement_messages(question: str, answer: str, critique: str, context: str, failed_checks: list[dict] | None = None) -> list[dict]:
    failed_block = ""
    if failed_checks:
        failed_block = f"Verifier feedback: {failed_checks}\n"
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Improve your answer based on critique and verifier feedback.",
        },
        {
            "role": "user",
            "content": (
                f"{format_context(context)}Question: {question}\n"
                f"Previous answer: {answer}\n"
                f"Critique: {critique}\n"
                f"{failed_block}Provide a concise improved answer:"
            ),
        },
    ]



def run(question: str, ground_truth: str, context: str, is_unanswerable: bool) -> dict:
    del ground_truth

    start_time = time.perf_counter()
    initial_answer = call_llm(
        build_direct_answer_messages(
            question=question,
            context=context,
            system_prompt="You are a helpful assistant. Answer the question directly.",
        )
    )
    snapshots = [
        {
            "iteration": 1,
            "answer": initial_answer,
            "confidence": None,
            "response_time_seconds_cumulative": time.perf_counter() - start_time,
        }
    ]

    critique = call_llm(_critique_messages(question=question, answer=initial_answer, context=context))
    current_answer = call_llm(
        _refinement_messages(
            question=question,
            answer=initial_answer,
            critique=critique,
            context=context,
        )
    )
    snapshots.append(
        {
            "iteration": 2,
            "answer": current_answer,
            "confidence": None,
            "response_time_seconds_cumulative": time.perf_counter() - start_time,
        }
    )

    contradiction_detected, contradiction_response = detect_contradiction(
        question=question,
        answer=current_answer,
        context=context,
    )
    verification_passed, verification_checks = verify(
        question=question,
        answer=current_answer,
        context=context,
        is_unanswerable=is_unanswerable,
        contradiction_detected=contradiction_detected,
        contradiction_response=contradiction_response,
    )

    retries = 0
    while not verification_passed and retries < MAX_RETRIES:
        retries += 1
        current_answer = call_llm(
            _refinement_messages(
                question=question,
                answer=current_answer,
                critique=critique,
                context=context,
                failed_checks=verification_checks,
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
        contradiction_detected, contradiction_response = detect_contradiction(
            question=question,
            answer=current_answer,
            context=context,
        )
        verification_passed, verification_checks = verify(
            question=question,
            answer=current_answer,
            context=context,
            is_unanswerable=is_unanswerable,
            contradiction_detected=contradiction_detected,
            contradiction_response=contradiction_response,
        )

    elapsed = time.perf_counter() - start_time
    return {
        "answer": current_answer,
        "system": SYSTEM_NAME,
        "iterations": 1 + retries,
        "confidence": None,
        "initial_answer": initial_answer,
        "verification_passed": verification_passed,
        "verification_checks": verification_checks,
        "retries": retries,
        "response_time_seconds": elapsed,
        "snapshots": snapshots,
    }
