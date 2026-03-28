def format_context(context: str) -> str:
    if not context:
        return ""
    return f"Context:\n{context}\n\n"



def build_direct_answer_messages(question: str, context: str, system_prompt: str) -> list[dict]:
    user_prompt = (
        f"{format_context(context)}Question: {question}\n"
        "Answer concisely using the provided context when relevant. "
        "If the answer is not in the context, say so instead of guessing. "
        "Prefer 1-2 short sentences, or just the function/file name when that fully answers the question."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
