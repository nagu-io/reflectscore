import json
import os
import sys
import threading
import time
from typing import Any

from google import genai
from google.genai import errors, types

from config import GEMINI_API_KEY, MAX_TOKENS, MODEL, RATE_LIMIT_COOLDOWN_SECONDS, TEMPERATURE, validate_runtime_config


class LLMError(RuntimeError):
    pass


_RATE_LOCK = threading.Lock()
_CLIENT_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_COOLDOWN_UNTIL = 0.0
_CLIENT: genai.Client | None = None
LLM_VERBOSE = os.getenv("LLM_VERBOSE", "0") == "1"
MIN_REQUEST_INTERVAL_SECONDS = 3.0
MAX_RATE_LIMIT_RETRIES = 3
MAX_SERVER_RETRIES = 2
SERVER_RETRY_WAIT_SECONDS = 5.0


def _apply_request_pacing() -> None:
    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        now = time.monotonic()
        wait_until = max(_COOLDOWN_UNTIL, _LAST_REQUEST_AT + MIN_REQUEST_INTERVAL_SECONDS)
        sleep_seconds = max(0.0, wait_until - now)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        _LAST_REQUEST_AT = time.monotonic()



def _enter_cooldown(wait_seconds: float) -> None:
    global _COOLDOWN_UNTIL
    with _RATE_LOCK:
        _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + max(0.0, wait_seconds))



def _log(message: str, *, always: bool = False) -> None:
    if always or LLM_VERBOSE:
        print(message, file=sys.stderr, flush=True)



def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("Could not find JSON object", text, 0)



def _ensure_client_configured() -> None:
    global _CLIENT
    if _CLIENT is not None:
        return
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = genai.Client(api_key=GEMINI_API_KEY)



def _split_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    return system_instruction, contents



def _build_model(system_instruction: str | None):
    config_kwargs: dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    return types.GenerateContentConfig(**config_kwargs)



def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())
        if chunks:
            return "\n".join(chunks).strip()

    raise LLMError(f"Unexpected Gemini response payload: {response!r}")



def _status_code_from_exception(exc: Exception) -> int | None:
    if isinstance(exc, errors.APIError):
        status = getattr(exc, "status", None)
        if isinstance(status, int):
            return status
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code

    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
        if hasattr(value, "value") and isinstance(value.value, int):
            return value.value

    text = str(exc)
    for code in (429, 500, 502, 503, 504):
        if str(code) in text:
            return code
    return None



def call_llm(messages: list[dict[str, str]], temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    validate_runtime_config()
    _ensure_client_configured()

    system_instruction, contents = _split_messages(messages)
    generation_config = _build_model(system_instruction)
    generation_config.temperature = temperature
    generation_config.max_output_tokens = max_tokens

    rate_limit_retries = 0
    server_retries = 0
    attempts = 0

    while True:
        attempts += 1
        _apply_request_pacing()
        request_started_at = time.perf_counter()
        try:
            response = _CLIENT.models.generate_content(
                model=MODEL,
                contents=contents,
                config=generation_config,
            )
            elapsed = time.perf_counter() - request_started_at
            _log(f"Gemini request succeeded in {elapsed:.2f}s on attempt {attempts}.")
            return _response_text(response)
        except Exception as exc:
            status_code = _status_code_from_exception(exc)

            if status_code == 429 and rate_limit_retries < MAX_RATE_LIMIT_RETRIES:
                rate_limit_retries += 1
                wait_seconds = float(RATE_LIMIT_COOLDOWN_SECONDS)
                _enter_cooldown(wait_seconds)
                _log(
                    f"Gemini rate limit hit on attempt {attempts}. Waiting {wait_seconds:.1f}s before retry {rate_limit_retries}/{MAX_RATE_LIMIT_RETRIES}.",
                    always=True,
                )
                time.sleep(wait_seconds)
                continue

            if status_code is not None and status_code >= 500 and server_retries < MAX_SERVER_RETRIES:
                server_retries += 1
                _log(
                    f"Gemini server error {status_code} on attempt {attempts}. Waiting {SERVER_RETRY_WAIT_SECONDS:.1f}s before retry {server_retries}/{MAX_SERVER_RETRIES}.",
                    always=True,
                )
                time.sleep(SERVER_RETRY_WAIT_SECONDS)
                continue

            if status_code == 429:
                raise LLMError(f"Gemini rate limit exceeded after {MAX_RATE_LIMIT_RETRIES} retries.") from exc
            if status_code is not None and status_code >= 500:
                raise LLMError(f"Gemini server error after retries: {status_code} {exc}") from exc
            raise LLMError(f"Gemini request failed: {exc}") from exc



def call_llm_json(messages: list[dict[str, str]], temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    raw_text = call_llm(messages=messages, temperature=temperature, max_tokens=max_tokens)
    try:
        parsed = _extract_json_object(raw_text)
        parsed["_raw_text"] = raw_text
        return parsed
    except json.JSONDecodeError:
        return {"answer": raw_text, "confidence": 0.5, "reason": "JSON parsing failed.", "_raw_text": raw_text}
