import math
from typing import Callable, Iterable

import numpy as np

from config import BOOTSTRAP_RESAMPLES, SEED


MetricFn = Callable[[list[dict], object | None], float | dict | None]



def _to_list(results: Iterable[dict]) -> list[dict]:
    return list(results)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return False
        return bool(value)
    return bool(value)


def _has_real_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        if math.isnan(value):
            return False
        return True
    if isinstance(value, str):
        return bool(value.strip())
    return True



def hallucination_rate(results: Iterable[dict], evaluator=None):
    rows = _to_list(results)
    if not rows:
        return None
    incorrect = sum(1 for row in rows if not _as_bool(row.get("is_correct", False)))
    return incorrect / len(rows)



def grounding_score(results: Iterable[dict], evaluator=None):
    rows = [row for row in _to_list(results) if row.get("category") == "code"]
    if not rows:
        return None
    grounded = 0
    for row in rows:
        answer = str(row.get("answer", ""))
        file_reference = str(row.get("file_reference", ""))
        keywords = row.get("keywords", []) or []
        primary_keyword = keywords[0] if keywords else ""
        if file_reference and file_reference in answer:
            grounded += 1
        elif primary_keyword and primary_keyword in answer:
            grounded += 1
    return grounded / len(rows)



def refusal_accuracy(results: Iterable[dict], evaluator=None):
    rows = [row for row in _to_list(results) if bool(row.get("unanswerable"))]
    if not rows:
        return None
    correct = sum(1 for row in rows if _as_bool(row.get("is_correct", False)))
    return correct / len(rows)



def correction_rate(results: Iterable[dict], evaluator=None):
    rows = [row for row in _to_list(results) if _has_real_value(row.get("initial_answer"))]
    if not rows:
        return None
    corrected = sum(
        1
        for row in rows
        if not _as_bool(row.get("initial_is_correct", False)) and _as_bool(row.get("is_correct", False))
    )
    return corrected / len(rows)



def backfire_rate(results: Iterable[dict], evaluator=None):
    rows = [row for row in _to_list(results) if _has_real_value(row.get("initial_answer"))]
    if not rows:
        return None
    backfired = sum(
        1
        for row in rows
        if _as_bool(row.get("initial_is_correct", False)) and not _as_bool(row.get("is_correct", False))
    )
    return backfired / len(rows)



def confidence_calibration(results: Iterable[dict], evaluator=None):
    rows = [row for row in _to_list(results) if row.get("system") == "confidence_reflection"]
    if not rows:
        return None

    buckets = {0.0: [], 0.2: [], 0.4: [], 0.6: [], 0.8: [], 1.0: []}
    for row in rows:
        try:
            confidence = float(row.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        bucket = min(1.0, math.floor((confidence / 0.2) + 0.5) * 0.2)
        bucket = round(bucket, 1)
        buckets.setdefault(bucket, []).append(1.0 if _as_bool(row.get("is_correct", False)) else 0.0)

    calibration = {}
    for bucket, values in buckets.items():
        if values:
            calibration[f"{bucket:.1f}"] = float(sum(values) / len(values))
    return calibration



def mean_latency(results: Iterable[dict], evaluator=None):
    rows = _to_list(results)
    if not rows:
        return None
    values = [float(row.get("response_time_seconds", 0.0) or 0.0) for row in rows]
    return float(sum(values) / len(values))



def bootstrap_ci(
    results: Iterable[dict],
    metric_fn: MetricFn,
    evaluator=None,
    seed: int = SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
):
    rows = _to_list(results)
    if not rows:
        return None

    baseline = metric_fn(rows, evaluator)
    if baseline is None or isinstance(baseline, dict):
        return None

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[index] for index in indices]
        value = metric_fn(sample, evaluator)
        if value is not None and not isinstance(value, dict):
            samples.append(float(value))

    if not samples:
        return None
    lower = float(np.percentile(samples, 2.5))
    upper = float(np.percentile(samples, 97.5))
    return [lower, upper]
