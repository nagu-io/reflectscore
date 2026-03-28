import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import RESULTS_DIR
from evaluation.metrics import (
    backfire_rate,
    bootstrap_ci,
    confidence_calibration,
    correction_rate,
    grounding_score,
    hallucination_rate,
    mean_latency,
    refusal_accuracy,
)

RAW_COLUMNS = [
    "id",
    "question",
    "category",
    "unanswerable",
    "system",
    "answer",
    "ground_truth",
    "is_correct",
    "initial_answer",
    "initial_is_correct",
    "backfired",
    "triggered",
    "confidence",
    "verification_passed",
    "iterations",
    "response_time_seconds",
]

ITERATION_COLUMNS = [
    "id",
    "question",
    "category",
    "system",
    "iteration",
    "answer",
    "confidence",
    "is_correct",
    "response_time_seconds_cumulative",
]


class BenchmarkScorer:
    def __init__(
        self,
        evaluator,
        raw_path: Path | None = None,
        iteration_path: Path | None = None,
        summary_path: Path | None = None,
    ):
        self.evaluator = evaluator
        self.raw_path = raw_path or RESULTS_DIR / "raw_results.csv"
        self.iteration_path = iteration_path or RESULTS_DIR / "iteration_results.csv"
        self.summary_path = summary_path or RESULTS_DIR / "summary.json"

    def load_completed_pairs(self) -> set[tuple[str, str]]:
        dataframe = self.load_raw_results()
        if dataframe.empty:
            return set()
        return set(zip(dataframe["id"], dataframe["system"]))

    def build_final_row(self, question_row: dict, system_result: dict) -> dict:
        is_correct = self.evaluator.is_correct(
            answer=system_result.get("answer", ""),
            ground_truth=question_row["answer"],
            keywords=question_row.get("keywords", []),
            unanswerable=bool(question_row.get("unanswerable", False)),
        )

        initial_answer = system_result.get("initial_answer")
        initial_is_correct = None
        if initial_answer:
            initial_is_correct = self.evaluator.is_correct(
                answer=initial_answer,
                ground_truth=question_row["answer"],
                keywords=question_row.get("keywords", []),
                unanswerable=bool(question_row.get("unanswerable", False)),
            )

        backfired = None
        if initial_answer:
            backfired = bool(initial_is_correct) and not bool(is_correct)

        row = {
            "id": question_row["id"],
            "question": question_row["question"],
            "category": question_row["category"],
            "unanswerable": bool(question_row.get("unanswerable", False)),
            "system": system_result["system"],
            "answer": system_result.get("answer", ""),
            "ground_truth": question_row["answer"],
            "is_correct": bool(is_correct),
            "initial_answer": initial_answer,
            "initial_is_correct": initial_is_correct,
            "backfired": backfired,
            "triggered": system_result.get("triggered"),
            "confidence": system_result.get("confidence"),
            "verification_passed": system_result.get("verification_passed"),
            "iterations": system_result.get("iterations"),
            "response_time_seconds": float(system_result.get("response_time_seconds", 0.0)),
        }
        return row

    def build_iteration_rows(self, question_row: dict, system_result: dict) -> list[dict]:
        rows = []
        for snapshot in system_result.get("snapshots", []):
            answer = snapshot.get("answer", "")
            is_correct = self.evaluator.is_correct(
                answer=answer,
                ground_truth=question_row["answer"],
                keywords=question_row.get("keywords", []),
                unanswerable=bool(question_row.get("unanswerable", False)),
            )
            rows.append(
                {
                    "id": question_row["id"],
                    "question": question_row["question"],
                    "category": question_row["category"],
                    "system": system_result["system"],
                    "iteration": snapshot.get("iteration"),
                    "answer": answer,
                    "confidence": snapshot.get("confidence"),
                    "is_correct": bool(is_correct),
                    "response_time_seconds_cumulative": float(
                        snapshot.get("response_time_seconds_cumulative", 0.0)
                    ),
                }
            )
        return rows

    def _upsert_rows(self, path: Path, rows: list[dict], key_columns: list[str], columns: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            dataframe = pd.read_csv(path)
        else:
            dataframe = pd.DataFrame(columns=columns)

        if dataframe.empty:
            updated = pd.DataFrame(rows)
        else:
            mask = pd.Series([True] * len(dataframe))
            for row in rows:
                row_mask = pd.Series([True] * len(dataframe))
                for column in key_columns:
                    row_mask &= dataframe[column].astype(str) == str(row[column])
                mask &= ~row_mask
            preserved = dataframe[mask]
            updated = pd.concat([preserved, pd.DataFrame(rows)], ignore_index=True)

        ordered_columns = list(dict.fromkeys(columns + list(updated.columns)))
        updated = updated.reindex(columns=ordered_columns)
        updated.to_csv(path, index=False)

    def persist_result(self, question_row: dict, system_result: dict) -> None:
        final_row = self.build_final_row(question_row, system_result)
        iteration_rows = self.build_iteration_rows(question_row, system_result)
        self._upsert_rows(self.raw_path, [final_row], ["id", "system"], RAW_COLUMNS)
        self._upsert_rows(self.iteration_path, iteration_rows, ["id", "system", "iteration"], ITERATION_COLUMNS)

    def load_raw_results(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            return pd.DataFrame(columns=RAW_COLUMNS)
        dataframe = pd.read_csv(self.raw_path)
        for column in ["unanswerable", "is_correct", "triggered", "verification_passed", "initial_is_correct", "backfired"]:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].map(self._coerce_optional_bool)
        return dataframe

    def load_iteration_results(self) -> pd.DataFrame:
        if not self.iteration_path.exists():
            return pd.DataFrame(columns=ITERATION_COLUMNS)
        dataframe = pd.read_csv(self.iteration_path)
        if "is_correct" in dataframe.columns:
            dataframe["is_correct"] = dataframe["is_correct"].map(self._coerce_optional_bool)
        return dataframe

    @staticmethod
    def _coerce_optional_bool(value):
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        return bool(value)

    def build_summary(self, final_results: Iterable[dict], question_lookup: dict[str, dict]) -> dict:
        summary = {}
        rows = list(final_results)
        systems = sorted({row["system"] for row in rows})
        for system in systems:
            system_rows = []
            for row in rows:
                if row["system"] != system:
                    continue
                merged = dict(row)
                question_row = question_lookup[row["id"]]
                merged["keywords"] = question_row.get("keywords", [])
                merged["file_reference"] = question_row.get("file_reference")
                system_rows.append(merged)

            summary[system] = {
                "hallucination_rate": hallucination_rate(system_rows, self.evaluator),
                "hallucination_rate_ci": bootstrap_ci(system_rows, hallucination_rate, self.evaluator),
                "grounding_score": grounding_score(system_rows, self.evaluator),
                "grounding_score_ci": bootstrap_ci(system_rows, grounding_score, self.evaluator),
                "refusal_accuracy": refusal_accuracy(system_rows, self.evaluator),
                "refusal_accuracy_ci": bootstrap_ci(system_rows, refusal_accuracy, self.evaluator),
                "correction_rate": correction_rate(system_rows, self.evaluator),
                "correction_rate_ci": bootstrap_ci(system_rows, correction_rate, self.evaluator),
                "backfire_rate": backfire_rate(system_rows, self.evaluator),
                "backfire_rate_ci": bootstrap_ci(system_rows, backfire_rate, self.evaluator),
                "confidence_calibration": confidence_calibration(system_rows, self.evaluator),
                "mean_response_time_seconds": mean_latency(system_rows, self.evaluator),
            }
            if system == "baseline":
                summary[system]["correction_rate"] = None
                summary[system]["correction_rate_ci"] = None
                summary[system]["backfire_rate"] = None
                summary[system]["backfire_rate_ci"] = None
        return summary

    def save_summary(self, summary: dict) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
