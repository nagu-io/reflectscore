import math
import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List

import numpy as np

from config import REFUSAL_PHRASES, SIMILARITY_THRESHOLD


class AutoEvaluator:
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self._embedding_model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def _load_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedding_model = False
        return None if self._embedding_model is False else self._embedding_model

    def _embed(self, text: str) -> np.ndarray:
        normalized_text = text.strip()
        if normalized_text not in self._embedding_cache:
            model = self._load_embedding_model()
            if model is None:
                raise RuntimeError("Embedding model unavailable")
            embedding = model.encode([normalized_text], convert_to_numpy=True, show_progress_bar=False)[0]
            self._embedding_cache[normalized_text] = embedding.astype("float32")
        return self._embedding_cache[normalized_text]

    def lexical_similarity(self, left: str, right: str) -> float:
        token_pattern = re.compile(r"[A-Za-z0-9_\.]+")
        left_tokens = {token.lower() for token in token_pattern.findall(left)}
        right_tokens = {token.lower() for token in token_pattern.findall(right)}
        if not left_tokens or not right_tokens:
            jaccard = 0.0
        else:
            union = len(left_tokens | right_tokens)
            jaccard = len(left_tokens & right_tokens) / union if union else 0.0
        sequence_ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
        return max(jaccard, sequence_ratio)

    def semantic_similarity(self, left: str, right: str) -> float:
        try:
            left_embedding = self._embed(left)
            right_embedding = self._embed(right)
        except RuntimeError:
            return self.lexical_similarity(left, right)

        left_norm = np.linalg.norm(left_embedding)
        right_norm = np.linalg.norm(right_embedding)
        if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
            return 0.0
        return float(np.dot(left_embedding, right_embedding) / (left_norm * right_norm))

    def keyword_match(self, answer: str, keywords: Iterable[str]) -> bool:
        answer_lower = answer.lower()
        for keyword in keywords:
            if keyword and keyword.lower() in answer_lower:
                return True
        return False

    def is_refusal(self, answer: str) -> bool:
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)

    def is_correct(self, answer: str, ground_truth: str, keywords: List[str], unanswerable: bool = False) -> bool:
        if not answer:
            return False
        if unanswerable or ground_truth == "NOT_IN_CONTEXT":
            return self.is_refusal(answer)
        if self.keyword_match(answer=answer, keywords=keywords):
            return True
        return self.semantic_similarity(answer, ground_truth) > self.similarity_threshold
