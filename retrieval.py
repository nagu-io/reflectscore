import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, TypeVar

import numpy as np

from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, TOP_K_RETRIEVAL


@dataclass
class RetrievedChunk:
    chunk_id: int
    source: str
    text: str
    score: float


TOKEN_PATTERN = re.compile(r"\S+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9_\.]+")
FILE_HEADER_PATTERN = re.compile(r"^# file: (?P<name>.+)$", re.MULTILINE)
FUNCTION_PATTERN = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CLASS_PATTERN = re.compile(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|:)")
TOKENIZER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ChunkToken = TypeVar("ChunkToken")



def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)



def normalize_terms(text: str) -> set[str]:
    return {term.lower() for term in WORD_PATTERN.findall(text)}



def lexical_overlap_score(query: str, text: str) -> float:
    query_terms = normalize_terms(query)
    text_terms = normalize_terms(text)
    if not query_terms or not text_terms:
        return 0.0
    overlap = len(query_terms & text_terms)
    union = len(query_terms | text_terms)
    return overlap / union if union else 0.0



def chunk_tokens(
    tokens: Sequence[ChunkToken],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> List[List[ChunkToken]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        chunk = list(tokens[start : start + chunk_size])
        if not chunk:
            break
        chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks



def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    tokenizer=None,
) -> List[str]:
    if tokenizer is None:
        return [" ".join(chunk) for chunk in chunk_tokens(tokenize(text), chunk_size=chunk_size, overlap=overlap)]

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for token_chunk in chunk_tokens(token_ids, chunk_size=chunk_size, overlap=overlap):
        decoded = tokenizer.decode(
            token_chunk,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if decoded:
            chunks.append(decoded)
    return chunks



def split_code_sections(text: str) -> List[tuple[str, str]]:
    headers = list(FILE_HEADER_PATTERN.finditer(text))
    if not headers:
        return [("code_context.txt", text)]

    sections = []
    for index, match in enumerate(headers):
        start = match.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        source = match.group("name").strip()
        section_text = text[start:end].strip()
        sections.append((source, section_text))
    return sections


class CodeRetriever:
    def __init__(self, context_path: str | Path):
        self.context_path = Path(context_path)
        self.context_text = self.context_path.read_text(encoding="utf-8")
        self._tokenizer = None
        self.chunking_backend = "whitespace_fallback"
        self._embedding_model = None
        self.embedding_backend = "lexical_fallback"
        self.chunks = self._build_chunks(self.context_text)
        self.chunk_texts = [chunk.text for chunk in self.chunks]
        self._chunk_matrix = None
        self._index = None
        self._use_faiss = False

    def _load_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
                self.chunking_backend = "model_tokenizer"
            except Exception:
                self._tokenizer = False
                self.chunking_backend = "whitespace_fallback"
        return None if self._tokenizer is False else self._tokenizer

    def _load_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_backend = "sentence_transformer"
            except Exception:
                self._embedding_model = False
                self.embedding_backend = "lexical_fallback"
        return None if self._embedding_model is False else self._embedding_model

    def _build_chunks(self, text: str) -> List[RetrievedChunk]:
        built = []
        chunk_id = 0
        tokenizer = self._load_tokenizer()
        for source, section_text in split_code_sections(text):
            for chunk_text_value in chunk_text(section_text, tokenizer=tokenizer):
                built.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        source=source,
                        text=chunk_text_value,
                        score=0.0,
                    )
                )
                chunk_id += 1
        return built

    def _build_index(self) -> None:
        if self._chunk_matrix is not None:
            return
        model = self._load_embedding_model()
        if model is None:
            return
        embeddings = model.encode(self.chunk_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._chunk_matrix = embeddings / norms

        try:
            import faiss

            index = faiss.IndexFlatIP(self._chunk_matrix.shape[1])
            index.add(self._chunk_matrix)
            self._index = index
            self._use_faiss = True
        except ImportError:
            self._index = None
            self._use_faiss = False

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[RetrievedChunk]:
        if not self.chunks:
            return []
        self._build_index()

        if self._chunk_matrix is not None:
            model = self._load_embedding_model()
            query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
            query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_norm[query_norm == 0] = 1.0
            normalized_query = query_embedding / query_norm

            if self._use_faiss:
                scores, indices = self._index.search(normalized_query, top_k)
                pairs = zip(indices[0].tolist(), scores[0].tolist())
            else:
                similarity = np.dot(self._chunk_matrix, normalized_query[0])
                indices = np.argsort(similarity)[::-1][:top_k]
                pairs = ((int(index), float(similarity[index])) for index in indices)
        else:
            lexical_scores = [lexical_overlap_score(query, chunk_text_value) for chunk_text_value in self.chunk_texts]
            indices = np.argsort(lexical_scores)[::-1][:top_k]
            pairs = ((int(index), float(lexical_scores[index])) for index in indices)

        retrieved = []
        for chunk_index, score in pairs:
            chunk = self.chunks[chunk_index]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                )
            )
        return retrieved

    def format_context(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
        retrieved = self.retrieve(query=query, top_k=top_k)
        if not retrieved:
            return ""
        blocks = []
        for position, chunk in enumerate(retrieved, start=1):
            blocks.append(
                f"[Chunk {position} | source={chunk.source} | score={chunk.score:.3f}]\n{chunk.text}"
            )
        return "\n\n".join(blocks)

    def extract_function_names(self, text: str | None = None) -> List[str]:
        search_text = text if text is not None else self.context_text
        return sorted(set(FUNCTION_PATTERN.findall(search_text)))

    def extract_class_names(self, text: str | None = None) -> List[str]:
        search_text = text if text is not None else self.context_text
        return sorted(set(CLASS_PATTERN.findall(search_text)))

    def extract_file_references(self, text: str | None = None) -> List[str]:
        search_text = text if text is not None else self.context_text
        return sorted(set(match.group("name").strip() for match in FILE_HEADER_PATTERN.finditer(search_text)))

    def extract_symbols(self, text: str | None = None) -> List[str]:
        return sorted(set(self.extract_function_names(text) + self.extract_class_names(text) + self.extract_file_references(text)))
