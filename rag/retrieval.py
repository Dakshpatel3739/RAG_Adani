import re
from dataclasses import dataclass
from typing import List, Optional

from .indexing import Chunk, embed_texts, tokenize

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - handled at runtime
    BM25Okapi = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    bm25: float
    cosine: float


def build_bm25(tokenized: List[List[str]]) -> "BM25Okapi":
    if BM25Okapi is None:
        raise SystemExit("rank_bm25 not installed. Run: pip install -r requirements.txt")
    return BM25Okapi(tokenized)


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return [0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def cosine_similarity(matrix: "np.ndarray", vector: "np.ndarray") -> List[float]:
    if np is None:
        raise SystemExit("numpy not installed. Run: pip install -r requirements.txt")
    if matrix.size == 0:
        return []
    vector = vector.astype("float32")
    dot = matrix @ vector
    matrix_norm = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    denom = matrix_norm * vector_norm
    scores = np.where(denom == 0, 0.0, dot / denom)
    return scores.tolist()


def retrieve(
    query: str,
    chunks: List[Chunk],
    bm25: "BM25Okapi",
    embeddings: Optional["np.ndarray"],
    embed_model: Optional[str],
    client,
    top_k: int,
) -> List[RetrievalResult]:
    tokens = tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    if hasattr(bm25_scores, "tolist"):
        bm25_scores = bm25_scores.tolist()

    cosine_scores = [0.0] * len(chunks)
    if embeddings is not None and embed_model and client is not None:
        query_emb = embed_texts(client, [query], embed_model)[0]
        cosine_scores = cosine_similarity(embeddings, query_emb)

    bm25_norm = normalize_scores(bm25_scores)
    cosine_norm = normalize_scores(cosine_scores)

    combined = []
    for idx in range(len(chunks)):
        if embeddings is not None:
            score = 0.5 * bm25_norm[idx] + 0.5 * cosine_norm[idx]
        else:
            score = bm25_norm[idx]
        combined.append(score)

    top_indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        results.append(
            RetrievalResult(
                chunk=chunks[idx],
                score=combined[idx],
                bm25=bm25_scores[idx],
                cosine=cosine_scores[idx],
            )
        )
    return results


def snippet(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_retrieval(retrieved: List[RetrievalResult]) -> None:
    print("Retrieved:")
    for result in retrieved:
        chunk = result.chunk
        print(
            f"- [p{chunk.page}:{chunk.cid}] score={result.score:.3f} "
            f"bm25={result.bm25:.3f} cosine={result.cosine:.3f}"
        )
        print(f"  {snippet(chunk.text)}")


def has_query_overlap(question: str, text: str) -> bool:
    query_tokens = [t for t in tokenize(question) if t not in STOPWORDS and len(t) > 2]
    if not query_tokens:
        return True
    text_tokens = set(tokenize(text))
    return any(t in text_tokens for t in query_tokens)


def should_refuse(question: str, retrieved: List[RetrievalResult]) -> bool:
    if not retrieved:
        return True
    best_score = max(item.score for item in retrieved)
    if best_score < 0.12:
        return True
    if not any(has_query_overlap(question, item.chunk.text) for item in retrieved[:3]):
        return True
    return False
