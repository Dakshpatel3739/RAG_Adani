import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

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


def merge_results(results_list: Iterable[List[RetrievalResult]], top_k: int) -> List[RetrievalResult]:
    merged = {}
    for results in results_list:
        for item in results:
            key = (item.chunk.page, item.chunk.cid)
            if key not in merged or item.score > merged[key].score:
                merged[key] = item
    ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    return ranked[:top_k]


def expand_queries(question: str) -> List[str]:
    base = question.strip()
    if not base:
        return []
    q = base.lower()
    queries = [base]
    if "segment" in q:
        queries.append("business portfolio segments")
        queries.append(q.replace("segments", "business portfolio").replace("segment", "business portfolio"))
    if "driver" in q or "drivers" in q or "reason" in q or "cause" in q:
        queries.append(q.replace("drivers", "reasons").replace("driver", "reason"))
        queries.append(q.replace("drivers", "impacts").replace("driver", "impact"))
        queries.append(f"impact due to {q}")
        queries.append(f"{q} impact")
    if "change" in q or "changes" in q:
        queries.append(q.replace("changes", "yoy change").replace("change", "yoy change"))
        queries.append(q.replace("changes", "impact").replace("change", "impact"))
    if "passenger" in q or "cargo" in q:
        queries.append("passenger cargo H1-26 H1-25 change")
    if "ebitda" in q and "h1" in q and ("driver" in q or "change" in q):
        queries.append("EBITDA impact due to volume price tariff H1-26")

    deduped = []
    seen = set()
    for item in queries:
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


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
    query_tokens = expand_query_tokens(query_tokens)
    return any(t in text_tokens for t in query_tokens)


def overlap_score(question: str, text: str) -> int:
    query_tokens = [t for t in tokenize(question) if t not in STOPWORDS and len(t) > 2]
    if not query_tokens:
        return 0
    text_tokens = set(tokenize(text))
    query_tokens = expand_query_tokens(query_tokens)
    return sum(1 for t in query_tokens if t in text_tokens)


def expand_query_tokens(tokens: List[str]) -> List[str]:
    expanded = set(tokens)
    if "passenger" in expanded or "passengers" in expanded:
        expanded.add("pax")
    if "cargo" in expanded:
        expanded.add("cargo")
    if "airport" in expanded or "airports" in expanded:
        expanded.add("aahl")
    if "air" in expanded and "traffic" in expanded:
        expanded.add("atm")
    return list(expanded)


def select_context(question: str, retrieved: List[RetrievalResult], max_chunks: int = 4) -> List[RetrievalResult]:
    if len(retrieved) <= max_chunks:
        return retrieved
    scored = []
    for item in retrieved:
        scored.append((item.score, overlap_score(question, item.chunk.text), item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item for _, _, item in scored[:max_chunks]]


def should_refuse(question: str, retrieved: List[RetrievalResult]) -> bool:
    if not retrieved:
        return True
    best_score = max(item.score for item in retrieved)
    if best_score < 0.12:
        return True
    if not any(has_query_overlap(question, item.chunk.text) for item in retrieved[:3]):
        return True
    return False
