import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .chunking import chunk_text
from .pdf_loader import extract_pdf_pages, normalize_text

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None


@dataclass
class Chunk:
    cid: str
    page: int
    text: str


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"[A-Za-z0-9]+", text.lower())


def embed_texts(client, texts: List[str], model: str) -> "np.ndarray":
    if np is None:
        raise SystemExit("numpy not installed. Run: pip install -r requirements.txt")
    embeddings = []
    batch_size = 64
    for idx in range(0, len(texts), batch_size):
        batch = texts[idx : idx + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return np.array(embeddings, dtype="float32")


def build_index(
    pdf_path: Path,
    embed_model: Optional[str],
    max_chars: int,
    overlap: int,
    client,
) -> dict:
    pages: List[Tuple[int, str]] = extract_pdf_pages(pdf_path)
    if not pages:
        raise SystemExit("No text extracted from PDF.")

    chunks: List[Chunk] = []
    tokenized: List[List[str]] = []
    for page_num, text in pages:
        cleaned = normalize_text(text)
        for chunk in chunk_text(cleaned, max_chars=max_chars, overlap=overlap):
            cid = f"c{len(chunks)}"
            chunk_obj = Chunk(cid=cid, page=page_num, text=chunk)
            chunks.append(chunk_obj)
            tokenized.append(tokenize(chunk))

    embeddings = None
    if embed_model:
        if client is None:
            raise SystemExit("OpenAI client not available for embeddings.")
        embeddings = embed_texts(client, [c.text for c in chunks], embed_model)

    return {
        "version": 1,
        "pdf_path": str(pdf_path),
        "chunks": [c.__dict__ for c in chunks],
        "tokenized": tokenized,
        "embeddings": embeddings,
        "embed_model": embed_model,
    }


def save_index(index_path: Path, index_data: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as handle:
        pickle.dump(index_data, handle)


def load_index(index_path: Path) -> dict:
    with index_path.open("rb") as handle:
        return pickle.load(handle)
