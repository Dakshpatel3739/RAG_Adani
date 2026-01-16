import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag.env import load_env
from rag.indexing import Chunk, build_index, load_index, save_index
from rag.llm import OpenAI, answer_with_retry, get_openai_client, rewrite_query
from rag.retrieval import (
    BM25Okapi,
    RetrievalResult,
    build_bm25,
    retrieve,
    should_refuse,
    snippet,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
INDEX_DIR = BASE_DIR / "data" / "index"


def ensure_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if BM25Okapi is None:
        missing.append("rank_bm25")
    if OpenAI is None:
        missing.append("openai")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Missing dependencies: "
            + joined
            + ". Install with: pip install -r requirements.txt"
        )


def sanitize_filename(filename: str) -> str:
    keep = []
    for ch in filename:
        if ch.isalnum() or ch in {".", "_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "document.pdf"


@dataclass
class SessionState:
    session_id: str
    pdf_name: str
    chunks: List[Chunk]
    bm25: "BM25Okapi"
    embeddings: Optional["np.ndarray"]
    embed_model: Optional[str]
    qa_model: str
    rewrite_model: str
    history: List[dict] = field(default_factory=list)


app = FastAPI(title="PDF RAG Chat")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

SESSIONS: Dict[str, SessionState] = {}


@app.on_event("startup")
def startup() -> None:
    load_env()
    ensure_dependencies()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    app.state.client = get_openai_client()
    app.state.embed_model = "text-embedding-3-small"
    app.state.qa_model = "gpt-4o-mini"
    app.state.rewrite_model = "gpt-4o-mini"


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    bm25_only: bool = Form(False),
) -> RedirectResponse:
    if not file.filename:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Please choose a PDF file."},
            status_code=400,
        )
    filename = sanitize_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Only PDF files are supported."},
            status_code=400,
        )

    data = await file.read()
    if not data:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Uploaded file is empty."},
            status_code=400,
        )

    file_hash = hashlib.sha256(data).hexdigest()
    stored_name = f"{file_hash[:8]}_{filename}"
    pdf_path = UPLOAD_DIR / stored_name
    pdf_path.write_bytes(data)

    index_path = INDEX_DIR / f"index_{file_hash}.pkl"
    embed_model = None if bm25_only else app.state.embed_model

    if index_path.exists():
        index_data = load_index(index_path)
        if embed_model != index_data.get("embed_model"):
            index_data = build_index(
                pdf_path,
                embed_model=embed_model,
                max_chars=1200,
                overlap=200,
                client=app.state.client,
            )
            save_index(index_path, index_data)
    else:
        index_data = build_index(
            pdf_path,
            embed_model=embed_model,
            max_chars=1200,
            overlap=200,
            client=app.state.client,
        )
        save_index(index_path, index_data)

    chunks = [Chunk(**c) for c in index_data["chunks"]]
    tokenized = index_data["tokenized"]
    embeddings = index_data["embeddings"]
    embed_model = index_data.get("embed_model")
    if bm25_only:
        embeddings = None
        embed_model = None

    bm25 = build_bm25(tokenized)

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = SessionState(
        session_id=session_id,
        pdf_name=filename,
        chunks=chunks,
        bm25=bm25,
        embeddings=embeddings,
        embed_model=embed_model,
        qa_model=app.state.qa_model,
        rewrite_model=app.state.rewrite_model,
    )

    return RedirectResponse(url=f"/chat/{session_id}", status_code=303)


@app.get("/chat/{session_id}", response_class=HTMLResponse)
def chat_page(request: Request, session_id: str) -> HTMLResponse:
    session = SESSIONS.get(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "session_id": session_id,
            "pdf_name": session.pdf_name,
        },
    )


@app.post("/ask")
async def ask_question(payload: dict) -> JSONResponse:
    session_id = payload.get("session_id")
    question = (payload.get("question") or "").strip()
    if not session_id or session_id not in SESSIONS:
        return JSONResponse({"error": "Invalid session."}, status_code=400)
    if not question:
        return JSONResponse({"error": "Question is required."}, status_code=400)

    session = SESSIONS[session_id]
    client = app.state.client

    standalone = rewrite_query(client, session.rewrite_model, session.history, question)
    retrieved = retrieve(
        standalone,
        session.chunks,
        session.bm25,
        session.embeddings,
        session.embed_model,
        client,
        top_k=8,
    )

    if should_refuse(standalone, retrieved):
        answer = "Not found in the document."
    else:
        answer = answer_with_retry(client, session.qa_model, standalone, retrieved)

    session.history.append({"user": question, "assistant": answer})

    retrieved_payload = [
        {
            "citation": f"p{item.chunk.page}:{item.chunk.cid}",
            "score": round(item.score, 3),
            "bm25": round(item.bm25, 3),
            "cosine": round(item.cosine, 3),
            "snippet": snippet(item.chunk.text),
        }
        for item in retrieved
    ]

    return JSONResponse(
        {
            "answer": answer,
            "retrieved": retrieved_payload,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
