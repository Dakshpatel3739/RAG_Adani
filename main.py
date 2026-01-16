import argparse
from pathlib import Path

from rag.chat import chat_loop
from rag.env import load_env
from rag.indexing import Chunk, build_index, load_index, save_index, sha256_file, np
from rag.llm import OpenAI, get_openai_client
from rag.pdf_loader import resolve_pdf_path
from rag.retrieval import BM25Okapi, build_bm25


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG chat over a PDF with citations and retrieval debug."
    )
    parser.add_argument("--pdf", help="Path or URL to a PDF")
    picker_group = parser.add_mutually_exclusive_group()
    picker_group.add_argument(
        "--picker",
        action="store_true",
        help="Force a GUI file picker to choose the PDF",
    )
    picker_group.add_argument(
        "--no-picker",
        action="store_true",
        help="Disable GUI picker and use terminal prompt",
    )
    parser.add_argument(
        "--index-dir", default="data/index", help="Directory for index cache"
    )
    parser.add_argument(
        "--cache-dir", default="data/cache", help="Directory for downloaded PDFs"
    )
    parser.add_argument("--reindex", action="store_true", help="Force rebuild index")
    parser.add_argument("--k", type=int, default=10, help="Top-k chunks to retrieve")
    parser.add_argument(
        "--max-chars", type=int, default=1200, help="Max chars per chunk"
    )
    parser.add_argument(
        "--overlap", type=int, default=200, help="Chunk overlap in chars"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="LLM model for answering"
    )
    parser.add_argument(
        "--rewrite-model",
        default="gpt-4o-mini",
        help="LLM model for query rewriting",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="Embedding model for retrieval",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Disable embeddings and use BM25 only",
    )

    args = parser.parse_args()
    if not args.pdf:
        use_picker = not args.no_picker
        if use_picker:
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                pdf_input = filedialog.askopenfilename(
                    title="Select a PDF file",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                )
                root.destroy()
                if pdf_input:
                    args.pdf = pdf_input
            except Exception as exc:
                if args.picker:
                    raise SystemExit(f"Failed to open file picker: {exc}") from exc
        if not args.pdf:
            while True:
                pdf_input = input("Enter PDF path or URL: ").strip()
                if pdf_input:
                    args.pdf = pdf_input
                    break
                print("PDF path/URL is required.")
    load_env()
    ensure_dependencies()

    client = get_openai_client()

    index_dir = Path(args.index_dir)
    cache_dir = Path(args.cache_dir)
    pdf_path = resolve_pdf_path(args.pdf, cache_dir)
    pdf_hash = sha256_file(pdf_path)
    index_path = index_dir / f"index_{pdf_hash}.pkl"

    embed_model = None if args.bm25_only else args.embed_model
    reindex_needed = args.reindex or not index_path.exists()
    if not reindex_needed:
        index_data = load_index(index_path)
        if embed_model != index_data.get("embed_model"):
            reindex_needed = True
    if reindex_needed:
        index_data = build_index(
            pdf_path,
            embed_model=embed_model,
            max_chars=args.max_chars,
            overlap=args.overlap,
            client=client,
        )
        save_index(index_path, index_data)

    chunks = [Chunk(**c) for c in index_data["chunks"]]
    tokenized = index_data["tokenized"]
    embeddings = index_data["embeddings"]
    embed_model = index_data.get("embed_model")
    if args.bm25_only:
        embeddings = None
        embed_model = None

    bm25 = build_bm25(tokenized)

    chat_loop(
        client,
        qa_model=args.model,
        rewrite_model=args.rewrite_model,
        chunks=chunks,
        bm25=bm25,
        embeddings=embeddings,
        embed_model=embed_model,
        top_k=args.k,
    )


if __name__ == "__main__":
    main()
