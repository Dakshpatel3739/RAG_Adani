# Doc_RAG — Hybrid Retrieval with Grounded, Cited Answers

A document Q&A system that answers questions grounded in an uploaded PDF, with
**page-level citations** on every claim and explicit **abstention** when the
retrieved context does not support an answer.

Built to demonstrate the retrieval-engineering pattern that matters in
production: don't just retrieve — retrieve the *right* passages, fuse multiple
signals, and make the generator cite or refuse.

---

## What it does

- **Hybrid retrieval** — BM25 (sparse, lexical) fused with dense embedding
  similarity for better recall on both keyword-heavy and semantic queries.
  Use `--bm25-only` to ablate and measure the lift from the dense component.
- **Grounded generation** — the LLM is prompted to answer only from retrieved
  chunks and to emit `Not found in the document.` when context is insufficient.
  Abstention is a design choice, not a fallback.
- **Page-level citations** — every answer surfaces the top-k retrieved chunks
  with scores and page numbers, so users can verify claims against the source.
- **Persistent index** — chunked and embedded once, cached under `data/index/`,
  reused on subsequent queries. `--reindex` forces a rebuild.
- **Two interfaces** — a browser UI (FastAPI + vanilla JS) for end users, and
  a CLI for batch / scripting use.

## Architecture

```
PDF upload
   │
   ▼
Chunking (page-aware)
   │
   ├──► BM25 index  ─────┐
   │                     │
   └──► Embedding index ─┤
                         │
                         ▼
                  Hybrid retrieval
                  (score fusion, top-k)
                         │
                         ▼
                  Grounded LLM call
                  (cite-or-refuse prompt)
                         │
                         ▼
                Answer + chunk evidence
                (page numbers, scores)
```

## Stack

Python · FastAPI · OpenAI API · `rank_bm25` · sentence-transformers
(embeddings) · HTML / CSS / JS · CLI via `argparse`.

## Quick start

```bash
python3 -m pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python3 server.py
# open http://localhost:8000
```

Upload any PDF, ask questions. Each answer ships with the retrieval evidence
it was grounded on.

## CLI

```bash
python3 main.py --pdf ./doc.pdf      # local file
python3 main.py --pdf https://...    # URL
python3 main.py --bm25-only          # ablate dense retrieval
python3 main.py --reindex            # force rebuild
python3 main.py --no-picker          # skip GUI file dialog
```

## Output contract

Every query prints:

1. Top-k retrieved chunks with hybrid scores and page citations
2. A grounded answer with inline citations, **or** `Not found in the document.`

No answers without evidence. No confident guessing when the document doesn't
cover the question.

## What's next

- Cross-encoder reranking stage on top of hybrid retrieval (precision lift
  at the cost of one extra model call).
- Retrieval evaluation harness — retrieval@k, MRR, groundedness on a held-out
  QA set.
- Port to a Kubernetes-native serving pattern using NVIDIA NIM microservices
  for LLM / embedding / reranking inference with Milvus as the vector store,
  following the architecture covered in the NVIDIA DLI "Deploying RAG
  Pipelines for Production at Scale" course.

## Author

Daksh Patel — [github.com/Dakshpatel3739](https://github.com/Dakshpatel3739)
First-author research at PReMI 2025 (IIT Delhi, Springer LNCS).
NVIDIA Inception Program member.
