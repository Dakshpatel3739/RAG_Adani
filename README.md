# PDF RAG Chat

A minimal RAG app that answers questions grounded in a PDF and cites page/chunk sources.

## Quick Start (Upload Any PDF)

1) Install dependencies
```
python3 -m pip install -r requirements.txt
```

2) Add your OpenAI key in `.env`
```
OPENAI_API_KEY=your_api_key_here
```

3) Start the web UI
```
python3 server.py
```

4) Open `http://localhost:8000`, upload a PDF, and start chatting.

The UI builds an index from your uploaded file and shows retrieval evidence with each answer.

## CLI Mode (Optional)

Local PDF (auto-opens a GUI file picker if available):
```
python3 main.py
```

Pass a path directly:
```
python3 main.py --pdf ./doc.pdf
```

PDF URL:
```
python3 main.py --pdf "https://example.com/doc.pdf"
```

Optional flags:
- `--bm25-only` to skip embeddings
- `--reindex` to rebuild the index
- `--no-picker` to skip the GUI picker and use the terminal prompt

## Output Format

Each question prints:
- Top-k retrieved chunks with scores and citations
- A grounded answer with citations, or `Not found in the document.`

## Notes

- Uploaded PDFs are stored under `data/uploads`.
- Indexes are cached under `data/index` for faster re-runs.
