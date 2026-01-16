# PDF RAG Chat

A minimal RAG chat app that answers questions grounded in a PDF and cites page/chunk sources.

## Setup

1) Install deps
```
pip install -r requirements.txt
```

2) Add your OpenAI key in `.env`
```
OPENAI_API_KEY=your_api_key_here
```

## Run

Local PDF (auto-opens a GUI file picker if available):
```
python3 main.py
```

Or pass a path directly:
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

## Web UI (Port 8000)

Start the upload + chat UI:
```
python3 server.py
```

Then open `http://localhost:8000` and upload any PDF. The UI will build the index and open a chat panel with retrieval evidence.

## Output

Each question prints:
- Top-k retrieved chunks with scores and citations
- A grounded answer with citations, or `Not found in the document.`
