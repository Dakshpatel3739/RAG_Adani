from typing import List, Optional

from .indexing import Chunk
from .llm import generate_answer, rewrite_query, validate_answer
from .retrieval import RetrievalResult, print_retrieval, retrieve, should_refuse


def chat_loop(
    client,
    qa_model: str,
    rewrite_model: str,
    chunks: List[Chunk],
    bm25,
    embeddings: Optional[object],
    embed_model: Optional[str],
    top_k: int,
) -> None:
    history: List[dict] = []
    print("Type 'exit' to quit.")
    while True:
        question = input("\nQ: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        standalone = rewrite_query(client, rewrite_model, history, question)
        retrieved = retrieve(
            standalone, chunks, bm25, embeddings, embed_model, client, top_k
        )

        print_retrieval(retrieved)

        if should_refuse(standalone, retrieved):
            answer = "Not found in the document."
        else:
            answer = generate_answer(client, qa_model, standalone, retrieved)
            answer = validate_answer(answer, retrieved)

        print("\nAnswer:")
        print(answer)
        history.append({"user": question, "assistant": answer})
