from typing import List, Optional

from .indexing import Chunk
from .llm import answer_with_retry, rewrite_query
from .retrieval import (
    RetrievalResult,
    expand_queries,
    merge_results,
    print_retrieval,
    retrieve,
    select_context,
    should_refuse,
)


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
        queries = expand_queries(standalone)
        results = [
            retrieve(q, chunks, bm25, embeddings, embed_model, client, top_k)
            for q in queries
        ]
        retrieved = merge_results(results, top_k)

        print_retrieval(retrieved)

        if should_refuse(standalone, retrieved):
            answer = "Not found in the document."
        else:
            context_query = " ".join(queries)
            context_results = select_context(context_query, retrieved, max_chunks=4)
            answer = answer_with_retry(client, qa_model, standalone, context_results)

        print("\nAnswer:")
        print(answer)
        history.append({"user": question, "assistant": answer})
