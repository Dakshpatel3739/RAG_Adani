import os
import re
from typing import List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None

from .retrieval import RetrievalResult


def get_openai_client() -> "OpenAI":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Add it to .env")
    if OpenAI is None:
        raise SystemExit("openai package not installed. Run: pip install -r requirements.txt")
    return OpenAI(api_key=api_key)


def rewrite_query(client: "OpenAI", model: str, history: List[dict], question: str) -> str:
    if not history:
        return question
    prompt = (
        "Rewrite the latest question into a standalone search query. "
        "Use chat history only to resolve references. "
        "Return only the rewritten question."
    )
    history_text = "\n".join(
        [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in history[-4:]]
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Chat history:\n{history_text}\n\nLatest question: {question}",
        },
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten or question


def generate_answer(
    client: "OpenAI",
    model: str,
    question: str,
    retrieved: List[RetrievalResult],
    strict: bool = False,
) -> str:
    if not retrieved:
        return "Not found in the document."

    context_lines = []
    for result in retrieved:
        chunk = result.chunk
        context_lines.append(f"[p{chunk.page}:{chunk.cid}] {chunk.text}")
    context = "\n".join(context_lines)

    system_prompt = (
        "You are a factual QA assistant. Use only the provided context to answer. "
        "If the answer is not fully supported by the context, reply exactly: "
        "Not found in the document.\n\n"
        "Rules:\n"
        "- Do not use outside knowledge.\n"
        "- Answer in 1-2 short sentences.\n"
        "- Include citations in square brackets with page and chunk ids for every sentence, "
        "like [p13:c42].\n"
        "- If you output 'Not found in the document.', output nothing else."
    )
    if strict:
        system_prompt += (
            "\n- Citations are mandatory; use only the provided context citations."
        )

    user_prompt = f"Question: {question}\n\nContext:\n{context}"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    return answer


def _normalize_number(value: str) -> str:
    value = value.strip().strip(",")
    value = value.replace(",", "")
    return value


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"-?\d[\d,]*\.?\d*", text)


def _numbers_supported(answer: str, context: str) -> bool:
    answer_wo_cites = re.sub(r"\[p\d+(?::c\d+)?\]", "", answer, flags=re.IGNORECASE)
    answer_nums = [_normalize_number(n) for n in _extract_numbers(answer_wo_cites)]
    if not answer_nums:
        return True
    context_nums = {_normalize_number(n) for n in _extract_numbers(context)}
    return all(num in context_nums for num in answer_nums)


def validate_answer(answer: str, retrieved: List[RetrievalResult]) -> str:
    if answer.strip().lower() == "not found in the document.":
        return "Not found in the document."

    citation_pattern = re.compile(r"\[(p\d+(?::c\d+)?)\]", re.IGNORECASE)
    citations = citation_pattern.findall(answer)
    if not citations:
        return "Not found in the document."

    pages = {f"p{res.chunk.page}" for res in retrieved}
    chunk_citations = {f"p{res.chunk.page}:{res.chunk.cid}" for res in retrieved}
    allowed_pages = {p.lower() for p in pages}
    allowed_chunks = {c.lower() for c in chunk_citations}
    for cite in citations:
        cite = cite.lower()
        if ":" in cite:
            if cite not in allowed_chunks:
                return "Not found in the document."
        else:
            if cite not in allowed_pages:
                return "Not found in the document."

    context = "\n".join(
        f"[p{res.chunk.page}:{res.chunk.cid}] {res.chunk.text}" for res in retrieved
    )
    if not _numbers_supported(answer, context):
        return "Not found in the document."

    return answer


def answer_with_retry(
    client: "OpenAI",
    model: str,
    question: str,
    retrieved: List[RetrievalResult],
    retries: int = 1,
) -> str:
    answer = generate_answer(client, model, question, retrieved, strict=False)
    validated = validate_answer(answer, retrieved)
    if validated != "Not found in the document." or retries <= 0:
        return validated
    answer = generate_answer(client, model, question, retrieved, strict=True)
    return validate_answer(answer, retrieved)
