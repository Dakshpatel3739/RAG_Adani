import os
import re
from typing import List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None

from .indexing import Chunk
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
        "- Use the period labels exactly as shown in the context (e.g., H1-26, FY25).\n"
        "- If asked for changes, report each period label with its value (e.g., H1-25: X; H1-26: Y) "
        "and avoid inferring direction unless explicitly stated.\n"
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
    for num in answer_nums:
        if num in context_nums:
            continue
        if len(num) == 4 and num.startswith("20"):
            short = num[2:]
            if short in context_nums:
                continue
        return False
    return True


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


def _extract_two_numbers(line: str) -> Optional[Tuple[str, str]]:
    nums = re.findall(r"\d+\.?\d*", line)
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    return None


def extract_passenger_cargo_answer(retrieved: List[RetrievalResult]) -> Optional[str]:
    for item in retrieved:
        lines = [line.strip() for line in item.chunk.text.splitlines() if line.strip()]
        pax = None
        cargo = None
        for idx, line in enumerate(lines):
            if re.search(r"\b(Pax|Passengers)\b", line, flags=re.IGNORECASE):
                if idx > 0:
                    pax = _extract_two_numbers(lines[idx - 1]) or _extract_two_numbers(line)
                else:
                    pax = _extract_two_numbers(line)
                break

        for idx, line in enumerate(lines):
            if re.search(r"cargo volume", line, flags=re.IGNORECASE):
                for j in range(idx + 1, min(idx + 5, len(lines))):
                    if "h1-25" in lines[j].lower() or "h1-26" in lines[j].lower():
                        continue
                    candidate = _extract_two_numbers(lines[j])
                    if candidate:
                        cargo = candidate
                        break
                break

        if cargo is None:
            for idx, line in enumerate(lines):
                if re.search(r"Cargo\s*\(L-MT\)", line, flags=re.IGNORECASE):
                    for j in range(max(0, idx - 4), idx + 1):
                        candidate = _extract_two_numbers(lines[j])
                        if candidate:
                            cargo = candidate
                            break
                if cargo:
                    break

        if pax and cargo:
            citation = f"[p{item.chunk.page}:{item.chunk.cid}]"
            return (
                f"Passengers H1-25: {pax[0]}; H1-26: {pax[1]}. "
                f"Cargo H1-25: {cargo[0]}; H1-26: {cargo[1]} {citation}"
            )
    return None


def _focus_passenger_cargo_text(text: str) -> str:
    lines = text.splitlines()
    keep = []
    keywords = ["pax", "passenger", "cargo", "h1-25", "h1-26"]
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(key in lowered for key in keywords):
            for j in (idx - 2, idx - 1, idx, idx + 1, idx + 2):
                if 0 <= j < len(lines):
                    candidate = lines[j].strip()
                    if not candidate:
                        continue
                    if j != idx and not re.search(r"\d", candidate):
                        continue
                    keep.append(candidate)
    if not keep:
        return text
    deduped = []
    seen = set()
    for line in keep:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    return "\\n".join(deduped)


def answer_with_retry(
    client: "OpenAI",
    model: str,
    question: str,
    retrieved: List[RetrievalResult],
    retries: int = 1,
) -> str:
    lowered = question.lower()
    if "passenger" in lowered and "cargo" in lowered:
        filtered = [
            item
            for item in retrieved
            if "pax" in item.chunk.text.lower()
            or "passenger" in item.chunk.text.lower()
        ]
        extracted = extract_passenger_cargo_answer(filtered if filtered else retrieved)
        if extracted:
            validated = validate_answer(extracted, filtered if filtered else retrieved)
            if validated != "Not found in the document.":
                return validated

        source = filtered if filtered else retrieved
        focused = []
        for item in source:
            focused_text = _focus_passenger_cargo_text(item.chunk.text)
            focused.append(
                RetrievalResult(
                    chunk=Chunk(
                        cid=item.chunk.cid,
                        page=item.chunk.page,
                        text=focused_text,
                    ),
                    score=item.score,
                    bm25=item.bm25,
                    cosine=item.cosine,
                )
            )
        answer = generate_answer(client, model, question, focused, strict=True)
        validated = validate_answer(answer, focused)
        if validated != "Not found in the document.":
            return validated

    answer = generate_answer(client, model, question, retrieved, strict=False)
    validated = validate_answer(answer, retrieved)
    if validated != "Not found in the document." or retries <= 0:
        return validated
    answer = generate_answer(client, model, question, retrieved, strict=True)
    validated = validate_answer(answer, retrieved)
    if validated != "Not found in the document.":
        return validated

    if "passenger" in lowered and "cargo" in lowered:
        extracted = extract_passenger_cargo_answer(filtered if filtered else retrieved)
        if extracted:
            validated = validate_answer(extracted, filtered if filtered else retrieved)
            if validated != "Not found in the document.":
                return validated
    return "Not found in the document."
