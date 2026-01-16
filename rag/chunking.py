from typing import List


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        window = text[start:end]
        boundary = window.rfind("\n")
        if boundary < int(max_chars * 0.6):
            boundary = window.rfind(". ")
        if boundary < int(max_chars * 0.6):
            boundary = len(window)
        chunk = window[:boundary].strip()
        if chunk:
            chunks.append(chunk)
        start = start + boundary
        if start < length:
            start = max(0, start - overlap)
            if boundary == 0:
                start = end
    return chunks
