import logging
import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve


def download_pdf(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "document.pdf"
    target = cache_dir / filename
    urlretrieve(url, target)  # nosec - controlled input from user
    return target


def resolve_pdf_path(pdf_arg: str, cache_dir: Path) -> Path:
    path = Path(pdf_arg)
    if path.is_file():
        return path
    if pdf_arg.startswith("http://") or pdf_arg.startswith("https://"):
        return download_pdf(pdf_arg, cache_dir)
    raise SystemExit(f"PDF not found: {pdf_arg}")


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    errors = []
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append((idx + 1, text))
        return pages
    except Exception as exc:
        errors.append(f"pdfplumber: {exc}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((idx + 1, text))
        return pages
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    joined = "; ".join(errors)
    raise SystemExit(
        "Failed to extract text from PDF. Install pdfplumber or pypdf. Errors: "
        + joined
    )


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()
