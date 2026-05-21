"""
PDF text extraction using PyMuPDF (fitz).
Returns a list of page-level dicts for downstream chunking.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Generator
import fitz  # PyMuPDF


def parse_pdf(filepath: str | Path) -> list[dict]:
    """
    Extract text from a PDF, one dict per page.

    Returns:
        [{"source": filename, "page": 0, "text": "..."},  ...]
    """
    path = Path(filepath)
    docs = []
    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf):
            text = page.get_text("text").strip()
            text = _clean(text)
            if text:
                docs.append({
                    "source":   path.name,
                    "filepath": str(path),
                    "page":     page_num,
                    "text":     text,
                })
    return docs


def parse_directory(directory: str | Path, extension: str = ".pdf") -> Generator[dict, None, None]:
    """
    Walk a directory and yield parsed pages from all PDFs.
    Yields one dict per page.
    """
    directory = Path(directory)
    pdf_files = sorted(directory.glob(f"*{extension}"))
    if not pdf_files:
        raise FileNotFoundError(f"No {extension} files found in {directory}")

    for pdf_path in pdf_files:
        try:
            pages = parse_pdf(pdf_path)
            yield from pages
        except Exception as exc:
            print(f"[pdf_parser] Warning: could not parse {pdf_path.name}: {exc}")


def _clean(text: str) -> str:
    """Basic text cleaning: collapse multiple newlines and spaces."""
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
