"""
Split extracted page text into overlapping chunks for ChromaDB ingestion.
Uses LangChain's RecursiveCharacterTextSplitter (400 tokens, 50-token overlap).
"""
from __future__ import annotations
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 400 tokens ≈ 1,600 chars for typical English text
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1_600,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],
)


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Take a list of page dicts (from pdf_parser) and return a flat list
    of chunk dicts ready for ChromaDB upsert.

    Each chunk dict:
        {
            "id":       "<source>_p<page>_c<chunk_idx>",
            "text":     "...",
            "source":   "PO_0001.pdf",
            "page":     0,
            "doc_type": "PO" | "catalog" | "unknown",
        }
    """
    chunks = []
    for page in pages:
        splits = _SPLITTER.split_text(page["text"])
        doc_type = _infer_doc_type(page["source"])
        for ci, chunk_text in enumerate(splits):
            chunk_id = f"{page['source']}_p{page['page']}_c{ci}"
            chunks.append({
                "id":       chunk_id,
                "text":     chunk_text,
                "source":   page["source"],
                "page":     page["page"],
                "doc_type": doc_type,
            })
    return chunks


def _infer_doc_type(filename: str) -> str:
    fn = filename.lower()
    if fn.startswith("po_") or "purchase" in fn:
        return "PO"
    if "catalog" in fn or "supplier" in fn:
        return "catalog"
    return "unknown"
