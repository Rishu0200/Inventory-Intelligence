"""
Query ChromaDB collections to retrieve relevant document chunks.
Supports optional metadata filtering (e.g. doc_type="catalog").
"""
from __future__ import annotations
from knowledge.vector_store.embedder import get_client, get_collection
from config import settings


def retrieve(query: str,
             collection_name: str,
             k: int = 5,
             where: dict | None = None) -> list[dict]:
    """
    Semantic similarity search over a ChromaDB collection.

    Args:
        query:           Natural language query string.
        collection_name: Which ChromaDB collection to search.
        k:               Number of top results to return.
        where:           Optional ChromaDB metadata filter dict.
                         e.g. {"doc_type": "catalog"} or {"source": "PO_0001.pdf"}

    Returns:
        List of result dicts:
            {"text": "...", "source": "...", "page": 0, "doc_type": "..."}
    """
    client     = get_client()
    collection = get_collection(client, collection_name)

    if collection.count() == 0:
        return []

    query_params: dict = {"query_texts": [query], "n_results": min(k, collection.count())}
    if where:
        query_params["where"] = where

    results = collection.query(**query_params)

    docs      = results["documents"][0]      # list of strings
    metadatas = results["metadatas"][0]      # list of dicts

    return [
        {
            "text":     doc,
            "source":   meta.get("source", ""),
            "page":     meta.get("page", 0),
            "doc_type": meta.get("doc_type", ""),
        }
        for doc, meta in zip(docs, metadatas)
    ]


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a clean context string for the LLM."""
    if not results:
        return "No relevant documents found."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Source {i}: {r['source']} — Page {r['page']}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)
