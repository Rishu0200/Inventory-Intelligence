"""
Query Chroma collections to retrieve relevant document chunks.
"""
from __future__ import annotations
from knowledge.vector_store.embedder import get_client, get_collection, with_retry


def retrieve(query: str, collection_name: str, k: int = 5,
             where: dict | None = None) -> list[dict]:
    def _do():
        collection = get_collection(get_client(), collection_name)
        n = collection.count()
        if n == 0:
            return []

        params: dict = {"query_texts": [query], "n_results": min(k, n)}
        if where:
            params["where"] = where

        results   = collection.query(**params)
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]

        return [
            {
                "text":     doc,
                "source":   meta.get("source", ""),
                "page":     meta.get("page", 0),
                "doc_type": meta.get("doc_type", ""),
            }
            for doc, meta in zip(docs, metadatas)
        ]

    return with_retry(_do)


def format_context(results: list[dict]) -> str:
    if not results:
        return "No relevant documents found."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Source {i}: {r['source']} — Page {r['page']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)
