"""
Embed document chunks using sentence-transformers and persist to ChromaDB.
Model: all-MiniLM-L6-v2 (22MB, no API key needed).
"""
from __future__ import annotations
import chromadb
from chromadb.utils import embedding_functions

from config import Paths, settings


def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client backed by local disk."""
    return chromadb.PersistentClient(path=str(Paths.CHROMA_STORE))


def get_embedding_fn() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def get_collection(client: chromadb.PersistentClient,
                   name: str) -> chromadb.Collection:
    emb_fn = get_embedding_fn()
    return client.get_or_create_collection(name=name, embedding_function=emb_fn)


def embed_chunks(chunks: list[dict],
                 collection_name: str,
                 batch_size: int = 64) -> int:
    """
    Embed and upsert chunks into a ChromaDB collection.

    Args:
        chunks:          List of chunk dicts with keys: id, text, source, page, doc_type.
        collection_name: ChromaDB collection to upsert into.
        batch_size:      Number of chunks to embed per batch.

    Returns:
        Total number of chunks upserted.
    """
    client     = get_client()
    collection = get_collection(client, collection_name)

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        ids        = [c["id"]   for c in batch]
        documents  = [c["text"] for c in batch]
        metadatas  = [{"source": c["source"], "page": c["page"],
                       "doc_type": c["doc_type"]} for c in batch]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        total += len(batch)

    print(f"[embedder] Upserted {total} chunks → collection '{collection_name}'")
    return total


def collection_count(collection_name: str) -> int:
    client     = get_client()
    collection = get_collection(client, collection_name)
    return collection.count()
