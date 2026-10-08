"""
Embed document chunks using sentence-transformers and persist to ChromaDB.
Model: all-MiniLM-L6-v2 (22MB, no API key needed).
"""
from __future__ import annotations
from time import time
import chromadb
from chromadb.utils import embedding_functions
from config import Paths, settings

_client = None
_embedding_fn = None
_collections: dict = {}

def _reset_cache() -> None:
    """Drop cached connection objects so the next attempt reconnects cleanly."""
    global _client
    _client = None
    _collections.clear()

def get_client():
    global _client
    if _client is None:
        if settings.chroma_cloud_api_key:
            _client = chromadb.CloudClient(
                api_key=settings.chroma_cloud_api_key,
                tenant=settings.chroma_cloud_tenant,
                database=settings.chroma_cloud_database,
            )
        else:
            _client = chromadb.PersistentClient(path=str(Paths.CHROMA_STORE))
    return _client


def get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.ONNXMiniLM_L6_V2()
    return _embedding_fn


def get_collection(client, name: str):
    if name not in _collections:
        _collections[name] = client.get_or_create_collection(
            name=name, embedding_function=get_embedding_fn()
        )
    return _collections[name]

def with_retry(fn, attempts: int = 3, base_delay: float = 1.0):
    """
    Run fn(), retrying on any failure with exponential backoff (1s, 2s).
    Resets cached connections between attempts. A permanent error (e.g. a
    wrong tenant) just costs ~3s extra before raising the same error.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            _reset_cache()
            if attempt < attempts - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[chroma] {type(e).__name__}; retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{attempts})")
                time.sleep(delay)
    raise last_error

def embed_chunks(chunks: list[dict], collection_name: str, batch_size: int = 64) -> int:
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        ids       = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [{"source": c["source"], "page": c["page"],
                      "doc_type": c["doc_type"]} for c in batch]

        def _upsert():
            get_collection(get_client(), collection_name).upsert(
                ids=ids, documents=documents, metadatas=metadatas
            )

        with_retry(_upsert)
        total += len(batch)

    print(f"[embedder] Upserted {total} chunks → collection '{collection_name}'")
    return total


def collection_count(collection_name: str) -> int:
    return with_retry(lambda: get_collection(get_client(), collection_name).count())
