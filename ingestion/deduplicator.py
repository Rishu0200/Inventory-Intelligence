"""
Near-duplicate chunk detection using MinHash + LSH (datasketch).
Chunks with Jaccard similarity > threshold are treated as duplicates
and filtered before ChromaDB insertion.
"""
from __future__ import annotations
from datasketch import MinHash, MinHashLSH


def deduplicate(chunks: list[dict],
                threshold: float = 0.85,
                num_perm: int = 128) -> list[dict]:
    """
    Remove near-duplicate chunks from the list.

    Args:
        chunks:    List of chunk dicts (must have "text" and "id" keys).
        threshold: Jaccard similarity above which two chunks are duplicates.
        num_perm:  Number of permutations for MinHash accuracy.

    Returns:
        Deduplicated list of chunk dicts.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique: list[dict] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        mh = _make_minhash(chunk["text"], num_perm)
        chunk_id = chunk["id"]

        if chunk_id in seen_ids:
            continue

        try:
            result = lsh.query(mh)
        except Exception:
            result = []

        if result:
            # A near-duplicate already in the index — skip
            continue

        lsh.insert(chunk_id, mh)
        seen_ids.add(chunk_id)
        unique.append(chunk)

    removed = len(chunks) - len(unique)
    if removed:
        print(f"[deduplicator] Removed {removed} near-duplicate chunks "
              f"({len(unique)} unique kept).")
    return unique


def _make_minhash(text: str, num_perm: int) -> MinHash:
    """Create a MinHash from word-level shingles (4-gram)."""
    mh = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(max(1, len(words) - 3)):
        shingle = " ".join(words[i:i+4])
        mh.update(shingle.encode("utf8"))
    return mh
