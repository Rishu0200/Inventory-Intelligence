"""
scripts/ingest_docs.py — PDF ingestion pipeline.
parse → chunk → deduplicate → embed → ChromaDB
Usage: python scripts/ingest_docs.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Paths, settings
from ingestion.pdf_parser  import parse_directory
from ingestion.chunker     import chunk_pages
from ingestion.deduplicator import deduplicate
from knowledge.vector_store.embedder import embed_chunks, collection_count


def ingest_collection(pdf_dir: Path, collection_name: str, label: str):
    pdf_count = len(list(pdf_dir.glob("*.pdf")))
    if pdf_count == 0:
        print(f"  ⚠  No PDFs in {pdf_dir} — run generate scripts first.")
        return

    print(f"\n  [{label}] {pdf_count} PDFs → collection '{collection_name}'")

    # 1. Parse
    print("    Parsing PDFs...")
    pages = list(parse_directory(pdf_dir))
    print(f"    Extracted {len(pages)} pages.")

    # 2. Chunk
    print("    Chunking text...")
    chunks = chunk_pages(pages)
    print(f"    Created {len(chunks)} chunks.")

    # 3. Deduplicate
    print("    Deduplicating...")
    unique_chunks = deduplicate(chunks, threshold=0.85)

    # 4. Embed + upsert
    print("    Embedding and storing in ChromaDB...")
    n = embed_chunks(unique_chunks, collection_name=collection_name)
    print(f"    ✓ {n} chunks stored. Total in collection: {collection_count(collection_name)}")


def main():
    print("\n📄  Document Ingestion Pipeline — Uninox Houseware")
    print(f"    ChromaDB path: {Paths.CHROMA_STORE}")
    Paths.CHROMA_STORE.mkdir(parents=True, exist_ok=True)

    ingest_collection(
        pdf_dir=Paths.PO_PDFS,
        collection_name=settings.chroma_collection_pos,
        label="Purchase Orders",
    )
    ingest_collection(
        pdf_dir=Paths.CATALOG_PDFS,
        collection_name=settings.chroma_collection_catalogs,
        label="Supplier Catalogs",
    )

    print("\n✅  Ingestion complete.")
    print("    Verify: python -c \"from knowledge.vector_store.embedder import collection_count; "
          f"print(collection_count('{settings.chroma_collection_pos}'))\"")


if __name__ == "__main__":
    main()
