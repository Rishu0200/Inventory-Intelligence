"""
scripts/setup.py — One-click setup.
Runs: generate PDFs → ingest docs → train models
Usage: python scripts/setup.py
"""
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def step(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def main():
    start = time.time()
    print("\n🚀  Inventory Intelligence — Full Setup")
    print(f"    Python : {sys.version.split()[0]}")

    # ── Step 1: Generate synthetic PDFs ──────────────────────────────────────
    step("Step 1/3 — Generating PDF documents")
    try:
        from data_generation.generate_pos import main as gen_pos
        from data_generation.generate_supplier_docs import main as gen_catalogs
        gen_pos(count=200)
        gen_catalogs()
        print("✓ PDFs generated.")
    except ImportError as e:
        print(f"⚠  Skipping PDF generation (missing dependency: {e})")
        print("   Install: pip install reportlab faker")

    # ── Step 2: Ingest documents into ChromaDB ────────────────────────────────
    step("Step 2/3 — Ingesting documents into ChromaDB")
    try:
        from scripts.ingest_docs import main as ingest
        ingest()
    except Exception as e:
        print(f"⚠  Ingestion failed: {e}")
        print("   Install: pip install pymupdf chromadb sentence-transformers datasketch")

    # ── Step 3: Train ML models ────────────────────────────────────────────────
    step("Step 3/3 — Training ML models")
    try:
        from scripts.train_models import main as train
        train()
    except Exception as e:
        print(f"⚠  Training failed: {e}")
        print("   Install: pip install xgboost statsmodels scikit-learn mlflow")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ✅ Setup complete in {elapsed:.0f}s")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Copy .env.example → .env  (add OPENAI_API_KEY for LLM mode)")
    print("  2. Start API:      uvicorn api.main:app --reload")
    print("  3. Start frontend: streamlit run frontend/app.py")
    print("  4. Open API docs:  http://localhost:8000/docs\n")


if __name__ == "__main__":
    main()
