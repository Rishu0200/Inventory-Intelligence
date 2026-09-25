"""
eval/retrieval_eval.py — retrieval hit-rate@k. Checks whether a keyword that
SHOULD appear in a relevant chunk actually shows up in what retrieve_docs
returns. Catches gross retrieval failures (like the doc_type="all" starvation
bug we already fixed) — not a substitute for exact chunk-id ground truth,
but cheap and fast.
"""
from __future__ import annotations
from eval.golden_queries import GOLDEN_QUERIES
from orchestrator.tools import retrieve_docs


def run_retrieval_eval(k: int = 5) -> dict:
    results = []
    for item in GOLDEN_QUERIES:
        if not item.get("expected_keyword"):
            continue

        context = retrieve_docs.invoke({
            "query": item["query"], "doc_type": item["doc_type"], "k": k,
        })
        hit = item["expected_keyword"].lower() in context.lower()
        results.append({"query": item["query"], "hit": hit})

    total = len(results)
    hits = sum(r["hit"] for r in results)
    return {"hit_rate": hits / total if total else 0.0, "total": total, "hits": hits, "details": results}


if __name__ == "__main__":
    result = run_retrieval_eval()
    print(f"Retrieval hit-rate: {result['hit_rate']:.1%} ({result['hits']}/{result['total']})")
    for d in result["details"]:
        print(f"  {'✓' if d['hit'] else '✗'} {d['query']}")