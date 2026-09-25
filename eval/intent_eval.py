"""
eval/intent_eval.py — intent + SKU-extraction accuracy. No LLM calls needed
(fast, free) — safe to run on every commit, not just nightly.
"""
from __future__ import annotations
from eval.golden_queries import GOLDEN_QUERIES
from orchestrator.router import classify_intent


def run_intent_eval() -> dict:
    results = []
    for item in GOLDEN_QUERIES:
        intent, sku = classify_intent(item["query"])
        results.append({
            "query": item["query"],
            "expected_intent": item["expected_intent"], "got_intent": intent,
            "expected_sku": item["expected_sku"], "got_sku": sku,
            "intent_ok": intent == item["expected_intent"],
            "sku_ok": sku == item["expected_sku"],
        })

    total = len(results)
    intent_correct = sum(r["intent_ok"] for r in results)
    sku_correct = sum(r["sku_ok"] for r in results)

    return {
        "intent_accuracy": intent_correct / total if total else 0.0,
        "sku_accuracy": sku_correct / total if total else 0.0,
        "total": total,
        "details": results,
    }


if __name__ == "__main__":
    result = run_intent_eval()
    print(f"Intent accuracy: {result['intent_accuracy']:.1%}")
    print(f"SKU accuracy:    {result['sku_accuracy']:.1%}")
    for d in result["details"]:
        if not (d["intent_ok"] and d["sku_ok"]):
            print(f"  ✗ '{d['query']}' -> intent={d['got_intent']} "
                  f"(expected {d['expected_intent']}), sku='{d['got_sku']}' "
                  f"(expected '{d['expected_sku']}')")