"""
eval/run_eval.py — full eval harness, logs results to the database.
Usage: python -m eval.run_eval
Run locally anytime, or in the nightly GitHub Actions job after training.
"""
from __future__ import annotations
import subprocess
from datetime import datetime

from config import settings
from db.session import get_session
from db.models import EvalRun
from eval.golden_queries import GOLDEN_QUERIES
from eval.intent_eval import run_intent_eval
from eval.retrieval_eval import run_retrieval_eval
from eval.groundedness_judge import run_groundedness_eval


def _get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    print("=" * 60, "\n  Eval Harness\n" + "=" * 60)

    print("\n[1/3] Intent classification accuracy...")
    intent_result = run_intent_eval()
    print(f"  Intent accuracy: {intent_result['intent_accuracy']:.1%}  SKU accuracy: {intent_result['sku_accuracy']:.1%}")

    print("\n[2/3] Retrieval hit-rate...")
    retrieval_result = run_retrieval_eval()
    print(f"  Hit-rate: {retrieval_result['hit_rate']:.1%} ({retrieval_result['hits']}/{retrieval_result['total']})")

    groundedness_pass_rate = None
    if settings.use_llm:
        print("\n[3/3] Groundedness (LLM-as-judge)...")
        from orchestrator.graph import get_graph
        graph = get_graph()
        pairs = []
        for item in GOLDEN_QUERIES:
            state = graph.invoke({"query": item["query"], "intent": "", "sku_id": "",
                                   "rag_context": "", "tool_result": "", "final_response": ""})
            combined_context = (
                f"Analysis result:\n{state.get('tool_result', '')}\n\n"
                f"Supporting document context:\n{state.get('rag_context', '')}"
            )
            pairs.append({"question": item["query"], "context": combined_context,
                          "answer": state.get("final_response", "")})
        g = run_groundedness_eval(pairs)
        groundedness_pass_rate = g["groundedness_pass_rate"]
        print(f"  Groundedness pass rate: {groundedness_pass_rate:.1%}")
        for d in g["details"]:
            if d["grounded"] == "no":
                print(f"    ✗ '{d['question']}'")
                print(f"      context: {d.get('context', '')[:120]!r}")
                print(f"      answer:  {d.get('answer', '')[:150]!r}")
    else:
        print("\n[3/3] Skipping groundedness — settings.use_llm is False "
              "(needs DEMO_MODE=false + GROQ_API_KEY).")

    with get_session() as session:
        session.add(EvalRun(
            run_at=datetime.utcnow(), git_commit=_get_git_commit(),
            intent_accuracy=intent_result["intent_accuracy"],
            retrieval_hit_rate=retrieval_result["hit_rate"],
            groundedness_pass_rate=groundedness_pass_rate,
            n_queries=len(GOLDEN_QUERIES),
        ))

    print("\n✅ Eval run logged to database.")


if __name__ == "__main__":
    main()