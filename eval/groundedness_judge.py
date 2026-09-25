"""
eval/groundedness_judge.py — LLM-as-judge, groundedness + directness only.
Fixed rubric, never tuned alongside whatever it's judging, so scores stay
comparable run-to-run. Narrow yes/no/partial questions, not an open-ended
1-10 score — LLM judges are far more reliable on narrow questions.
Offline only — never runs inside the deployed API.
"""
from __future__ import annotations
import json
from orchestrator.llm_factory import get_llm

_JUDGE_PROMPT = """You are evaluating an AI assistant's answer to a factual question about inventory management.

Question: {question}

Retrieved context the assistant had access to:
{context}

Assistant's answer:
{answer}

Judge this answer on two dimensions. Respond with ONLY a JSON object, nothing else:
{{
  "grounded": "yes" | "partial" | "no",
  "direct": "yes" | "no"
}}
"""


def judge_answer(question: str, context: str, answer: str) -> dict:
    llm = get_llm(temperature=0, max_tokens=400)
    prompt = _JUDGE_PROMPT.format(question=question, context=context[:1500], answer=answer[:1000])
    try:
        text = llm.invoke(prompt).content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return {"grounded": parsed.get("grounded", "no"), "direct": parsed.get("direct", "no")}
    except Exception as e:
        return {"grounded": "no", "direct": "no", "error": str(e)}


def run_groundedness_eval(query_answer_pairs: list[dict]) -> dict:
    """query_answer_pairs: [{"question":..., "context":..., "answer":...}, ...]"""
    results = [{**pair, **judge_answer(pair["question"], pair.get("context", ""), pair["answer"])}
               for pair in query_answer_pairs]
    total = len(results)
    grounded = sum(r["grounded"] in ("yes", "partial") for r in results)
    direct = sum(r["direct"] == "yes" for r in results)
    return {
        "groundedness_pass_rate": grounded / total if total else 0.0,
        "directness_pass_rate": direct / total if total else 0.0,
        "total": total, "details": results,
    }