"""
Intent router — classifies a user query into one of 4 agent routes.
"""
from __future__ import annotations
import re
import string
from config import settings
from orchestrator.llm_factory import get_llm

# FIX #15: added BPO prefix (was missing, matches BPO-001 in demand_history.csv)
_SKU_RE = re.compile(
    r"\b(SC|TBW|PBW|PBP|TBP|PNT|CS|GTP|HNG|CRO|TNB|RSH|AAT|CHM|MGC|BPO"
    r"|RM-WR|RM-CH|RM-PL|RM-FT|RM-CB)-\d+\b",
    re.IGNORECASE,
)

_KEYWORDS: dict[str, list[str]] = {
    "demand": [
        "forecast", "predict", "demand", "how many units", "next month",
        "sales", "projection", "expected", "trend", "consumption",
    ],
    "reorder": [
        "reorder", "reorder point", "rop", "replenish", "stock", "inventory",
        "low stock", "order more", "safety stock", "days of stock",
        "out of stock", "stockout", "when to order", "how much to order",
    ],
    "supplier": [
        "supplier", "vendor", "who supplies", "lead time", "payment terms",
        "catalog", "price", "rate", "moq", "minimum order", "credit",
        "best supplier", "which supplier", "source",
    ],
    "anomaly": [
        "anomaly", "anomalies", "unusual", "spike", "spikes", "outlier",
        "outliers", "alert", "abnormal", "strange", "unexpected", "flag",
        "detect", "problem", "problems", "issue", "issues",
    ],
}

_TIE_BREAK_PRIORITY = ["anomaly", "supplier", "reorder", "demand"]


def classify_intent(query: str) -> tuple[str, str]:
    q_lower = query.lower()

    sku_match = _SKU_RE.search(query)
    sku_id    = sku_match.group(0).upper() if sku_match else ""

    scores: dict[str, int] = {intent: 0 for intent in _KEYWORDS}
    for intent, keywords in _KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                scores[intent] += 1

    best_intent = max(scores, key=lambda k: (scores[k], -_TIE_BREAK_PRIORITY.index(k)))
    best_score  = scores[best_intent]

    if best_score > 0:
        return best_intent, sku_id

    if settings.use_llm:
        return _llm_classify(query), sku_id

    return "general", sku_id


def _llm_classify(query: str) -> str:
    """FIX #15: strip punctuation from the LLM's response before matching."""
    try:
        llm = get_llm(temperature=0, max_tokens=150)
        prompt = (
            "Classify this inventory query into exactly one word: "
            "demand / reorder / supplier / anomaly / general\n\n"
            f"Query: {query}\nAnswer:"
        )
        resp = llm.invoke(prompt)
        raw_word = resp.content.strip().lower().split()[0]
        word = raw_word.strip(string.punctuation)
        return word if word in _KEYWORDS else "general"
    except Exception:
        return "general"
