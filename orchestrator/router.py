"""
Intent router — classifies a user query into one of 4 agent routes.
Strategy: keyword matching first (instant, free), LLM fallback only when ambiguous.
Also extracts SKU ID if mentioned in the query.
"""
from __future__ import annotations
import re
from config import settings
from orchestrator.llm_factory import get_llm

# ── SKU patterns for Uninox Houseware ────────────────────────────────────────
_SKU_RE = re.compile(
    r"\b(SC|TBW|PBW|PBP|TBP|PNT|CS|GTP|HNG|CRO|TNB|RSH|AAT|CHM|MGC"
    r"|RM-WR|RM-CH|RM-PL|RM-FT|RM-CB)-\d+\b",
    re.IGNORECASE,
)

# ── Keyword → intent mapping ──────────────────────────────────────────────────
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
        "anomaly", "unusual", "spike", "outlier", "alert", "abnormal",
        "strange", "unexpected", "flag", "detect", "problem", "issue",
    ],
}


def classify_intent(query: str) -> tuple[str, str]:
    """
    Classify query into an intent and extract SKU ID.

    Returns:
        (intent, sku_id) where intent is one of:
        "demand", "reorder", "supplier", "anomaly", "general"
    """
    q_lower = query.lower()

    # Extract SKU ID
    sku_match = _SKU_RE.search(query)
    sku_id    = sku_match.group(0).upper() if sku_match else ""

    # Keyword scoring
    scores: dict[str, int] = {intent: 0 for intent in _KEYWORDS}
    for intent, keywords in _KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                scores[intent] += 1

    best_intent = max(scores, key=lambda k: scores[k])
    best_score  = scores[best_intent]

    if best_score > 0:
        return best_intent, sku_id

    # ── LLM fallback for ambiguous queries ───────────────────────────────────
    if settings.use_llm:
        return _llm_classify(query), sku_id

    return "general", sku_id


def _llm_classify(query: str) -> str:
    """Use a small LLM call to resolve ambiguous intent."""
    try:
        llm = get_llm(temperature=0, max_tokens=10)
        # llm = ChatOpenAI(
        #     model=settings.openai_model,
        #     api_key=settings.openai_api_key,
        #     temperature=0,
        #     max_tokens=10,
        #)
        prompt = (
            "Classify this inventory query into exactly one word: "
            "demand / reorder / supplier / anomaly / general\n\n"
            f"Query: {query}\nAnswer:"
        )
        resp = llm.invoke(prompt)
        word = resp.content.strip().lower().split()[0]
        return word if word in _KEYWORDS else "general"
    except Exception:
        return "general"
