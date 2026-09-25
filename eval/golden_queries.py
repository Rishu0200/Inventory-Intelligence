"""
eval/golden_queries.py — hand-labeled test set for the eval harness.

Each entry:
  query            - natural language question
  expected_intent  - what the router SHOULD assign
  expected_sku     - the SKU it should extract, "" if none
  doc_type         - which RAG collection this query is really about
  expected_keyword - a keyword a genuinely relevant retrieved chunk should
                      contain (a lightweight proxy for "retrieval found
                      something useful" — extend with real chunk-id ground
                      truth later if you want a stricter check)

Extend this over time — real logged queries (once QueryLog is wired up) are
a great source of new cases, especially ones users gave thumbs-down on.
"""

GOLDEN_QUERIES = [
    {"query": "Forecast demand for TBP-001 next 3 months", "expected_intent": "demand", "expected_sku": "TBP-001", "doc_type": "PO", "expected_keyword": "TBP-001"},
    {"query": "How many units of RSH-001 will we need next month?", "expected_intent": "demand", "expected_sku": "RSH-001", "doc_type": "PO", "expected_keyword": "RSH-001"},
    {"query": "Which SKUs need reordering right now?", "expected_intent": "reorder", "expected_sku": "", "doc_type": "PO", "expected_keyword": "purchase"},
    {"query": "What is the reorder point for CHM-001?", "expected_intent": "reorder", "expected_sku": "CHM-001", "doc_type": "PO", "expected_keyword": "CHM-001"},
    {"query": "Who is the best supplier for RSH-001?", "expected_intent": "supplier", "expected_sku": "RSH-001", "doc_type": "catalog", "expected_keyword": "Lakshmi"},
    {"query": "What are the payment terms for Mehta Wire Industries?", "expected_intent": "supplier", "expected_sku": "", "doc_type": "catalog", "expected_keyword": "Mehta"},
    {"query": "What's the lead time for MGC-001's supplier?", "expected_intent": "supplier", "expected_sku": "MGC-001", "doc_type": "catalog", "expected_keyword": "MGC-001"},
    {"query": "Any unusual demand anomalies this month?", "expected_intent": "anomaly", "expected_sku": "", "doc_type": "all", "expected_keyword": ""},
    {"query": "Detect anomalies for BPO-001", "expected_intent": "anomaly", "expected_sku": "BPO-001", "doc_type": "all", "expected_keyword": ""},
    {"query": "Is there a demand spike for SC-001?", "expected_intent": "anomaly", "expected_sku": "SC-001", "doc_type": "all", "expected_keyword": ""},
    {"query": "hello there", "expected_intent": "general", "expected_sku": "", "doc_type": "all", "expected_keyword": ""},
    {"query": "What can you help me with?", "expected_intent": "general", "expected_sku": "", "doc_type": "all", "expected_keyword": ""},
]