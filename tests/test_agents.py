"""
tests/test_agents.py — Unit tests for all 4 LangGraph agent nodes.
Uses unittest.mock to avoid needing real CSV data or trained models.
Run: pytest tests/test_agents.py -v
"""
import pytest
from unittest.mock import patch, MagicMock


# ── Shared mock state ─────────────────────────────────────────────────────────

def base_state(sku_id="TBP-001", query="test query") -> dict:
    return {
        "query":          query,
        "intent":         "",
        "sku_id":         sku_id,
        "rag_context":    "",
        "tool_result":    "",
        "final_response": "",
    }


# ── Router tests ──────────────────────────────────────────────────────────────

class TestRouter:
    def test_demand_keywords(self):
        from orchestrator.router import classify_intent
        intent, _ = classify_intent("forecast demand for TBP-001 next month")
        assert intent == "demand"

    def test_reorder_keywords(self):
        from orchestrator.router import classify_intent
        intent, _ = classify_intent("which SKUs need reordering?")
        assert intent == "reorder"

    def test_supplier_keywords(self):
        from orchestrator.router import classify_intent
        intent, _ = classify_intent("who is the best supplier for RSH-001?")
        assert intent == "supplier"

    def test_anomaly_keywords(self):
        from orchestrator.router import classify_intent
        intent, _ = classify_intent("any unusual demand anomalies this month?")
        assert intent == "anomaly"

    def test_sku_extraction(self):
        from orchestrator.router import classify_intent
        _, sku = classify_intent("forecast for TBP-001 please")
        assert sku == "TBP-001"

    def test_no_sku_returns_empty(self):
        from orchestrator.router import classify_intent
        _, sku = classify_intent("which items need reordering?")
        assert sku == ""

    def test_general_fallback(self):
        from orchestrator.router import classify_intent
        intent, _ = classify_intent("hello there")
        assert intent == "general"


# ── Demand Agent tests ────────────────────────────────────────────────────────

class TestDemandAgent:
    @patch("agents.demand_agent.retrieve_docs")
    @patch("agents.demand_agent.get_forecast")
    def test_returns_tool_result(self, mock_forecast, mock_docs):
        mock_forecast.invoke.return_value = "Forecast for TBP-001: +1: 320 units"
        mock_docs.invoke.return_value     = "Relevant PO context."

        from agents.demand_agent import demand_agent_node
        result = demand_agent_node(base_state("TBP-001", "forecast next 3 months"))

        assert "tool_result" in result
        assert "rag_context" in result
        mock_forecast.invoke.assert_called_once()

    @patch("agents.demand_agent.retrieve_docs")
    @patch("agents.demand_agent.get_forecast")
    def test_handles_forecast_error_gracefully(self, mock_forecast, mock_docs):
        mock_forecast.invoke.side_effect = Exception("Model not loaded")
        mock_docs.invoke.return_value    = ""

        from agents.demand_agent import demand_agent_node
        result = demand_agent_node(base_state("SC-001"))

        assert "tool_result" in result
        assert "unavailable" in result["tool_result"].lower() or "failed" in result["tool_result"].lower()


# ── Reorder Agent tests ───────────────────────────────────────────────────────

class TestReorderAgent:
    @patch("agents.reorder_agent.retrieve_docs")
    @patch("agents.reorder_agent.compute_rop")
    @patch("agents.reorder_agent.check_stock")
    def test_single_sku_analysis(self, mock_stock, mock_rop, mock_docs):
        mock_stock.invoke.return_value = "Stock: TBP-001 | Available: 120 | ROP: 180 | ⚠️ REORDER"
        mock_rop.invoke.return_value   = "ROP: 180 | Current: 120 | ⚠️ REORDER NOW"
        mock_docs.invoke.return_value  = ""

        from agents.reorder_agent import reorder_agent_node
        result = reorder_agent_node(base_state("TBP-001", "check stock for TBP-001"))

        assert "tool_result" in result
        assert "TBP-001" in result["tool_result"] or "REORDER" in result["tool_result"]

    @patch("agents.reorder_agent.retrieve_docs")
    @patch("agents.reorder_agent.pd")
    def test_all_skus_scan_no_sku(self, mock_pd, mock_docs):
        mock_docs.invoke.return_value = ""
        # No SKU provided — should scan all
        state = base_state(sku_id="", query="which items need reordering?")

        from agents.reorder_agent import reorder_agent_node
        result = reorder_agent_node(state)

        assert "tool_result" in result


# ── Supplier Agent tests ──────────────────────────────────────────────────────

class TestSupplierAgent:
    @patch("agents.supplier_agent.retrieve_docs")
    @patch("agents.supplier_agent.get_supplier_info")
    @patch("agents.supplier_agent._find_suppliers_for_sku")
    def test_returns_supplier_info(self, mock_find, mock_info, mock_docs):
        mock_find.return_value         = ["SUP-001"]
        mock_info.invoke.return_value  = "Supplier: Mehta Wire | Lead: 36 days | MOQ: 116"
        mock_docs.invoke.return_value  = "Catalog context."

        from agents.supplier_agent import supplier_agent_node
        result = supplier_agent_node(base_state("TBP-001", "who supplies TBP-001?"))

        assert "tool_result" in result
        assert "Mehta" in result["tool_result"] or "SUP-001" in result["tool_result"]

    @patch("agents.supplier_agent.retrieve_docs")
    @patch("agents.supplier_agent.get_supplier_info")
    @patch("agents.supplier_agent._find_suppliers_for_sku")
    def test_no_supplier_returns_message(self, mock_find, mock_info, mock_docs):
        mock_find.return_value        = []
        mock_docs.invoke.return_value = ""

        from agents.supplier_agent import supplier_agent_node
        result = supplier_agent_node(base_state("UNKNOWN-001"))

        assert "No supplier" in result["tool_result"]


# ── Anomaly Agent tests ───────────────────────────────────────────────────────

class TestAnomalyAgent:
    @patch("agents.anomaly_agent.retrieve_docs")
    @patch("agents.anomaly_agent.detect_sku_anomaly")
    def test_with_sku(self, mock_detect, mock_docs):
        mock_detect.invoke.return_value = "Anomalies for SC-001: 2023-03: 890 units (3M avg: 420)"
        mock_docs.invoke.return_value   = ""

        from agents.anomaly_agent import anomaly_agent_node
        result = anomaly_agent_node(base_state("SC-001", "any anomalies for SC-001?"))

        assert "tool_result" in result
        mock_detect.invoke.assert_called_once()

    @patch("agents.anomaly_agent.retrieve_docs")
    @patch("agents.anomaly_agent.detect_sku_anomaly")
    def test_tool_error_handled(self, mock_detect, mock_docs):
        mock_detect.invoke.side_effect = Exception("Model not ready")
        mock_docs.invoke.return_value  = ""

        from agents.anomaly_agent import anomaly_agent_node
        result = anomaly_agent_node(base_state("SC-001"))

        assert "tool_result" in result
        assert "failed" in result["tool_result"].lower()
