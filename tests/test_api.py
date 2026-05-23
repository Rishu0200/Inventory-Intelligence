"""
tests/test_api.py — FastAPI integration tests.
Uses TestClient (no server needed). Mocks heavy ML/LangGraph dependencies.
Run: pytest tests/test_api.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── Patch heavy imports before app loads ─────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with ML dependencies mocked out."""
    with patch("knowledge.feature_store.demand_model.load_model", return_value=None), \
         patch("knowledge.feature_store.anomaly_model.load_anomaly_model", return_value=(None, None)), \
         patch("knowledge.vector_store.embedder.collection_count", return_value=0):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_ping_returns_200(self, client):
        r = client.get("/ping")
        assert r.status_code == 200

    def test_ping_response_schema(self, client):
        r = client.get("/ping")
        body = r.json()
        assert "status"       in body
        assert "demo_mode"    in body
        assert "models_ready" in body
        assert "chroma_docs"  in body
        assert body["status"] == "ok"


# ── Query endpoint ────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    @patch("orchestrator.graph.get_graph")
    def test_post_query_returns_200(self, mock_graph, client):
        mock_graph.return_value.invoke.return_value = {
            "intent":         "demand",
            "sku_id":         "TBP-001",
            "rag_context":    "some context",
            "tool_result":    "Forecast: 320 units",
            "final_response": "Next month demand for TBP-001 is ~320 units.",
        }
        r = client.post("/api/query", json={"question": "forecast for TBP-001"})
        assert r.status_code == 200

    @patch("orchestrator.graph.get_graph")
    def test_post_query_response_schema(self, mock_graph, client):
        mock_graph.return_value.invoke.return_value = {
            "intent":         "reorder",
            "sku_id":         "RSH-001",
            "rag_context":    "",
            "tool_result":    "Stock: 80, ROP: 120 — REORDER",
            "final_response": "RSH-001 needs reordering immediately.",
        }
        r = client.post("/api/query", json={"question": "check RSH-001 stock"})
        body = r.json()
        assert "question"          in body
        assert "intent"            in body
        assert "sku_id"            in body
        assert "answer"            in body
        assert "rag_context_used"  in body

    def test_query_validation_empty_string(self, client):
        r = client.post("/api/query", json={"question": ""})
        assert r.status_code == 422   # Pydantic validation error

    def test_query_validation_too_long(self, client):
        r = client.post("/api/query", json={"question": "x" * 501})
        assert r.status_code == 422


# ── Alerts endpoint ───────────────────────────────────────────────────────────

class TestAlertsEndpoint:
    @patch("api.routes.alerts._reorder_alerts", return_value=[])
    @patch("api.routes.alerts._anomaly_alerts", return_value=[])
    def test_alerts_returns_200(self, mock_anom, mock_reorder, client):
        r = client.get("/api/alerts?refresh=true")
        assert r.status_code == 200

    @patch("api.routes.alerts._reorder_alerts", return_value=[])
    @patch("api.routes.alerts._anomaly_alerts", return_value=[])
    def test_alerts_response_schema(self, mock_anom, mock_reorder, client):
        r = client.get("/api/alerts?refresh=true")
        body = r.json()
        assert "total_alerts"   in body
        assert "reorder_alerts" in body
        assert "anomaly_alerts" in body
        assert "alerts"         in body
        assert isinstance(body["alerts"], list)

    @patch("api.routes.alerts._reorder_alerts", return_value=[])
    @patch("api.routes.alerts._anomaly_alerts", return_value=[])
    def test_alerts_total_matches_lists(self, mock_anom, mock_reorder, client):
        r = client.get("/api/alerts?refresh=true")
        body = r.json()
        assert body["total_alerts"] == body["reorder_alerts"] + body["anomaly_alerts"]


# ── Forecast endpoint ─────────────────────────────────────────────────────────

class TestForecastEndpoint:
    @patch("api.routes.forecast.forecast_sku")
    def test_forecast_valid_sku(self, mock_fc, client):
        mock_fc.return_value = {
            "sku_id":   "TBP-001",
            "forecast": [320.0, 340.0, 310.0],
            "lower":    [260.0, 278.0, 254.0],
            "upper":    [380.0, 402.0, 366.0],
        }
        r = client.get("/api/forecast/TBP-001?horizon=3")
        assert r.status_code == 200

    @patch("api.routes.forecast.forecast_sku")
    def test_forecast_response_schema(self, mock_fc, client):
        mock_fc.return_value = {
            "sku_id":   "SC-001",
            "forecast": [150.0, 165.0],
            "lower":    [120.0, 132.0],
            "upper":    [180.0, 198.0],
        }
        r = client.get("/api/forecast/SC-001?horizon=2")
        body = r.json()
        assert body["sku_id"]    == "SC-001"
        assert body["horizon"]   == 2
        assert len(body["points"]) == 2
        assert "forecast"  in body["points"][0]
        assert "lower_ci"  in body["points"][0]
        assert "upper_ci"  in body["points"][0]

    @patch("api.routes.forecast.forecast_sku")
    def test_forecast_unknown_sku_returns_404(self, mock_fc, client):
        mock_fc.return_value = {"sku_id": "XYZ-999", "forecast": [], "lower": [], "upper": []}
        r = client.get("/api/forecast/XYZ-999")
        assert r.status_code == 404

    def test_forecast_horizon_out_of_range(self, client):
        r = client.get("/api/forecast/TBP-001?horizon=15")
        assert r.status_code == 422
