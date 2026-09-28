import pytest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from auth.dependencies import get_current_user
from config import settings
from db.session import get_session_dependency

FAKE_USER = SimpleNamespace(id=1, email="test@example.com", role="viewer", is_active=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app_instance():
    from api.main import app
    return app


@pytest.fixture
def client(app_instance):
    """Authenticated client — get_current_user is overridden with FAKE_USER."""
    app_instance.dependency_overrides[get_current_user] = lambda: FAKE_USER
    yield TestClient(app_instance)
    app_instance.dependency_overrides.clear()


@pytest.fixture
def anon_client(app_instance):
    """Unauthenticated client — no overrides, real auth dependency runs."""
    app_instance.dependency_overrides.clear()
    yield TestClient(app_instance)
    app_instance.dependency_overrides.clear()


@pytest.fixture
def no_cache():
    """Stop alerts tests from reading/writing the real Redis cache."""
    with patch("api.routes.alerts.cache_get", return_value=None), \
         patch("api.routes.alerts.cache_set", return_value=True), \
         patch("api.routes.alerts.cache_delete", return_value=True):
        yield


@pytest.fixture
def no_redis():
    """/ping reads models_ready from Redis — keep it offline."""
    with patch("cache.redis_client.cache_get", return_value=None):
        yield


class FakeSession:
    """Minimal stand-in for a SQLAlchemy session: finds no user."""
    def query(self, *a, **k): return self
    def filter_by(self, **k): return self
    def first(self): return None
    def get(self, *a, **k): return None

@pytest.fixture(autouse=True)
def no_rate_limit_backend():
    """Keep rate limiting out of every test by default (no real Redis, no cross-test counts)."""
    with patch("api.rate_limit.rate_limit_check", return_value=True), \
         patch("api.routes.auth.rate_limit_check", return_value=True):
        yield    


# ── Health (public) ───────────────────────────────────────────────────────────

class TestHealth:
    def test_ping_returns_200(self, anon_client, no_redis):
        assert anon_client.get("/ping").status_code == 200

    def test_ping_response_schema(self, anon_client, no_redis):
        body = anon_client.get("/ping").json()
        assert "status" in body
        assert "demo_mode" in body
        assert "models_ready" in body
        assert "chroma_docs" in body
        assert body["status"] == "ok"


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestAuth:
    @pytest.mark.parametrize("method,path", [
        ("get",  "/api/alerts"),
        ("get",  "/api/forecast/TBP-001"),
        ("get",  "/api/forecast"),
        ("post", "/api/query"),
    ])
    def test_protected_routes_reject_missing_token(self, anon_client, method, path):
        kwargs = {"json": {"question": "hello there"}} if method == "post" else {}
        r = getattr(anon_client, method)(path, **kwargs)
        assert r.status_code == 401

    def test_garbage_token_rejected(self, anon_client):
        r = anon_client.get("/api/alerts", headers={"Authorization": "Bearer not-a-real-token"})
        assert r.status_code == 401

    def test_valid_token_grants_access(self, app_instance, anon_client, no_cache, monkeypatch):
        # Real JWT + real get_current_user; only the DB lookup is faked.
        monkeypatch.setattr(settings, "jwt_secret_key", "unit-test-secret-key-0123456789abcdef")
        from auth.security import create_access_token
        token = create_access_token(FAKE_USER.id, FAKE_USER.email, FAKE_USER.role)

        class UserFoundSession(FakeSession):
            def get(self, *a, **k): return FAKE_USER

        app_instance.dependency_overrides[get_session_dependency] = lambda: UserFoundSession()
        with patch("api.routes.alerts._reorder_alerts", return_value=[]), \
             patch("api.routes.alerts._anomaly_alerts", return_value=[]):
            r = anon_client.get("/api/alerts", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_login_wrong_credentials_returns_401(self, app_instance, anon_client):
        app_instance.dependency_overrides[get_session_dependency] = lambda: FakeSession()
        r = anon_client.post("/api/auth/login",
                             data={"username": "nobody@example.com", "password": "wrong"})
        assert r.status_code == 401


# ── Query endpoint ────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    @patch("api.routes.query.get_graph")
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

    @patch("api.routes.query.get_graph")
    def test_post_query_response_schema(self, mock_graph, client):
        mock_graph.return_value.invoke.return_value = {
            "intent":         "reorder",
            "sku_id":         "RSH-001",
            "rag_context":    "",
            "tool_result":    "Stock: 80, ROP: 120 — REORDER",
            "final_response": "RSH-001 needs reordering immediately.",
        }
        body = client.post("/api/query", json={"question": "check RSH-001 stock"}).json()
        for key in ("question", "intent", "sku_id", "answer", "rag_context_used"):
            assert key in body

    @patch("api.routes.query.get_graph")
    def test_rag_context_used_true_when_docs_found(self, mock_graph, client):
        mock_graph.return_value.invoke.return_value = {
            "intent": "supplier", "sku_id": "RSH-001",
            "rag_context": "[Source 1: catalog_05.pdf — Page 0]\nLakshmi Rolling Shutters...",
            "tool_result": "x", "final_response": "y",
        }
        body = client.post("/api/query", json={"question": "supplier for RSH-001"}).json()
        assert body["rag_context_used"] is True

    @patch("api.routes.query.get_graph")
    def test_rag_context_used_false_when_no_docs_found(self, mock_graph, client):
        mock_graph.return_value.invoke.return_value = {
            "intent": "supplier", "sku_id": "XYZ-001",
            "rag_context": "No relevant documents found.",
            "tool_result": "No supplier data found.", "final_response": "No data available.",
        }
        body = client.post("/api/query", json={"question": "who supplies XYZ-001"}).json()
        assert body["rag_context_used"] is False

    @patch("api.routes.query.get_graph")
    def test_graph_failure_returns_graceful_error(self, mock_graph, client):
        mock_graph.return_value.invoke.side_effect = RuntimeError("boom")
        r = client.post("/api/query", json={"question": "forecast for TBP-001"})
        assert r.status_code == 200
        assert r.json()["intent"] == "error"

    def test_query_validation_empty_string(self, client):
        assert client.post("/api/query", json={"question": ""}).status_code == 422

    def test_query_validation_too_long(self, client):
        assert client.post("/api/query", json={"question": "x" * 501}).status_code == 422


# ── Alerts endpoint ───────────────────────────────────────────────────────────

class TestAlertsEndpoint:
    @patch("api.routes.alerts._reorder_alerts", return_value=[])
    @patch("api.routes.alerts._anomaly_alerts", return_value=[])
    def test_alerts_returns_200(self, _anom, _reorder, client, no_cache):
        assert client.get("/api/alerts?refresh=true").status_code == 200

    @patch("api.routes.alerts._reorder_alerts", return_value=[])
    @patch("api.routes.alerts._anomaly_alerts", return_value=[])
    def test_alerts_response_schema(self, _anom, _reorder, client, no_cache):
        body = client.get("/api/alerts?refresh=true").json()
        for key in ("total_alerts", "reorder_alerts", "anomaly_alerts", "alerts"):
            assert key in body
        assert isinstance(body["alerts"], list)

    @patch("api.routes.alerts._reorder_alerts", return_value=[])
    @patch("api.routes.alerts._anomaly_alerts", return_value=[])
    def test_alerts_total_matches_lists(self, _anom, _reorder, client, no_cache):
        body = client.get("/api/alerts?refresh=true").json()
        assert body["total_alerts"] == body["reorder_alerts"] + body["anomaly_alerts"]


# ── Forecast endpoint ─────────────────────────────────────────────────────────

@pytest.fixture
def no_model_load():
    """_get_model() would otherwise download the real model from Storage."""
    with patch("api.routes.forecast._get_model", return_value=None):
        yield


class TestForecastEndpoint:
    @patch("api.routes.forecast.forecast_sku")
    def test_forecast_valid_sku(self, mock_fc, client, no_model_load):
        mock_fc.return_value = {
            "sku_id": "TBP-001",
            "forecast": [320.0, 340.0, 310.0],
            "lower":    [260.0, 278.0, 254.0],
            "upper":    [380.0, 402.0, 366.0],
        }
        assert client.get("/api/forecast/TBP-001?horizon=3").status_code == 200

    @patch("api.routes.forecast.forecast_sku")
    def test_forecast_response_schema(self, mock_fc, client, no_model_load):
        mock_fc.return_value = {
            "sku_id": "SC-001",
            "forecast": [150.0, 165.0],
            "lower":    [120.0, 132.0],
            "upper":    [180.0, 198.0],
        }
        body = client.get("/api/forecast/SC-001?horizon=2").json()
        assert body["sku_id"] == "SC-001"
        assert body["horizon"] == 2
        assert len(body["points"]) == 2
        for key in ("forecast", "lower_ci", "upper_ci"):
            assert key in body["points"][0]
        assert body["model"] == "Rolling Mean (fallback)"   # _get_model() mocked to None

    @patch("api.routes.forecast.forecast_sku")
    def test_forecast_unknown_sku_returns_404(self, mock_fc, client, no_model_load):
        mock_fc.return_value = {"sku_id": "XYZ-999", "forecast": [], "lower": [], "upper": []}
        assert client.get("/api/forecast/XYZ-999").status_code == 404

    def test_forecast_horizon_out_of_range(self, client):
        assert client.get("/api/forecast/TBP-001?horizon=15").status_code == 422

# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimit:
    def test_query_returns_429_when_limited(self, client):
        with patch("api.rate_limit.rate_limit_check", return_value=False):
            r = client.post("/api/query", json={"question": "forecast for TBP-001"})
        assert r.status_code == 429
        assert "retry-after" in r.headers

    def test_login_returns_429_when_limited(self, app_instance, anon_client):
        app_instance.dependency_overrides[get_session_dependency] = lambda: FakeSession()
        with patch("api.routes.auth.rate_limit_check", return_value=False):
            r = anon_client.post("/api/auth/login",
                                 data={"username": "a@example.com", "password": "x"})
        assert r.status_code == 429