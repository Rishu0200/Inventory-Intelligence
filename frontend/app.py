"""
Streamlit Dashboard — Inventory Intelligence System

API-only frontend: all data comes from the JWT-protected FastAPI backend.
It imports nothing from the backend code and reads no files, so it runs
unchanged on Streamlit Cloud.

Run locally:  streamlit run frontend/app.py
Config:       API_BASE_URL (Streamlit secret or environment variable)
"""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_api_base() -> str:
    try:
        if "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"])
    except Exception:
        pass   # no secrets file locally — fall through to env var / default
    return os.getenv("API_BASE_URL", "http://localhost:8000")


API_BASE = _get_api_base().rstrip("/")
REQUEST_TIMEOUT = 60   # Render's free tier can take ~50s to wake from sleep

st.markdown("""
<style>
.alert-high{border-left:4px solid #e74c3c;padding:6px 12px;margin:4px 0;background:#2d1515;}
.alert-med {border-left:4px solid #f39c12;padding:6px 12px;margin:4px 0;background:#2d2515;}
.alert-low {border-left:4px solid #2ecc71;padding:6px 12px;margin:4px 0;background:#152d1a;}
</style>
""", unsafe_allow_html=True)


# ── API layer ─────────────────────────────────────────────────────────────────

class AuthExpired(Exception):
    """Token missing/expired/rejected — user must log in again."""


class ApiError(Exception):
    """Any other API failure, with a message safe to show the user."""


def _request(method, path, token=None, timeout=REQUEST_TIMEOUT, **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.request(method, f"{API_BASE}{path}", headers=headers,
                             timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot connect to the API at {API_BASE}. Is it running?")
    except requests.exceptions.Timeout:
        raise ApiError("The API took too long to respond. It may be waking up — try again in a moment.")

    if r.status_code == 401 and token:
        raise AuthExpired()
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = ""
        raise ApiError(f"API error {r.status_code}: {detail}".strip().rstrip(":"))
    return r.json()


# Cached fetchers. The token is part of every cache key, so entries are per-login.
# Exceptions are never cached, so a failed call is retried on the next run.

@st.cache_data(ttl=600, show_spinner=False)
def fetch_alerts(token, force_refresh=False):
    path = "/api/alerts" + ("?refresh=true" if force_refresh else "")
    return _request("GET", path, token)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_forecast(token, sku_id, horizon):
    return _request("GET", f"/api/forecast/{sku_id}", token, params={"horizon": horizon})


@st.cache_data(ttl=600, show_spinner=False)
def fetch_skus(token):
    return _request("GET", "/api/skus", token)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_demand_history(token, sku_id, months=12):
    return _request("GET", f"/api/demand/{sku_id}", token, params={"months": months})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_inventory(token):
    return _request("GET", "/api/inventory", token)


@st.cache_data(ttl=30, show_spinner=False)
def health_check():
    try:
        return _request("GET", "/ping", None, timeout=30)
    except ApiError:
        return {}


def ask_query(token, question):
    return _request("POST", "/api/query", token, json={"question": question}, timeout=90)


# ── Session helpers ───────────────────────────────────────────────────────────

def logout(message=None):
    for key in ("token", "email", "messages"):
        st.session_state.pop(key, None)
    if message:
        st.session_state["login_notice"] = message
    st.rerun()


def call(fn, *args, quiet=False, **kwargs):
    """Run a fetcher; log out on expired auth, show (or hide) other errors."""
    try:
        return fn(*args, **kwargs)
    except AuthExpired:
        logout("Your session expired. Please sign in again.")
    except ApiError as e:
        if not quiet:
            st.warning(str(e))
        return None


def login_screen():
    st.title("📦 Inventory Intelligence")
    st.caption("Uninox Houseware · sign in to continue")

    notice = st.session_state.pop("login_notice", None)
    if notice:
        st.warning(notice)

    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if not email or not password:
            st.error("Enter both email and password.")
            return
        with st.spinner("Signing in… (the API can take up to a minute to wake up)"):
            try:
                data = _request("POST", "/api/auth/login", None,
                                data={"username": email.strip(), "password": password})
            except ApiError as e:
                st.error(str(e))
                return
        st.session_state["token"] = data["access_token"]
        st.session_state["email"] = email.strip()
        st.rerun()


# ── Gate: nothing below renders without a token ───────────────────────────────

token = st.session_state.get("token")
if not token:
    login_screen()
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/4B8BBE/warehouse.png", width=60)
    st.title("Inventory Intelligence")
    st.caption("Uninox Houseware · Delhi")
    st.caption(f"Signed in as {st.session_state.get('email', '')}")
    if st.button("Sign out"):
        logout()
    st.divider()

    health = health_check()
    if health:
        c1, c2 = st.columns(2)
        c1.metric("Models", "✓ Ready" if health.get("models_ready") else "✗ Missing")
        c2.metric("ChromaDB", f"{health.get('chroma_docs', 0)} docs")
        if health.get("demo_mode"):
            st.info("🔵 Demo mode — no LLM calls")
    else:
        st.error("API offline or waking up")

    st.divider()
    st.subheader("🚨 Live Alerts")
    side_alerts = call(fetch_alerts, token, quiet=True)
    if side_alerts:
        c1, c2 = st.columns(2)
        c1.metric("Reorder", side_alerts.get("reorder_alerts", 0))
        c2.metric("Anomaly", side_alerts.get("anomaly_alerts", 0))

        css_map = {"high": "alert-high", "medium": "alert-med", "low": "alert-low"}
        icon_map = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for alert in side_alerts.get("alerts", [])[:5]:
            st.markdown(
                f'<div class="{css_map.get(alert["severity"], "alert-low")}">'
                f'{icon_map.get(alert["severity"], "🟢")} <b>{alert["sku_id"]}</b> — '
                f'{alert["item_name"][:20]}<br><small>{alert["message"][:80]}</small></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Could not load alerts.")

    st.divider()
    st.caption("📊 Built with LangGraph · ChromaDB · FastAPI")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_forecast, tab_stock, tab_alerts = st.tabs(
    ["💬 Chat", "📈 Forecast", "📦 Stock Levels", "🚨 All Alerts"]
)


# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

with tab_chat:
    st.header("Ask anything about your inventory")

    st.caption("Quick prompts:")
    quick_qs = [
        "Which SKUs need reordering?",
        "Forecast demand for TBP-001",
        "Who is the best supplier for RSH-001?",
        "Any anomalies this quarter?",
    ]
    for i, qcol in enumerate(st.columns(4)):
        if qcol.button(quick_qs[i], key=f"quick_{i}"):
            st.session_state["pending_question"] = quick_qs[i]

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Inventory Intelligence assistant for Uninox Houseware.\n\n"
                "I can help with:\n"
                "- 📊 **Demand forecasting** for any SKU\n"
                "- 🔄 **Reorder alerts** and safety stock calculations\n"
                "- 🤝 **Supplier recommendations** and lead times\n"
                "- 🔍 **Anomaly detection** in demand patterns\n\n"
                "Try a quick prompt above or type your question below."
            ),
        }]

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about your inventory...")
    if not user_input:
        user_input = st.session_state.pop("pending_question", None)

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            intent, sku = "", ""
            with st.spinner("Thinking..."):
                try:
                    result = ask_query(token, user_input)
                    answer = result.get("answer", "No response.")
                    intent = result.get("intent", "")
                    sku = result.get("sku_id", "")
                except AuthExpired:
                    logout("Your session expired. Please sign in again.")
                except ApiError as e:
                    answer = f"⚠️ {e}"

            st.markdown(answer)
            if intent and intent != "error":
                c1, c2 = st.columns(2)
                c1.caption(f"🎯 Intent: `{intent}`")
                if sku:
                    c2.caption(f"🏷️ SKU: `{sku}`")

        st.session_state["messages"].append({"role": "assistant", "content": answer})


# ── Tab 2: Forecast ───────────────────────────────────────────────────────────

with tab_forecast:
    st.header("📈 Demand Forecast")

    skus = call(fetch_skus, token) or []
    if not skus:
        st.info("No SKUs available yet — is the database populated?")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_sku = st.selectbox("Select SKU", skus)
            horizon = st.slider("Forecast horizon (months)", 1, 6, 3)
            show_hist = st.checkbox("Show historical data", value=True)

        fc_data = call(fetch_forecast, token, selected_sku, horizon)

        with col2:
            if fc_data and fc_data.get("points"):
                fig = go.Figure()

                last_hist_period = None
                if show_hist:
                    hist = call(fetch_demand_history, token, selected_sku, quiet=True)
                    if hist and hist.get("points"):
                        periods = pd.to_datetime([p["period"] for p in hist["points"]], format="%Y-%m")
                        last_hist_period = periods.max()
                        fig.add_trace(go.Scatter(
                            x=list(periods.strftime("%b %Y")),
                            y=[p["net_units"] for p in hist["points"]],
                            name="Historical",
                            line=dict(color="#4B8BBE", width=2),
                            mode="lines+markers",
                        ))

                pts = fc_data["points"]
                if last_hist_period is not None:
                    # Continue the same "Mon YYYY" labels so history and forecast
                    # sit on one continuous timeline.
                    months = [
                        (last_hist_period + pd.DateOffset(months=p["month_offset"])).strftime("%b %Y")
                        for p in pts
                    ]
                else:
                    months = [f"Month +{p['month_offset']}" for p in pts]

                forecasts = [p["forecast"] for p in pts]
                lower_ci  = [p["lower_ci"] for p in pts]
                upper_ci  = [p["upper_ci"] for p in pts]

                fig.add_trace(go.Bar(x=months, y=forecasts, name="Forecast",
                                     marker_color="#4EC994", opacity=0.8))
                fig.add_trace(go.Scatter(
                    x=months + months[::-1],
                    y=upper_ci + lower_ci[::-1],
                    fill="toself",
                    fillcolor="rgba(78,201,148,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="90% CI",
                ))
                fig.update_layout(
                    title=f"Demand Forecast — {selected_sku}",
                    xaxis_title="Period", yaxis_title="Units",
                    template="plotly_dark", height=400, showlegend=True,
                )
                st.plotly_chart(fig)
                st.caption(f"Model: {fc_data.get('model', 'unknown')}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Next Month", f"{pts[0]['forecast']:.0f} units")
                m2.metric("Lower CI", f"{pts[0]['lower_ci']:.0f}")
                m3.metric("Upper CI", f"{pts[0]['upper_ci']:.0f}")
            else:
                st.info("No forecast available for this SKU.")


# ── Tab 3: Stock Levels ───────────────────────────────────────────────────────

with tab_stock:
    st.header("📦 Current Stock Levels")
    inventory = call(fetch_inventory, token)

    if inventory:
        inv_df = pd.DataFrame(inventory)

        def status_color(status):
            if "Critical" in str(status): return "🔴"
            if "Low" in str(status):      return "🟡"
            return "🟢"

        display = inv_df.copy()
        display["Status"] = display["status"].apply(lambda s: f"{status_color(s)} {s}")
        display["Gap vs ROP"] = (display["total_available"] - display["reorder_point"]).round(0)

        show_cols = ["sku_id", "item_name", "qty_on_hand", "total_available",
                     "reorder_point", "Gap vs ROP", "days_of_stock", "Status"]
        st.dataframe(
            display[show_cols],
            hide_index=True,
            column_config={
                "Gap vs ROP": st.column_config.NumberColumn(format="%+.0f"),
                "days_of_stock": st.column_config.NumberColumn("Days Stock"),
            },
        )

        fig2 = go.Figure()
        fig2.add_bar(x=inv_df["sku_id"], y=inv_df["total_available"],
                     name="Available", marker_color="#4B8BBE")
        fig2.add_bar(x=inv_df["sku_id"], y=inv_df["reorder_point"],
                     name="Reorder Point", marker_color="#e74c3c", opacity=0.7)
        fig2.update_layout(barmode="overlay", template="plotly_dark",
                           title="Stock Available vs Reorder Point",
                           height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig2)
    else:
        st.warning("Could not load inventory data.")


# ── Tab 4: All Alerts ─────────────────────────────────────────────────────────

with tab_alerts:
    st.header("🚨 All Alerts")

    if st.button("🔄 Refresh Alerts"):
        fetch_alerts.clear()                        # drop this app's cached copy...
        call(fetch_alerts, token, force_refresh=True)   # ...and bust the API's cache too
        st.rerun()

    alerts_data = call(fetch_alerts, token)
    if alerts_data:
        a, b, c = st.columns(3)
        a.metric("Total Alerts",   alerts_data.get("total_alerts", 0))
        b.metric("Reorder Alerts", alerts_data.get("reorder_alerts", 0))
        c.metric("Anomaly Alerts", alerts_data.get("anomaly_alerts", 0))

        alerts_list = alerts_data.get("alerts", [])
        if alerts_list:
            cols = ["sku_id", "item_name", "alert_type", "severity", "current_stock",
                    "reorder_point", "gap", "anomaly_score", "message"]
            st.dataframe(pd.DataFrame(alerts_list).reindex(columns=cols), hide_index=True)
        else:
            st.success("✓ No alerts at this time. All inventory levels are healthy.")
    else:
        st.error("Could not load alerts from the API.")
