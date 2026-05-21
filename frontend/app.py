"""
Streamlit Dashboard — Inventory Intelligence System
Chat interface + Reorder alerts + Demand forecast charts + Stock table.
Run: streamlit run frontend/app.py
"""
import os
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:#1e1e2e;border-radius:10px;padding:16px;text-align:center;}
.alert-high{border-left:4px solid #e74c3c;padding:6px 12px;margin:4px 0;background:#2d1515;}
.alert-med {border-left:4px solid #f39c12;padding:6px 12px;margin:4px 0;background:#2d2515;}
.alert-low {border-left:4px solid #2ecc71;padding:6px 12px;margin:4px 0;background:#152d1a;}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def fetch_alerts():
    try:
        r = requests.get(f"{API_BASE}/api/alerts", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_forecast(sku_id: str, horizon: int = 3):
    try:
        r = requests.get(f"{API_BASE}/api/forecast/{sku_id}?horizon={horizon}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=600)
def load_inventory():
    try:
        from config import Paths
        return pd.read_csv(Paths.DATA_RAW / "inventory_history.csv")
    except Exception:
        return pd.DataFrame()


def ask_query(question: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/api/query",
            json={"question": question},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else {"answer": f"API error: {r.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"answer": "⚠️ Cannot connect to API. Make sure FastAPI is running:\n`uvicorn api.main:app --reload`"}
    except Exception as e:
        return {"answer": f"Error: {e}"}


def health_check() -> dict:
    try:
        r = requests.get(f"{API_BASE}/ping", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/4B8BBE/warehouse.png", width=60)
    st.title("Inventory Intelligence")
    st.caption("Uninox Houseware · Delhi")
    st.divider()

    # Health status
    health = health_check()
    if health:
        col1, col2 = st.columns(2)
        col1.metric("Models", "✓ Ready" if health.get("models_ready") else "✗ Missing")
        col2.metric("ChromaDB", f"{health.get('chroma_docs', 0)} docs")
        if health.get("demo_mode"):
            st.info("🔵 Demo mode — no LLM calls")
    else:
        st.error("API offline")

    st.divider()

    # Alert summary in sidebar
    st.subheader("🚨 Live Alerts")
    alerts_data = fetch_alerts()
    if alerts_data:
        c1, c2 = st.columns(2)
        c1.metric("Reorder", alerts_data.get("reorder_alerts", 0))
        c2.metric("Anomaly", alerts_data.get("anomaly_alerts", 0))

        for alert in alerts_data.get("alerts", [])[:5]:
            css = {"high": "alert-high", "medium": "alert-med", "low": "alert-low"}
            cls = css.get(alert["severity"], "alert-low")
            icon = "🔴" if alert["severity"] == "high" else "🟡" if alert["severity"] == "medium" else "🟢"
            st.markdown(
                f'<div class="{cls}">{icon} <b>{alert["sku_id"]}</b> — {alert["item_name"][:20]}<br>'
                f'<small>{alert["message"][:80]}</small></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Could not load alerts.")

    st.divider()
    st.caption("📊 Built with LangGraph · ChromaDB · FastAPI")


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_chat, tab_forecast, tab_stock, tab_alerts = st.tabs([
    "💬 Chat", "📈 Forecast", "📦 Stock Levels", "🚨 All Alerts"
])


# ── Tab 1: Chat Interface ─────────────────────────────────────────────────────

with tab_chat:
    st.header("Ask anything about your inventory")

    # Quick prompts
    st.caption("Quick prompts:")
    q_cols = st.columns(4)
    quick_qs = [
        "Which SKUs need reordering?",
        "Forecast demand for TBP-001",
        "Who is the best supplier for RSH-001?",
        "Any anomalies this quarter?",
    ]
    for i, qcol in enumerate(q_cols):
        if qcol.button(quick_qs[i], use_container_width=True):
            st.session_state.user_input = quick_qs[i]

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Inventory Intelligence assistant for Uninox Houseware.\n\n"
                "I can help you with:\n"
                "- 📊 **Demand forecasting** for any SKU\n"
                "- 🔄 **Reorder alerts** and safety stock calculations\n"
                "- 🤝 **Supplier recommendations** and lead times\n"
                "- 🔍 **Anomaly detection** in demand patterns\n\n"
                "Try one of the quick prompts above or type your question below."
            ),
        })

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about your inventory...")
    if not user_input and "user_input" in st.session_state:
        user_input = st.session_state.pop("user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_query(user_input)
                answer = result.get("answer", "No response.")
                intent = result.get("intent", "")
                sku    = result.get("sku_id", "")

            st.markdown(answer)
            if intent:
                c1, c2 = st.columns(2)
                c1.caption(f"🎯 Intent: `{intent}`")
                if sku:
                    c2.caption(f"🏷️ SKU: `{sku}`")

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ── Tab 2: Forecast Charts ────────────────────────────────────────────────────

with tab_forecast:
    st.header("📈 Demand Forecast")

    try:
        from config import Paths
        inv_df = pd.read_csv(Paths.DATA_RAW / "demand_history.csv")
        sku_list = sorted(inv_df["sku_id"].unique().tolist())
    except Exception:
        sku_list = ["SC-001", "TBP-001", "PBP-001", "RSH-001", "CHM-001", "MGC-001"]

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_sku = st.selectbox("Select SKU", sku_list)
        horizon      = st.slider("Forecast horizon (months)", 1, 6, 3)
        show_hist    = st.checkbox("Show historical data", value=True)

    fc_data = fetch_forecast(selected_sku, horizon)

    with col2:
        if fc_data and fc_data.get("points"):
            fig = go.Figure()

            # Historical line
            if show_hist:
                try:
                    hist = inv_df[inv_df["sku_id"] == selected_sku].copy()
                    hist["period"] = pd.to_datetime(hist["period"], format="%Y-%m")
                    hist = hist.sort_values("period").tail(12)
                    fig.add_trace(go.Scatter(
                        x=hist["period"].dt.strftime("%b %Y"),
                        y=hist["net_units"],
                        name="Historical",
                        line=dict(color="#4B8BBE", width=2),
                        mode="lines+markers",
                    ))
                except Exception:
                    pass

            # Forecast bars
            pts = fc_data["points"]
            months = [f"Month +{p['month_offset']}" for p in pts]
            forecasts = [p["forecast"] for p in pts]
            lower_ci  = [p["lower_ci"] for p in pts]
            upper_ci  = [p["upper_ci"] for p in pts]

            fig.add_trace(go.Bar(
                x=months, y=forecasts,
                name="Forecast",
                marker_color="#4EC994",
                opacity=0.8,
            ))
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
                xaxis_title="Period",
                yaxis_title="Units",
                template="plotly_dark",
                height=400,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Next Month", f"{pts[0]['forecast']:.0f} units")
            m2.metric("Lower CI",   f"{pts[0]['lower_ci']:.0f}")
            m3.metric("Upper CI",   f"{pts[0]['upper_ci']:.0f}")
        else:
            st.info("No forecast data. Train models first:\n```\npython scripts/train_models.py\n```")


# ── Tab 3: Stock Levels ───────────────────────────────────────────────────────

with tab_stock:
    st.header("📦 Current Stock Levels")
    inv_df = load_inventory()

    if not inv_df.empty:
        # Color-code status
        def status_color(status: str) -> str:
            if "Critical" in str(status): return "🔴"
            if "Low"      in str(status): return "🟡"
            return "🟢"

        display = inv_df.copy()
        display["Status"] = display["status"].apply(
            lambda s: f"{status_color(s)} {s}"
        )
        display["Gap vs ROP"] = (display["total_available"] - display["reorder_point"]).round(0)

        show_cols = ["sku_id", "item_name", "qty_on_hand", "total_available",
                     "reorder_point", "Gap vs ROP", "days_of_stock", "Status"]
        show_cols = [c for c in show_cols if c in display.columns]

        st.dataframe(
            display[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Gap vs ROP": st.column_config.NumberColumn(format="%+.0f"),
                "days_of_stock": st.column_config.NumberColumn("Days Stock"),
            }
        )

        # Bar chart: stock vs ROP
        fig2 = go.Figure()
        fig2.add_bar(x=inv_df["sku_id"], y=inv_df["total_available"],
                     name="Available", marker_color="#4B8BBE")
        fig2.add_bar(x=inv_df["sku_id"], y=inv_df["reorder_point"],
                     name="Reorder Point", marker_color="#e74c3c", opacity=0.7)
        fig2.update_layout(
            barmode="overlay", template="plotly_dark",
            title="Stock Available vs Reorder Point",
            height=350, xaxis_tickangle=-45,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Could not load inventory data.")


# ── Tab 4: All Alerts ─────────────────────────────────────────────────────────

with tab_alerts:
    st.header("🚨 All Alerts")

    if st.button("🔄 Refresh Alerts"):
        fetch_alerts.clear()

    alerts_data = fetch_alerts()
    if alerts_data:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Alerts",   alerts_data.get("total_alerts", 0))
        col_b.metric("Reorder Alerts", alerts_data.get("reorder_alerts", 0))
        col_c.metric("Anomaly Alerts", alerts_data.get("anomaly_alerts", 0))

        alerts_list = alerts_data.get("alerts", [])
        if alerts_list:
            df_alerts = pd.DataFrame(alerts_list)
            st.dataframe(
                df_alerts[["sku_id", "item_name", "alert_type", "severity",
                            "current_stock", "reorder_point", "gap", "message"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("✓ No alerts at this time. All inventory levels are healthy.")
    else:
        st.error("Could not load alerts from API.")
