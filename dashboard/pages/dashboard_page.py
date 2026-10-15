"""
Dashboard Page — Financial Intelligence Platform
Main overview page: KPI metrics, market snapshot, predictions feed, system health.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from dashboard.components.data_loader import (
    TICKERS,
    TICKER_NAMES,
    load_processed_data,
    get_system_status,
)
from dashboard.components.charts import sparkline, COLORS


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_all_ticker_summaries() -> pd.DataFrame:
    """Load latest row for every ticker and build a summary DataFrame."""
    rows: list[dict] = []
    for ticker in TICKERS:
        try:
            df = load_processed_data(ticker)
            if df.empty:
                continue
            last = df.iloc[-1]
            prev_close = df["Close"].iloc[-2] if len(df) >= 2 else last["Close"]
            daily_ret = (last["Close"] - prev_close) / prev_close * 100

            rsi = last.get("RSI_14", np.nan)

            # Trend arrow based on 5-day slope of close prices
            if len(df) >= 5:
                slope = np.polyfit(range(5), df["Close"].iloc[-5:].values, 1)[0]
            else:
                slope = 0.0

            if slope > 0.3:
                trend = "🟢 ▲"
            elif slope < -0.3:
                trend = "🔴 ▼"
            else:
                trend = "🟡 ▸"

            rows.append(
                {
                    "Ticker": ticker,
                    "Name": TICKER_NAMES.get(ticker, ticker),
                    "Close": round(last["Close"], 2),
                    "Daily Return %": round(daily_ret, 2),
                    "RSI (14)": round(rsi, 1) if not np.isnan(rsi) else "—",
                    "Trend": trend,
                }
            )
        except Exception:
            # Silently skip tickers that can't be loaded
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _mock_predictions(n: int = 8) -> pd.DataFrame:
    """Generate realistic-looking mock XGBoost prediction rows."""
    np.random.seed(42)
    now = datetime.now()
    records = []
    for i in range(n):
        ticker = np.random.choice(TICKERS)
        direction = np.random.choice(["UP", "DOWN"], p=[0.55, 0.45])
        confidence = round(np.random.uniform(0.58, 0.94), 2)
        ts = now - timedelta(minutes=np.random.randint(5, 480))
        records.append(
            {
                "Ticker": ticker,
                "Prediction": direction,
                "Confidence": f"{confidence:.0%}",
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            }
        )
    return pd.DataFrame(records).sort_values("Timestamp", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Section renderers
# ─────────────────────────────────────────────────────────────

def _render_kpi_row(summary_df: pd.DataFrame) -> None:
    """Top-level KPI metric cards."""
    assets_tracked = len(summary_df) if not summary_df.empty else 0
    predictions_today = np.random.RandomState(
        int(datetime.now().strftime("%Y%m%d"))
    ).randint(12, 35)

    # Approximate RAG document count (mock)
    rag_docs = 1_247

    # Uptime (mock — would come from a monitoring service in prod)
    uptime = "99.97 %"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Assets Tracked",
            value=assets_tracked,
            delta=f"{assets_tracked} active",
        )
    with c2:
        st.metric(
            label="Predictions Today",
            value=predictions_today,
            delta="live",
        )
    with c3:
        st.metric(
            label="RAG Documents",
            value=f"{rag_docs:,}",
            delta="+18 new",
        )
    with c4:
        st.metric(
            label="System Uptime",
            value=uptime,
            delta="healthy",
        )


def _render_market_overview(summary_df: pd.DataFrame) -> None:
    """Market summary table with sparklines."""
    st.markdown("### 📊 Market Overview")

    if summary_df.empty:
        st.info(
            "No processed data found. Run the feature-engineering pipeline "
            "to populate the market overview.",
            icon="ℹ️",
        )
        return

    # Style the dataframe — colour returns
    def _colour_return(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v > 0:
            return f"color: {COLORS['green']}; font-weight: 600;"
        elif v < 0:
            return f"color: {COLORS['red']}; font-weight: 600;"
        return ""

    styled = (
        summary_df.style
        .applymap(_colour_return, subset=["Daily Return %"])
        .format({"Close": "${:,.2f}", "Daily Return %": "{:+.2f} %"})
        .set_properties(**{"text-align": "center"})
        .set_properties(subset=["Name"], **{"text-align": "left"})
    )
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(56 + len(summary_df) * 38, 400),
    )

    # Sparkline row — show one sparkline per ticker for last 30 closes
    st.markdown(
        "<p style='margin-top:0.8rem;margin-bottom:0.2rem;color:#94A3B8;"
        "font-size:0.78rem;font-weight:600;letter-spacing:0.06em;"
        "text-transform:uppercase;'>30-Day Price Sparklines</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(TICKERS))
    for col, ticker in zip(cols, TICKERS):
        try:
            df = load_processed_data(ticker)
            if df.empty or len(df) < 5:
                with col:
                    st.caption(ticker)
                    st.caption("—")
                continue
            series = df["Close"].iloc[-30:]
            fig = sparkline(series, height=64)
            with col:
                st.caption(ticker)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            with col:
                st.caption(ticker)
                st.caption("n/a")


def _render_recent_predictions() -> None:
    """Mock table of recent XGBoost predictions."""
    st.markdown("### 🤖 Recent Predictions")

    preds = _mock_predictions()

    def _style_pred(val):
        if val == "UP":
            return f"color: {COLORS['green']}; font-weight: 700;"
        elif val == "DOWN":
            return f"color: {COLORS['red']}; font-weight: 700;"
        return ""

    def _style_conf(val):
        try:
            v = float(val.strip("%")) / 100
        except (ValueError, AttributeError):
            return ""
        if v >= 0.80:
            return f"color: {COLORS['green']};"
        elif v >= 0.65:
            return f"color: {COLORS['amber']};"
        return f"color: {COLORS['gray']};"

    styled = (
        preds.style
        .applymap(_style_pred, subset=["Prediction"])
        .applymap(_style_conf, subset=["Confidence"])
        .set_properties(**{"text-align": "center"})
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(56 + len(preds) * 38, 380),
    )


def _render_system_status() -> None:
    """Show health indicators for each subsystem."""
    st.markdown("### ⚙️ System Status")

    try:
        status = get_system_status()
    except Exception as exc:
        st.error(f"Could not retrieve system status: {exc}", icon="🚨")
        return

    labels = {
        "raw_data": ("Raw Data Pipeline", "Market data ingestion via yfinance"),
        "processed_data": ("Feature Engineering", "Technical indicators & feature generation"),
        "xgboost_model": ("XGBoost Model", "Direction-prediction model artefact"),
        "chroma_db": ("ChromaDB Vector Store", "RAG document embeddings"),
    }

    cols = st.columns(len(labels))
    for col, (key, (name, desc)) in zip(cols, labels.items()):
        ok = status.get(key, False)
        with col:
            if ok:
                st.success(f"**{name}**\n\n{desc}", icon="✅")
            else:
                st.error(f"**{name}**\n\n{desc}", icon="❌")


# ─────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────

def render() -> None:
    """Render the main Dashboard overview page."""

    # ── Page header ──────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0.1rem;'>"
        "📈 Financial Intelligence Platform"
        "</h1>"
        "<p style='color:#94A3B8;margin-top:0;margin-bottom:1.4rem;"
        "font-size:0.92rem;'>"
        "Real-time market analytics · ML predictions · RAG-powered insights"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── KPI row ──────────────────────────────────────────────
    try:
        summary_df = _load_all_ticker_summaries()
    except Exception as exc:
        st.error(f"Failed to load ticker data: {exc}", icon="🚨")
        summary_df = pd.DataFrame()

    _render_kpi_row(summary_df)

    st.divider()

    # ── Main content: Market Overview + Predictions side-by-side ─
    left, right = st.columns([3, 2], gap="large")

    with left:
        try:
            _render_market_overview(summary_df)
        except Exception as exc:
            st.error(f"Market overview error: {exc}", icon="⚠️")

    with right:
        try:
            _render_recent_predictions()
        except Exception as exc:
            st.error(f"Predictions table error: {exc}", icon="⚠️")

    st.divider()

    # ── System Status row ────────────────────────────────────
    try:
        _render_system_status()
    except Exception as exc:
        st.error(f"System status error: {exc}", icon="⚠️")

    # ── Footer timestamp ─────────────────────────────────────
    st.markdown(
        f"<p style='text-align:right;color:#64748B;font-size:0.75rem;"
        f"margin-top:2rem;'>"
        f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</p>",
        unsafe_allow_html=True,
    )
