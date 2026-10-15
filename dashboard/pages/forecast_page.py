"""
Stock Forecast Page — Financial Intelligence Platform
=====================================================
Renders technical analysis overlays, key metrics, and an XGBoost-based
directional prediction (UP / DOWN) with confidence score.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.data_loader import (
    TICKERS,
    TICKER_NAMES,
    load_processed_data,
)
from dashboard.components.charts import technical_chart, candlestick_chart, COLORS

# ---------------------------------------------------------------------------
# XGBoost helpers
# ---------------------------------------------------------------------------
FEATURES = [
    "EMA_20",
    "EMA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "ATR_14",
    "Daily_Return",
    "Volatility_20",
]

MODEL_PATH = os.path.join("models", "xgboost_direction_model.json")


def _load_model():
    """Load the saved XGBClassifier; returns (model, error_msg | None)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, "xgboost is not installed. Run `pip install xgboost`."

    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at `{MODEL_PATH}`."

    try:
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        return model, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


def _build_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the latest row's feature vector.
    Dynamically includes any RAG_ columns present in the dataframe.
    """
    feature_cols = [f for f in FEATURES if f in df.columns]
    # Include any RAG-derived columns (e.g. RAG_Sentiment, RAG_Score, …)
    rag_cols = [c for c in df.columns if c.startswith("RAG_")]
    feature_cols = feature_cols + [c for c in rag_cols if c not in feature_cols]

    if not feature_cols:
        return pd.DataFrame()

    latest = df[feature_cols].dropna().tail(1)
    return latest


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    """Entry-point called by the app router."""

    st.markdown("## 📈 Stock Forecast")

    # ── Layout: sidebar-style column + main area ──────────────────────────
    sidebar_col, main_col = st.columns([1, 3], gap="large")

    # ── Sidebar controls ─────────────────────────────────────────────────
    with sidebar_col:
        st.markdown("#### Controls")

        ticker = st.selectbox(
            "Ticker",
            options=TICKERS,
            format_func=lambda t: f"{t}  —  {TICKER_NAMES.get(t, '')}",
            key="forecast_ticker",
        )

        days = st.selectbox(
            "Date Range (days)",
            options=[60, 90, 120, 250],
            index=2,
            key="forecast_days",
        )

        st.markdown("---")
        st.markdown("##### Indicators")
        show_bb = st.checkbox("Bollinger Bands", value=True, key="forecast_bb")
        show_ema = st.checkbox("EMA (20 & 50)", value=True, key="forecast_ema")
        show_macd = st.checkbox("MACD", value=False, key="forecast_macd")
        show_rsi = st.checkbox("RSI", value=False, key="forecast_rsi")

        st.markdown("---")
        run_prediction = st.button(
            "🚀 Run Prediction", use_container_width=True, key="forecast_run"
        )

    # ── Main content area ────────────────────────────────────────────────
    with main_col:
        # Load data with spinner
        try:
            with st.spinner(f"Loading data for **{ticker}** …"):
                df = load_processed_data(ticker)
        except Exception as exc:
            st.error(f"Error loading data: {exc}")
            return

        if df.empty:
            st.warning(
                f"No processed data found for **{ticker}**. "
                "Run the feature-engineering pipeline first."
            )
            return

        # ── Technical chart ──────────────────────────────────────────────
        indicators = {
            "bb": show_bb,
            "ema": show_ema,
            "macd": show_macd,
            "rsi": show_rsi,
        }

        try:
            fig = technical_chart(df, indicators=indicators, days=days)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Chart rendering error: {exc}")

        # ── Key metrics row ──────────────────────────────────────────────
        st.markdown("##### Key Metrics")
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else latest

            last_close = latest.get("Close", np.nan)
            daily_return = latest.get("Daily_Return", np.nan)
            rsi_14 = latest.get("RSI_14", np.nan)
            atr_14 = latest.get("ATR_14", np.nan)
            vol_20 = latest.get("Volatility_20", np.nan)

            m1, m2, m3, m4, m5 = st.columns(5)

            with m1:
                delta_close = last_close - prev.get("Close", last_close)
                st.metric(
                    "Last Close",
                    f"${last_close:,.2f}" if pd.notna(last_close) else "N/A",
                    delta=f"{delta_close:+.2f}" if pd.notna(delta_close) else None,
                )
            with m2:
                st.metric(
                    "Daily Return",
                    f"{daily_return:+.2%}" if pd.notna(daily_return) else "N/A",
                    delta=None,
                )
            with m3:
                rsi_label = (
                    "Overbought" if pd.notna(rsi_14) and rsi_14 > 70
                    else "Oversold" if pd.notna(rsi_14) and rsi_14 < 30
                    else None
                )
                st.metric(
                    "RSI 14",
                    f"{rsi_14:.1f}" if pd.notna(rsi_14) else "N/A",
                    delta=rsi_label,
                    delta_color="inverse" if rsi_label == "Overbought" else "normal",
                )
            with m4:
                st.metric(
                    "ATR 14",
                    f"{atr_14:.2f}" if pd.notna(atr_14) else "N/A",
                )
            with m5:
                st.metric(
                    "Volatility 20",
                    f"{vol_20:.2%}" if pd.notna(vol_20) else "N/A",
                )
        except Exception as exc:
            st.warning(f"Could not compute metrics: {exc}")

        # ── XGBoost prediction ───────────────────────────────────────────
        if run_prediction:
            st.markdown("---")
            st.markdown("##### 🤖 XGBoost Direction Prediction")

            with st.spinner("Running prediction model …"):
                model, err = _load_model()

            if err:
                st.warning(err)
            else:
                try:
                    X = _build_feature_row(df)

                    if X.empty:
                        st.warning(
                            "Feature columns are missing from the dataset. "
                            "Make sure the feature-engineering pipeline has run."
                        )
                    else:
                        # Align features to what the model expects
                        try:
                            expected = model.get_booster().feature_names
                            if expected:
                                # Re-order / filter to model's expected columns
                                missing = [f for f in expected if f not in X.columns]
                                if missing:
                                    st.info(
                                        f"Some expected features are missing and "
                                        f"will be zero-filled: {', '.join(missing)}"
                                    )
                                    for col in missing:
                                        X[col] = 0.0
                                X = X[expected]
                        except Exception:
                            pass  # model may not expose feature_names; proceed

                        pred = model.predict(X)[0]
                        proba = model.predict_proba(X)[0]
                        confidence = float(proba[int(pred)])

                        direction = "UP ⬆️" if int(pred) == 1 else "DOWN ⬇️"
                        color = COLORS["green"] if int(pred) == 1 else COLORS["red"]

                        r1, r2 = st.columns([1, 2])
                        with r1:
                            st.markdown(
                                f"<div style='text-align:center; padding:1.5rem; "
                                f"border-radius:12px; "
                                f"background:linear-gradient(135deg, {color}22, {color}08); "
                                f"border:1px solid {color}44;'>"
                                f"<span style='font-size:2.4rem; font-weight:700; "
                                f"color:{color};'>{direction}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        with r2:
                            st.markdown(
                                f"<div style='padding:1rem 1.5rem;'>"
                                f"<b>Confidence:</b> "
                                f"<span style='font-size:1.6rem; color:{color};'>"
                                f"{confidence:.1%}</span><br/>"
                                f"<span style='color:#94A3B8; font-size:0.85rem;'>"
                                f"Based on latest technical features for "
                                f"<b>{ticker}</b></span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        # Probability breakdown
                        st.progress(
                            confidence,
                            text=f"P(UP) = {proba[1]:.1%}  ·  P(DOWN) = {proba[0]:.1%}",
                        )

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
