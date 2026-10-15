"""
News Sentiment Analysis page for the Financial Intelligence Platform.

Provides interactive FinBERT-powered sentiment analysis on user-supplied
headlines / earnings summaries, plus a one-click sample batch mode.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.components.charts import (
    sentiment_gauge,
    COLORS,
    PLOTLY_BG,
    PLOTLY_PAPER_BG,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SAMPLE_HEADLINES: list[str] = [
    "Apple Inc. beats quarterly revenue expectations, shares surge 4% in after-hours trading.",
    "Tesla recalls over 100,000 vehicles due to software glitch, stock drops in premarket.",
    "Federal Reserve officials hold interest rates steady as inflation concerns linger.",
    "Nvidia launches new Blackwell chips, projecting massive revenue growth next year.",
    "Goldman Sachs downgrades retail sector amid weakening consumer spending data.",
    "Microsoft Azure cloud revenue jumps 29%, sending shares to all-time highs.",
]

_SENTIMENT_COLORS: dict[str, str] = {
    "Positive": COLORS["green"],
    "Neutral": COLORS["amber"],
    "Negative": COLORS["red"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_analyzer():
    """Return a cached FinBERTSentimentAnalyzer stored in session state."""
    if "finbert_analyzer" not in st.session_state:
        with st.spinner("⏳ Loading FinBERT model — this may take a moment on first run…"):
            try:
                from src.sentiment_analyzer import FinBERTSentimentAnalyzer

                st.session_state["finbert_analyzer"] = FinBERTSentimentAnalyzer()
            except Exception as exc:
                st.error(f"❌ Failed to load FinBERT model: {exc}")
                return None
    return st.session_state["finbert_analyzer"]


def _run_analysis(texts: list[str]) -> pd.DataFrame | None:
    """Analyse *texts* and return the results DataFrame, or None on failure."""
    analyzer = _get_analyzer()
    if analyzer is None:
        return None
    try:
        with st.spinner("🔍 Analysing sentiment…"):
            return analyzer.analyze_texts(texts)
    except Exception as exc:
        st.error(f"❌ Sentiment analysis failed: {exc}")
        return None


def _probability_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing averaged Positive / Neutral / Negative probabilities."""
    avg_pos = float(df["Positive_Prob"].mean())
    avg_neu = float(df["Neutral_Prob"].mean())
    avg_neg = float(df["Negative_Prob"].mean())

    categories = ["Positive", "Neutral", "Negative"]
    values = [avg_pos, avg_neu, avg_neg]
    bar_colors = [COLORS["green"], COLORS["amber"], COLORS["red"]]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in values],
            textposition="auto",
            textfont=dict(size=13, color=COLORS["white"]),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=PLOTLY_BG,
        paper_bgcolor=PLOTLY_PAPER_BG,
        font=dict(family="Inter, sans-serif", color=COLORS["white"], size=12),
        title=dict(
            text="Average Probability Breakdown",
            font=dict(size=16, color=COLORS["white"]),
            x=0.01,
        ),
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(42, 48, 66, 0.5)",
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, zeroline=False, categoryorder="array",
                    categoryarray=["Negative", "Neutral", "Positive"]),
        bargap=0.35,
    )
    return fig


def _display_results(df: pd.DataFrame) -> None:
    """Render gauge, probability chart, and detailed results table."""

    avg_score = float(df["Score"].mean())

    # ── Row 1: Gauge + Probability Breakdown ──────────────────────────
    col_gauge, col_bar = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(
            sentiment_gauge(avg_score, label="Overall Sentiment"),
            use_container_width=True,
            key=f"gauge_{id(df)}",
        )

    with col_bar:
        st.plotly_chart(
            _probability_bar_chart(df),
            use_container_width=True,
            key=f"prob_bar_{id(df)}",
        )

    # ── Row 2: Detailed Results Table ─────────────────────────────────
    st.markdown("#### 📋 Detailed Results")

    display_df = df.copy()
    display_df["Score"] = display_df["Score"].map("{:+.3f}".format)
    display_df["Confidence"] = display_df["Confidence"].map("{:.1%}".format)
    display_df["Positive_Prob"] = display_df["Positive_Prob"].map("{:.1%}".format)
    display_df["Negative_Prob"] = display_df["Negative_Prob"].map("{:.1%}".format)
    display_df["Neutral_Prob"] = display_df["Neutral_Prob"].map("{:.1%}".format)

    display_df = display_df.rename(
        columns={
            "Positive_Prob": "P(Pos)",
            "Negative_Prob": "P(Neg)",
            "Neutral_Prob": "P(Neu)",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 38 + 35 * len(display_df)),
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:  # noqa: C901
    """Entry-point called by the app router to render the Sentiment page."""

    st.header("📰 News Sentiment Analysis")
    st.caption(
        "Powered by **FinBERT** — enter financial headlines or earnings-call "
        "snippets and receive an instant sentiment breakdown."
    )

    st.divider()

    # ── Section 1: Free-Text Analysis ────────────────────────────────
    st.subheader("✏️ Custom Text Analysis")

    user_text = st.text_area(
        "Paste headlines or summaries (one per line)",
        height=180,
        placeholder=(
            "Apple reports record Q4 revenue of $94.9 billion, up 8% YoY\n"
            "Inflation fears rattle markets as CPI exceeds expectations\n"
            "Amazon announces $10 billion share buyback programme"
        ),
    )

    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        texts = [line.strip() for line in user_text.splitlines() if line.strip()]
        if not texts:
            st.warning("⚠️ Please enter at least one headline or summary above.")
        else:
            results = _run_analysis(texts)
            if results is not None and not results.empty:
                st.session_state["custom_results"] = results

    # Persist results across reruns
    if "custom_results" in st.session_state:
        _display_results(st.session_state["custom_results"])

    st.divider()

    # ── Section 2: Sample / Batch Analysis ───────────────────────────
    st.subheader("📦 Sample Batch Analysis")
    st.markdown("Try the model instantly with these curated financial headlines:")

    with st.expander("View sample headlines", expanded=True):
        for i, headline in enumerate(_SAMPLE_HEADLINES, 1):
            st.markdown(f"**{i}.** {headline}")

    if st.button("⚡ Run Sample Analysis", use_container_width=True):
        results = _run_analysis(_SAMPLE_HEADLINES)
        if results is not None and not results.empty:
            st.session_state["sample_results"] = results

    if "sample_results" in st.session_state:
        _display_results(st.session_state["sample_results"])
