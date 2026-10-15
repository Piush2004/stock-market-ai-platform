"""
Portfolio Optimizer Page
========================
Mean-variance portfolio optimization powered by PyPortfolioOpt.
Displays optimal allocation, performance metrics, and efficient frontier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.components.data_loader import TICKERS, TICKER_NAMES, load_raw_data
from dashboard.components.charts import portfolio_donut, COLORS, PLOTLY_BG, PLOTLY_PAPER_BG, PLOTLY_GRID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_close_prices(tickers: list[str]) -> pd.DataFrame:
    """Load adjusted close prices for the given tickers into a single DataFrame."""
    frames: dict[str, pd.Series] = {}
    for t in tickers:
        df = load_raw_data(t)
        if df.empty:
            continue
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        frames[t] = df[col]
    if not frames:
        return pd.DataFrame()
    prices = pd.DataFrame(frames).dropna()
    return prices


def _simulate_random_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 3_000,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Generate random portfolio weight combinations and compute return / risk / Sharpe."""
    n_assets = len(mean_returns)
    results = np.zeros((n_portfolios, 3))
    rng = np.random.default_rng(42)

    for i in range(n_portfolios):
        weights = rng.random(n_assets)
        weights /= weights.sum()

        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

        results[i] = [port_return, port_vol, sharpe]

    return pd.DataFrame(results, columns=["Return", "Volatility", "Sharpe"])


def _build_efficient_frontier_chart(
    sim_df: pd.DataFrame,
    opt_return: float,
    opt_vol: float,
) -> go.Figure:
    """Scatter plot of simulated portfolios coloured by Sharpe, with the optimal point highlighted."""
    fig = go.Figure()

    # Random portfolios
    fig.add_trace(go.Scatter(
        x=sim_df["Volatility"],
        y=sim_df["Return"],
        mode="markers",
        marker=dict(
            size=4,
            color=sim_df["Sharpe"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Sharpe",
                titlefont=dict(color=COLORS["white"], size=11),
                tickfont=dict(color=COLORS["gray"], size=10),
            ),
            opacity=0.65,
        ),
        name="Random Portfolios",
        hovertemplate="Return: %{y:.2%}<br>Volatility: %{x:.2%}<br><extra></extra>",
    ))

    # Optimal point
    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_return],
        mode="markers+text",
        marker=dict(size=16, color=COLORS["amber"], symbol="star", line=dict(width=2, color=COLORS["white"])),
        text=["Optimal"],
        textposition="top center",
        textfont=dict(color=COLORS["amber"], size=13, family="Inter"),
        name="Optimal Portfolio",
        hovertemplate="Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=PLOTLY_BG,
        paper_bgcolor=PLOTLY_PAPER_BG,
        font=dict(family="Inter, sans-serif", color=COLORS["white"], size=12),
        title=dict(text="Efficient Frontier", font=dict(size=16, color=COLORS["white"]), x=0.01),
        height=480,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=COLORS["white"])),
        xaxis=dict(
            title="Annual Volatility",
            tickformat=".1%",
            gridcolor=PLOTLY_GRID,
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Expected Annual Return",
            tickformat=".1%",
            gridcolor=PLOTLY_GRID,
            showgrid=True,
            zeroline=False,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> None:
    """Main entry-point called by the Streamlit router."""

    st.markdown("## 📊 Portfolio Optimizer")
    st.caption("Mean-variance optimisation · Powered by PyPortfolioOpt")

    # ------------------------------------------------------------------
    # Check dependency early so the rest of the page can reference it
    # ------------------------------------------------------------------
    try:
        from pypfopt import EfficientFrontier, risk_models, expected_returns  # noqa: F401
        _pypfopt_available = True
    except ImportError:
        _pypfopt_available = False
        st.error(
            "⚠️ **PyPortfolioOpt is not installed.**  \n"
            "Run `pip install pyportfolioopt` and restart the app."
        )
        return

    # ------------------------------------------------------------------
    # Layout: sidebar controls (left) | results (right)
    # ------------------------------------------------------------------
    col_controls, col_results = st.columns([1, 3], gap="large")

    with col_controls:
        st.markdown("### ⚙️ Parameters")

        investment = st.number_input(
            "💰 Investment Amount ($)",
            min_value=1_000,
            value=10_000,
            step=1_000,
            format="%d",
        )

        risk = st.slider(
            "🎯 Risk Tolerance",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = Conservative · 10 = Aggressive",
        )
        risk_label = (
            "🟢 Conservative" if risk < 4
            else "🔴 Aggressive" if risk > 7
            else "🟡 Moderate"
        )
        st.markdown(f"**Profile:** {risk_label}")

        selected_tickers = st.multiselect(
            "📈 Select Tickers",
            options=TICKERS,
            default=TICKERS,
            format_func=lambda t: f"{t} — {TICKER_NAMES.get(t, t)}",
        )

        optimize_clicked = st.button("🚀 Optimize Portfolio", use_container_width=True, type="primary")

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    with col_results:
        if not optimize_clicked:
            st.info("Configure parameters on the left and click **Optimize Portfolio** to begin.", icon="👈")
            return

        if len(selected_tickers) < 2:
            st.warning("Please select at least **2 tickers** for meaningful diversification.", icon="⚠️")
            return

        with st.spinner("Loading market data & running optimisation …"):
            # ---- data ------------------------------------------------
            prices = _load_close_prices(selected_tickers)

            if prices.empty or prices.shape[0] < 60:
                st.warning(
                    "Insufficient price data.  \n"
                    "Make sure raw CSV files exist for the selected tickers and contain ≥ 60 trading days.",
                    icon="⚠️",
                )
                return

            if prices.shape[1] < 2:
                st.warning("Price data found for fewer than 2 tickers — cannot optimise.", icon="⚠️")
                return

            # ---- returns & covariance --------------------------------
            try:
                mu = expected_returns.mean_historical_return(prices)
                cov = risk_models.sample_cov(prices)
            except Exception as exc:
                st.error(f"Error computing return statistics: {exc}")
                return

            # ---- optimise --------------------------------------------
            try:
                ef = EfficientFrontier(mu, cov)

                if risk > 7:
                    # Aggressive → maximise Sharpe ratio
                    weights = ef.max_sharpe()
                elif risk < 4:
                    # Conservative → minimise volatility
                    weights = ef.min_volatility()
                else:
                    # Moderate → target a risk level scaled between min-vol and max-Sharpe vol
                    ef_temp = EfficientFrontier(mu, cov)
                    ef_temp.min_volatility()
                    perf_min = ef_temp.portfolio_performance()
                    min_vol = perf_min[1]

                    ef_temp2 = EfficientFrontier(mu, cov)
                    ef_temp2.max_sharpe()
                    perf_max = ef_temp2.portfolio_performance()
                    max_vol = perf_max[1]

                    # Interpolate target volatility based on risk slider (4-7 → 0.0-1.0)
                    t = (risk - 4) / 3
                    target_vol = min_vol + t * (max_vol - min_vol)
                    target_vol = max(target_vol, min_vol + 1e-6)

                    weights = ef.efficient_risk(target_vol)

                cleaned = ef.clean_weights(cutoff=0.01)
                perf = ef.portfolio_performance(verbose=False)
            except Exception as exc:
                st.error(f"Optimisation failed: {exc}")
                return

        # ------------------------------------------------------------------
        # Extract results
        # ------------------------------------------------------------------
        exp_return, annual_vol, sharpe = perf
        weight_series = pd.Series(cleaned).loc[lambda s: s > 0]

        if weight_series.empty:
            st.warning("All weights were rounded to zero — try selecting different tickers.", icon="⚠️")
            return

        alloc_labels = [f"{t}" for t in weight_series.index]
        alloc_values = [round(w * 100, 2) for w in weight_series.values]

        # ------------------------------------------------------------------
        # Row 1: Donut + Metrics
        # ------------------------------------------------------------------
        st.markdown("---")
        c1, c2 = st.columns([1, 1], gap="medium")

        with c1:
            st.markdown("#### Optimal Allocation")
            fig_donut = portfolio_donut(alloc_labels, alloc_values, title="Weight Distribution (%)")
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.markdown("#### Performance Snapshot")
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Return", f"{exp_return:.2%}")
            m2.metric("Annual Volatility", f"{annual_vol:.2%}")
            m3.metric("Sharpe Ratio", f"{sharpe:.2f}")

            # ----------------------------------------------------------
            # Allocation table
            # ----------------------------------------------------------
            st.markdown("#### Allocation Breakdown")
            table_data = {
                "Ticker": [],
                "Company": [],
                "Weight (%)": [],
                "Amount ($)": [],
            }
            for ticker, weight in weight_series.items():
                table_data["Ticker"].append(ticker)
                table_data["Company"].append(TICKER_NAMES.get(ticker, ticker))
                table_data["Weight (%)"].append(f"{weight * 100:.2f}")
                table_data["Amount ($)"].append(f"{investment * weight:,.2f}")

            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True,
            )

        # ------------------------------------------------------------------
        # Row 2: Efficient Frontier
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### Efficient Frontier")

        with st.spinner("Simulating random portfolios …"):
            try:
                daily_returns = prices.pct_change().dropna()
                mean_daily = daily_returns.mean() * 252
                cov_annual = daily_returns.cov() * 252

                sim_df = _simulate_random_portfolios(mean_daily, cov_annual)
                fig_ef = _build_efficient_frontier_chart(sim_df, exp_return, annual_vol)
                st.plotly_chart(fig_ef, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render efficient frontier: {exc}", icon="⚠️")
