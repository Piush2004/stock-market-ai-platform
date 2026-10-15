import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG = "rgba(10, 14, 23, 0)"
PLOTLY_PAPER_BG = "rgba(10, 14, 23, 0)"
PLOTLY_GRID = "rgba(42, 48, 66, 0.5)"

COLORS = {
    "blue": "#3B82F6",
    "cyan": "#06B6D4",
    "green": "#10B981",
    "red": "#EF4444",
    "amber": "#F59E0B",
    "purple": "#8B5CF6",
    "gray": "#64748B",
    "white": "#F1F5F9",
}

def _base_layout(title: str = "", height: int = 450) -> dict:
    """Returns a base layout dict for consistent chart styling."""
    return dict(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=PLOTLY_BG,
        paper_bgcolor=PLOTLY_PAPER_BG,
        font=dict(family="Inter, sans-serif", color=COLORS["white"], size=12),
        title=dict(text=title, font=dict(size=16, color=COLORS["white"]), x=0.01),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=PLOTLY_GRID, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=PLOTLY_GRID, showgrid=True, zeroline=False),
    )


def candlestick_chart(df: pd.DataFrame, title: str = "Price Action", days: int = 120) -> go.Figure:
    """Creates a candlestick chart with volume subplot."""
    subset = df.iloc[-days:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.75, 0.25])
    
    fig.add_trace(go.Candlestick(
        x=subset.index, open=subset["Open"], high=subset["High"],
        low=subset["Low"], close=subset["Close"], name="OHLC",
        increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"]
    ), row=1, col=1)
    
    # Volume bars colored by direction
    colors = [COLORS["green"] if c >= o else COLORS["red"] 
              for c, o in zip(subset["Close"], subset["Open"])]
    fig.add_trace(go.Bar(
        x=subset.index, y=subset["Volume"], marker_color=colors,
        opacity=0.5, name="Volume", showlegend=False
    ), row=2, col=1)
    
    layout = _base_layout(title, height=500)
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor=PLOTLY_GRID)
    fig.update_yaxes(title_text="Vol", row=2, col=1, gridcolor=PLOTLY_GRID)
    return fig


def technical_chart(df: pd.DataFrame, indicators: dict, days: int = 150) -> go.Figure:
    """
    Creates a technical analysis chart with selectable overlays.
    indicators: dict of booleans, e.g. {"ema": True, "bb": True, "macd": False}
    """
    subset = df.iloc[-days:]
    show_macd = indicators.get("macd", False)
    show_rsi = indicators.get("rsi", False)
    
    n_rows = 1 + int(show_macd) + int(show_rsi)
    heights = [0.6]
    if show_macd: heights.append(0.2)
    if show_rsi: heights.append(0.2)
    
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=heights)
    
    # Candlestick base
    fig.add_trace(go.Candlestick(
        x=subset.index, open=subset["Open"], high=subset["High"],
        low=subset["Low"], close=subset["Close"], name="OHLC",
        increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"]
    ), row=1, col=1)
    
    # Bollinger Bands
    if indicators.get("bb") and "BB_Upper" in subset.columns:
        fig.add_trace(go.Scatter(
            x=subset.index, y=subset["BB_Upper"], mode="lines",
            line=dict(color=COLORS["cyan"], width=1, dash="dot"), name="BB Upper"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=subset.index, y=subset["BB_Lower"], mode="lines",
            line=dict(color=COLORS["cyan"], width=1, dash="dot"), name="BB Lower",
            fill="tonexty", fillcolor="rgba(6, 182, 212, 0.06)"
        ), row=1, col=1)
    
    # EMAs
    if indicators.get("ema"):
        if "EMA_20" in subset.columns:
            fig.add_trace(go.Scatter(
                x=subset.index, y=subset["EMA_20"], mode="lines",
                line=dict(color=COLORS["amber"], width=2), name="EMA 20"
            ), row=1, col=1)
        if "EMA_50" in subset.columns:
            fig.add_trace(go.Scatter(
                x=subset.index, y=subset["EMA_50"], mode="lines",
                line=dict(color=COLORS["purple"], width=2), name="EMA 50"
            ), row=1, col=1)
    
    # MACD
    current_row = 2
    if show_macd and "MACD" in subset.columns:
        hist_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in subset["MACD_Hist"]]
        fig.add_trace(go.Bar(x=subset.index, y=subset["MACD_Hist"], marker_color=hist_colors,
                             name="MACD Hist", opacity=0.6), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=subset.index, y=subset["MACD"], mode="lines",
                                 line=dict(color=COLORS["blue"], width=1.5), name="MACD"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=subset.index, y=subset["MACD_Signal"], mode="lines",
                                 line=dict(color=COLORS["amber"], width=1.5), name="Signal"), row=current_row, col=1)
        current_row += 1
    
    # RSI
    if show_rsi and "RSI_14" in subset.columns:
        fig.add_trace(go.Scatter(x=subset.index, y=subset["RSI_14"], mode="lines",
                                 line=dict(color=COLORS["purple"], width=2), name="RSI 14"), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.5, row=current_row, col=1)
    
    layout = _base_layout("Technical Analysis", height=200 + n_rows * 200)
    layout["xaxis_rangeslider_visible"] = False
    fig.update_layout(**layout)
    return fig


def sentiment_gauge(score: float, label: str = "Sentiment") -> go.Figure:
    """Creates a semicircular gauge chart for sentiment score (-1 to 1)."""
    color = COLORS["green"] if score > 0.2 else COLORS["red"] if score < -0.2 else COLORS["amber"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=36, color=COLORS["white"]), valueformat="+.2f"),
        title=dict(text=label, font=dict(size=14, color=COLORS["gray"])),
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor=COLORS["gray"], dtick=0.5,
                      tickfont=dict(size=10, color=COLORS["gray"])),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(26, 31, 46, 0.8)",
            borderwidth=0,
            steps=[
                dict(range=[-1, -0.2], color="rgba(239, 68, 68, 0.15)"),
                dict(range=[-0.2, 0.2], color="rgba(245, 158, 11, 0.10)"),
                dict(range=[0.2, 1], color="rgba(16, 185, 129, 0.15)"),
            ],
        )
    ))
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG, font=dict(family="Inter", color=COLORS["white"]),
        height=250, margin=dict(l=30, r=30, t=50, b=10)
    )
    return fig


def portfolio_donut(labels: list, values: list, title: str = "Allocation") -> go.Figure:
    """Creates a donut chart for portfolio allocation."""
    palette = [COLORS["blue"], COLORS["cyan"], COLORS["green"], COLORS["amber"], COLORS["purple"], COLORS["red"]]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=palette[:len(labels)], line=dict(color=PLOTLY_BG, width=2)),
        textfont=dict(size=12, color=COLORS["white"]),
        textinfo="label+percent", hoverinfo="label+value+percent"
    )])
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG, plot_bgcolor=PLOTLY_BG, 
        font=dict(family="Inter", color=COLORS["white"]),
        height=380, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["white"], size=11)),
        title=dict(text=title, font=dict(size=14), x=0.5)
    )
    return fig


def sparkline(series: pd.Series, color: str = None, height: int = 80) -> go.Figure:
    """Creates a tiny sparkline chart for inline metrics."""
    if color is None:
        color = COLORS["green"] if series.iloc[-1] >= series.iloc[0] else COLORS["red"]
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba") if "rgb" in color else f"{color}14"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    return fig
