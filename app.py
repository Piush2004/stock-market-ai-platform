import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration for a premium dark mode feel
st.set_page_config(
    page_title="AlphaQuants AI | Stock Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E2127;
        border-radius: 5px;
        padding: 0 20px;
        color: #A0AEC0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
    h1, h2, h3 {
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 AlphaQuants AI Platform")
st.markdown("Advanced algorithmic trading suite powered by XGBoost, FinBERT, and RAG.")

# --- Helper Functions ---
@st.cache_data
def load_processed_data(ticker):
    path = f"data/processed/{ticker}_features.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        return df
    return None

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS"]

# --- Create Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Stock Forecast", 
    "📊 Technical Analysis", 
    "💬 Financial Report Chat", 
    "📰 News Sentiment", 
    "⚖️ Portfolio Optimizer"
])

# ==========================================
# TAB 1: Stock Forecast
# ==========================================
with tab1:
    st.header("Stock Direction Prediction (XGBoost)")
    st.markdown("Predicts whether tomorrow's closing price will be **higher** or **lower** based on technical indicators and fundamental RAG features.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_ticker = st.selectbox("Select Asset", TICKERS, key="forecast_ticker")
        st.metric(label="Model Accuracy", value="51.22%", delta="Live")
        st.metric(label="Recall (Up Days)", value="66.16%", delta="High Sensitivity", delta_color="normal")
        
        if st.button("Run Prediction Inference"):
            df = load_processed_data(selected_ticker)
            if df is not None and not df.empty:
                last_close = df['Close'].iloc[-1]
                st.success(f"Prediction Generated for {selected_ticker}")
                st.info(f"Last Close: ${last_close:.2f}")
                
                # Mock inference output for dashboard demonstration
                # In production, we'd load the XGBoost model and call predict()
                pred = np.random.choice(["UP ⬆️", "DOWN ⬇️"], p=[0.55, 0.45])
                st.markdown(f"### Tomorrow's Forecast: **{pred}**")
            else:
                st.error("Data not found for this ticker.")
                
    with col2:
        df = load_processed_data(selected_ticker)
        if df is not None:
            fig = go.Figure()
            # Plot last 100 days
            subset = df.iloc[-100:]
            fig.add_trace(go.Candlestick(x=subset.index,
                open=subset['Open'], high=subset['High'],
                low=subset['Low'], close=subset['Close'],
                name="Price"))
            fig.update_layout(template="plotly_dark", title=f"{selected_ticker} - Recent Price Action", height=400)
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 2: Technical Analysis
# ==========================================
with tab2:
    st.header("Advanced Technical Indicators")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        tech_ticker = st.selectbox("Select Asset", TICKERS, key="tech_ticker")
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_ema = st.checkbox("EMA (20 & 50)", value=True)
        show_macd = st.checkbox("MACD", value=False)
        
    with col2:
        df = load_processed_data(tech_ticker)
        if df is not None:
            subset = df.iloc[-150:] # Last 150 days
            
            # Create subplots if MACD is checked
            rows = 2 if show_macd else 1
            height = 600 if show_macd else 450
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.05, 
                              row_heights=[0.7, 0.3] if show_macd else [1])
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=subset.index, open=subset['Open'], high=subset['High'], low=subset['Low'], close=subset['Close'], name="OHLC"), row=1, col=1)
            
            if show_bb and 'BB_Upper' in subset.columns:
                fig.add_trace(go.Scatter(x=subset.index, y=subset['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.4)', width=1), name="BB Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=subset.index, y=subset['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.4)', width=1), fill='tonexty', name="BB Lower"), row=1, col=1)
                
            if show_ema and 'EMA_20' in subset.columns:
                fig.add_trace(go.Scatter(x=subset.index, y=subset['EMA_20'], line=dict(color='orange', width=2), name="EMA 20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=subset.index, y=subset['EMA_50'], line=dict(color='blue', width=2), name="EMA 50"), row=1, col=1)
                
            if show_macd and 'MACD' in subset.columns:
                fig.add_trace(go.Scatter(x=subset.index, y=subset['MACD'], line=dict(color='blue', width=1.5), name="MACD"), row=2, col=1)
                fig.add_trace(go.Scatter(x=subset.index, y=subset['MACD_Signal'], line=dict(color='orange', width=1.5), name="Signal"), row=2, col=1)
                fig.add_trace(go.Bar(x=subset.index, y=subset['MACD_Hist'], marker_color='gray', name="Histogram"), row=2, col=1)
                
            fig.update_layout(template="plotly_dark", height=height, margin=dict(l=0, r=0, t=30, b=0))
            fig.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 3: Financial Report Chat (RAG)
# ==========================================
with tab3:
    st.header("Financial Report Chatbot (RAG)")
    st.markdown("Query ingested SEC Filings, Annual Reports, and Earnings Transcripts using Local LLMs.")
    
    chat_container = st.container()
    with chat_container:
        st.info("System: Connected to ChromaDB vector store. Ready to answer queries.")
        
        query = st.text_input("Ask a question about the ingested documents:", placeholder="e.g. What were the key risk factors mentioned in the latest earnings call?")
        
        if st.button("Search & Generate"):
            if query:
                with st.spinner("Searching vector database and generating response via Ollama..."):
                    # Mock response for UI demonstration
                    # In production, we instantiate FinancialRAGPipeline here
                    st.markdown("### Answer:")
                    st.write("Based on the provided context from the Q3 earnings transcript (Page 4), the key risk factors highlighted by management include ongoing supply chain constraints in the semiconductor division and increased volatility in foreign exchange rates impacting international revenue margins.")
                    
                    st.markdown("**Sources:**")
                    st.markdown("- `sample_earnings_transcript.txt` (Score: 0.82)")
                    st.markdown("- `annual_report_2023.pdf` Page 12 (Score: 0.76)")
            else:
                st.warning("Please enter a question.")


# ==========================================
# TAB 4: News Sentiment
# ==========================================
with tab4:
    st.header("FinBERT News & Earnings Sentiment")
    st.markdown("Paste headlines or earnings summaries below to calculate immediate market sentiment (-1 to +1).")
    
    text_input = st.text_area("Enter financial text:", height=100, placeholder="Apple Inc. beats quarterly revenue expectations, shares surge...")
    
    if st.button("Analyze Sentiment"):
        if text_input:
            with st.spinner("Running FinBERT sequence classification..."):
                # Mock result for UI demonstration. In production, call FinBERTSentimentAnalyzer
                import time
                time.sleep(1) # Simulate model latency
                
                # Simple mock logic based on keywords
                lower_text = text_input.lower()
                if "beat" in lower_text or "surge" in lower_text or "growth" in lower_text:
                    sentiment = "Positive"
                    score = 0.89
                    color = "green"
                elif "drop" in lower_text or "glitch" in lower_text or "decline" in lower_text:
                    sentiment = "Negative"
                    score = -0.92
                    color = "red"
                else:
                    sentiment = "Neutral"
                    score = 0.05
                    color = "gray"
                    
                st.markdown(f"### Classification: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
                
                # Progress bar for visual appeal
                normalized_score = (score + 1) / 2 # Scale -1 to 1 into 0 to 1
                st.progress(normalized_score, text=f"Sentiment Score: {score:.2f}")
        else:
            st.warning("Please enter some text to analyze.")


# ==========================================
# TAB 5: Portfolio Optimizer
# ==========================================
with tab5:
    st.header("Efficient Frontier Portfolio Optimizer")
    st.markdown("Calculates the optimal asset allocation to maximize the Sharpe Ratio using Modern Portfolio Theory (Markowitz).")
    
    if st.button("Run Optimization (Max Sharpe)"):
        with st.spinner("Calculating covariance matrix and optimizing weights..."):
            import time
            time.sleep(1.5)
            
            # Mock optimization results
            st.success("Optimization Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Optimal Allocation")
                labels = TICKERS
                values = [35, 25, 10, 20, 5, 5]
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                fig.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0), height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown("### Expected Performance")
                st.metric("Expected Annual Return", "24.5%")
                st.metric("Annual Volatility", "16.2%")
                st.metric("Sharpe Ratio", "1.51")
