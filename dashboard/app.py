import os
import sys
import streamlit as st

# Ensure project root is on the path so all imports resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.pages import dashboard_page, forecast_page, chat_page, sentiment_page, portfolio_page

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AlphaQuants AI | Financial Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== LOAD CUSTOM CSS =====
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===== SIDEBAR NAVIGATION =====
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0 1.5rem 0;'>
            <span style='font-size: 2.5rem;'>🧠</span>
            <h2 style='margin: 0.3rem 0 0 0; font-size: 1.3rem; 
                        background: linear-gradient(135deg, #3B82F6, #8B5CF6); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        font-weight: 800;'>AlphaQuants AI</h2>
            <p style='color: #64748B; font-size: 0.75rem; margin: 0.2rem 0 0 0; 
                      letter-spacing: 0.1em; text-transform: uppercase;'>Financial Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "NAVIGATION",
        options=[
            "📊 Dashboard",
            "🚀 Stock Forecast",
            "💬 Financial Chat",
            "📰 News Sentiment",
            "⚖️ Portfolio Optimizer",
        ],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    
    # System info footer
    st.markdown("""
        <div style='position: fixed; bottom: 1rem; padding: 0.5rem; font-size: 0.7rem; color: #64748B;'>
            <div>v1.0.0 · Built with Streamlit</div>
            <div style='margin-top: 0.2rem;'>XGBoost · FinBERT · ChromaDB · Ollama</div>
        </div>
    """, unsafe_allow_html=True)

# ===== PAGE ROUTING =====
if page == "📊 Dashboard":
    dashboard_page.render()
elif page == "🚀 Stock Forecast":
    forecast_page.render()
elif page == "💬 Financial Chat":
    chat_page.render()
elif page == "📰 News Sentiment":
    sentiment_page.render()
elif page == "⚖️ Portfolio Optimizer":
    portfolio_page.render()
