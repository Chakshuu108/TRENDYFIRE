"""
Main Streamlit Application Entry Point
Orchestrates all pages and manages session state
"""

import streamlit as st
from datetime import datetime
import sys
import os

# Import all page modules
from home_page import show_home_page
from data_input_page import show_data_input_page
from nlp_analysis_page import show_nlp_analysis_page
from forecasting_page import show_forecasting_page
from insights_page import show_insights_page
from analytics_page import show_analytics_page
from custom_bi_page import show_custom_bi_page

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="TrendyFire",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6366f1;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(99,102,241,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #6366f1;
        box-shadow: 0 4px 15px rgba(99,102,241,0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #ffffff;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.2);
    }
    
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data_loaded': False,
        'scraped_data': None,
        'demand_data': None,
        'processed_data': None,
        'model_trained': False,
        'forecast_data': None,
        'best_model': None,
        'metrics': None,
        'forecast_values': None,
        'forecast_dates': None,
        'model_name': None,
        'multi_graph_mode': False,
        'num_multi_graphs': 2,
        'original_csv_data': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# ============================================================================
# SIDEBAR
# ============================================================================

def show_sidebar():
    """Display sidebar with status and quick actions"""
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        
        st.markdown("---")
        
        st.markdown("#### 📈 Data Status")
        if st.session_state.demand_data is not None:
            st.success(f"✅ Demand Data: {len(st.session_state.demand_data)} rows")
        else:
            st.warning("⚠️ No demand data")
        
        if st.session_state.scraped_data is not None:
            st.success(f"✅ Review Data: {len(st.session_state.scraped_data)} rows")
        else:
            st.warning("⚠️ No review data")
        
        st.markdown("---")
        
        st.markdown("#### 🤖 Model Status")
        if st.session_state.model_trained:
            st.success("✅ Model trained")
            if st.session_state.model_name:
                st.info(f"📊 Model: {st.session_state.model_name}")
            if st.session_state.metrics:
                st.metric("RMSE", f"{st.session_state.metrics['RMSE']:.2f}")
                st.metric("R² Score", f"{st.session_state.metrics['R2']:.4f}")
        else:
            st.info("⏳ No model trained yet")
        
        st.markdown("---")
        
        st.markdown("#### 🎯 Quick Actions")
        
        if st.session_state.processed_data is not None:
            csv = st.session_state.processed_data.to_csv(index=False)
            st.download_button(
                "📥 Download Processed Data",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.session_state.forecast_data is not None:
            csv = st.session_state.forecast_data.to_csv(index=False)
            st.download_button(
                "📥 Download Forecast",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
  
# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Show sidebar
    show_sidebar()
    
    # Main header
    st.markdown('<div class="main-header">TrendiFY</div>', unsafe_allow_html=True)
    st.markdown("### Predict future demand using customer sentiment and feedback")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Home",
        "📊 Data Input",
        "🔍 NLP Analysis",
        "📈 Forecasting",
        "💡 Insights",
        "📊 Analytics",
        "🎨 Custom BI"
    ])
    
    with tab1:
        show_home_page()
    
    with tab2:
        show_data_input_page()
    
    with tab3:
        show_nlp_analysis_page()
    
    with tab4:
        show_forecasting_page()
    
    with tab5:
        show_insights_page()
    
    with tab6:
        show_analytics_page()

    with tab7:
        show_custom_bi_page()


if __name__ == "__main__":
    main()
