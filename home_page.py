"""
Home Page Module
Displays feature overview and application capabilities
"""

import streamlit as st


def show_home_page():
    """Display home page with 6 feature boxes"""
    
    # Add vertical spacing to ensure content starts below tabs
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    # FIRST ROW: Main 3 feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🌐 Web Scraping")
        st.write("Collect customer reviews and feedback from any URL automatically")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 NLP Analysis")
        st.write("Analyze sentiment, extract topics, and identify key themes from text data")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Forecasting")
        st.write("Predict future demand using 5 ML models including XGBoost and Gradient Boosting")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacing between rows
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # SECOND ROW: Additional 3 feature cards
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Insights")
        st.write("Generate recommendations and PDF reports with visualizations")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Analytics")
        st.write("Dive into demand statistics, correlation analysis, model comparison, and feature importance")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 Custom BI")
        st.write("Build custom visualizations with intelligent column detection and 11+ chart types")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature details section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### 🔹 Data Collection
        - **Upload historical demand data** (any CSV format)
        - **Scrape customer reviews** from web URLs
        - **Smart column detection** handles various formats automatically
        - **Supports multiple data sources** (CSV, web scraping, manual upload)
        
        #### 🔹 NLP Processing
        - **Text preprocessing** and cleaning
        - **Sentiment analysis** (VADER/TextBlob methods)
        - **Topic modeling** to identify key themes and patterns
        - **Extract insights** from customer feedback and reviews
        
        #### 🔹 Advanced Analytics
        - **Comprehensive statistical analysis** of demand patterns
        - **Correlation & relationship discovery** between features
        - **Feature importance visualization** to understand drivers
        - **Time-series decomposition** (trend, seasonality, residuals)
        """)
    
    with col2:
        st.markdown("""
        #### 🔹 Feature Engineering
        - **Create time-based features** (day of week, month, seasonality)
        - **Generate lag features** (previous periods' demand)
        - **Calculate rolling statistics** (moving averages, trends)
        - **Automated feature selection** for optimal performance
        
        #### 🔹 Model Training & Forecasting
        - **5 ML Models**: XGBoost, Gradient Boosting, Random Forest, Linear Regression, SVR
        - **Iterative forecasting** - realistic predictions over time horizon
        - **Confidence intervals** - understand prediction uncertainty
        - **Performance metrics**: MAE, RMSE, MAPE, R² Score
        
        #### 🔹 Insights & Reporting
        - **Strategic recommendations** based on analysis
        - **Multiple visualization types** (8+ chart types available)
        - **Export capabilities** - download all data and charts
        - **Custom BI Builder** - create your own visualizations
        """)
    
    st.markdown("---")
    
    # Visualization types
    st.markdown("### 📊 8 Professional Visualization Types")
    
    viz_col1, viz_col2, viz_col3, viz_col4 = st.columns(4)
    
    with viz_col1:
        st.info("📈 **Line Chart**")
    with viz_col2:
        st.info("📊 **Bar Chart**")
    with viz_col3:
        st.info("🎨 **Area Chart**")
    with viz_col4:
        st.info("🔵 **Scatter Plot**")
    
    viz_col5, viz_col6, viz_col7, viz_col8 = st.columns(4)
    
    with viz_col5:
        st.info("🕯️ **Candlestick**")
    with viz_col6:
        st.info("🔥 **Heatmap**")
    with viz_col7:
        st.info("📦 **Box Plot**")
    with viz_col8:
        st.info("🎻 **Violin Plot**")
    
    st.markdown("---")
    
    # Key Features Highlights
    st.markdown("### ⭐ Key Features & Capabilities")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        **🎯 Forecasting Excellence**
        - 5 ML models included
        - Automatic best model selection
        - Up to 90-day forecasts
        - Confidence intervals
        - Iterative prediction engine
        - Real-time performance metrics
        """)
    
    with feature_col2:
        st.markdown("""
        **🧠 NLP Intelligence**
        - Sentiment analysis (2 methods)
        - Topic modeling & extraction
        - Text preprocessing pipeline
        - Aggregate insights by date
        - Review impact on demand
        - Automated text cleaning
        """)
    
    with feature_col3:
        st.markdown("""
        **📊 Professional Reporting**
        - PDF reports with graphs
        - Multiple chart types
        - Strategic recommendations
        - Executive summaries
        - Download all outputs
        - Share-ready formats
        """)
        
    
    st.markdown("---")
    
    
    