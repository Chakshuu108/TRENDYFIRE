"""
Analytics Page Module
Advanced statistical analysis, model comparison, and feature importance
Part 1 of 2: This file contains the main analytics page and helper functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def show_analytics_page():
    """Advanced Analytics & Comparison page"""
    
    st.markdown("### 📊 Advanced Data Analytics & Model Comparison")
    
    # Check if we have data
    has_demand = st.session_state.demand_data is not None
    has_reviews = st.session_state.scraped_data is not None
    has_model = st.session_state.model_trained
    
    if not has_demand and not has_reviews:
        st.warning("⚠️ Please upload demand data or review data first!")
        return
    
    # Create sub-tabs for different analytics
    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
        "📈 Demand Statistics",
        "🔍 Correlation Analysis", 
        "🎯 Model Comparison",
        "💎 Feature Importance"
    ])
    
    # Tab 1: Demand Statistics
    with analytics_tab1:
        show_demand_statistics_tab(has_demand)
    
    # Tab 2: Correlation Analysis
    with analytics_tab2:
        show_correlation_analysis_tab(has_reviews, has_demand)
    
    # Tab 3: Model Comparison
    with analytics_tab3:
        show_model_comparison_tab(has_model)
    
    # Tab 4: Feature Importance
    with analytics_tab4:
        show_feature_importance_tab(has_model)


def show_demand_statistics_tab(has_demand):
    """Display demand statistics and distribution analysis"""
    
    if has_demand:
        st.markdown("#### 📊 Demand Data Statistics & Distribution")
        
        df = st.session_state.demand_data.copy()
        
        # Statistical Summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Basic Statistics")
            stats_data = {
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range', '25th Percentile', '75th Percentile'],
                'Value': [
                    f"{df['demand'].mean():.2f}",
                    f"{df['demand'].median():.2f}",
                    f"{df['demand'].std():.2f}",
                    f"{df['demand'].min():.2f}",
                    f"{df['demand'].max():.2f}",
                    f"{df['demand'].max() - df['demand'].min():.2f}",
                    f"{df['demand'].quantile(0.25):.2f}",
                    f"{df['demand'].quantile(0.75):.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("##### Distribution Analysis")
            
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df['demand'],
                nbinsx=30,
                marker=dict(color='#6366f1', line=dict(color='white', width=1))
            ))
            fig_hist.update_layout(
                title='Demand Distribution',
                xaxis_title='Demand',
                yaxis_title='Frequency',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#111827',
                font=dict(color='#ffffff'),
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Time-based Analysis
        st.markdown("---")
        st.markdown("##### Time-Based Patterns")
        
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month_name()
        df['year'] = df['date'].dt.year
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Day of week pattern
            dow_avg = df.groupby('day_of_week')['demand'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig_dow = go.Figure(data=[
                go.Bar(x=dow_avg.index, y=dow_avg.values, marker_color='#8b5cf6')
            ])
            fig_dow.update_layout(
                title='Average Demand by Day of Week',
                xaxis_title='Day',
                yaxis_title='Avg Demand',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#111827',
                font=dict(color='#ffffff'),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_dow, use_container_width=True)
        
        with col2:
            # Monthly pattern
            month_avg = df.groupby('month')['demand'].mean()
            
            fig_month = go.Figure(data=[
                go.Bar(x=month_avg.index, y=month_avg.values, marker_color='#a78bfa')
            ])
            fig_month.update_layout(
                title='Average Demand by Month',
                xaxis_title='Month',
                yaxis_title='Avg Demand',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#111827',
                font=dict(color='#ffffff'),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_month, use_container_width=True)
        
        with col3:
            # Box plot for outlier detection
            fig_box = go.Figure(data=[
                go.Box(y=df['demand'], marker_color='#c4b5fd', name='Demand')
            ])
            fig_box.update_layout(
                title='Demand Distribution (Box Plot)',
                yaxis_title='Demand',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#111827',
                font=dict(color='#ffffff'),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Trend Analysis
        st.markdown("---")
        st.markdown("##### Trend & Seasonality Analysis")
        
        # Moving averages
        df_trend = df.copy().sort_values('date')
        df_trend['MA_7'] = df_trend['demand'].rolling(window=7).mean()
        df_trend['MA_30'] = df_trend['demand'].rolling(window=30).mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_trend['date'], y=df_trend['demand'],
            mode='lines', name='Actual', line=dict(color='#6366f1', width=1),
            opacity=0.5
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_trend['date'], y=df_trend['MA_7'],
            mode='lines', name='7-Day MA', line=dict(color='#8b5cf6', width=2)
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_trend['date'], y=df_trend['MA_30'],
            mode='lines', name='30-Day MA', line=dict(color='#a78bfa', width=2)
        ))
        
        fig_trend.update_layout(
            title='Demand with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Demand',
            plot_bgcolor='#1f2937',
            paper_bgcolor='#111827',
            font=dict(color='#ffffff'),
            height=400,
            legend=dict(bgcolor='rgba(31, 41, 55, 0.8)')
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    else:
        st.info("📊 Upload demand data to see statistics")


def show_correlation_analysis_tab(has_reviews, has_demand):
    """Display correlation analysis between features"""
    
    st.markdown("#### 🔍 Feature Correlation & Relationships")
    
    if has_reviews and has_demand and st.session_state.processed_data is not None:
        
        # Prepare combined data
        demand_df = st.session_state.demand_data.copy()
        review_df = st.session_state.processed_data.copy()
        
        # Aggregate reviews by date
        if 'sentiment_compound' in review_df.columns:
            review_agg = review_df.groupby('date').agg({
                'sentiment_compound': 'mean',
                'review_text': 'count'
            }).reset_index()
            review_agg.columns = ['date', 'avg_sentiment', 'review_count']
            
            # Merge with demand
            merged = pd.merge(demand_df, review_agg, on='date', how='left')
            merged = merged.dropna()
            
            if len(merged) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter: Sentiment vs Demand
                    fig_scatter1 = px.scatter(
                        merged, x='avg_sentiment', y='demand',
                        trendline='ols',
                        title='Sentiment vs Demand',
                        color='demand',
                        color_continuous_scale='Purples'
                    )
                    fig_scatter1.update_layout(
                        plot_bgcolor='#1f2937',
                        paper_bgcolor='#111827',
                        font=dict(color='#ffffff'),
                        height=400
                    )
                    st.plotly_chart(fig_scatter1, use_container_width=True)
                    
                    # Calculate correlation
                    corr = merged['avg_sentiment'].corr(merged['demand'])
                    st.metric("Correlation Coefficient", f"{corr:.3f}")
                
                with col2:
                    # Scatter: Review Count vs Demand
                    fig_scatter2 = px.scatter(
                        merged, x='review_count', y='demand',
                        trendline='ols',
                        title='Review Count vs Demand',
                        color='demand',
                        color_continuous_scale='Purples'
                    )
                    fig_scatter2.update_layout(
                        plot_bgcolor='#1f2937',
                        paper_bgcolor='#111827',
                        font=dict(color='#ffffff'),
                        height=400
                    )
                    st.plotly_chart(fig_scatter2, use_container_width=True)
                    
                    corr2 = merged['review_count'].corr(merged['demand'])
                    st.metric("Correlation Coefficient", f"{corr2:.3f}")
                
                # Correlation heatmap
                st.markdown("##### Correlation Matrix")
                corr_matrix = merged[['demand', 'avg_sentiment', 'review_count']].corr()
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='Purples',
                    text=corr_matrix.values.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 16},
                    showscale=True
                ))
                fig_heatmap.update_layout(
                    title='Feature Correlation Heatmap',
                    plot_bgcolor='#1f2937',
                    paper_bgcolor='#111827',
                    font=dict(color='#ffffff'),
                    height=400
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("Not enough matching dates between demand and review data")
        else:
            st.info("Run sentiment analysis first to see correlations")
    else:
        st.info("🔍 Upload both demand and review data, then run NLP analysis to see correlations")


def show_model_comparison_tab(has_model):
    """Display model performance comparison"""
    
    st.markdown("#### 🎯 Model Performance Comparison")
    
    if has_model and st.session_state.metrics:
        # Show current model metrics
        st.markdown("##### Current Model Performance")
        
        metrics = st.session_state.metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}", 
                     help="Mean Absolute Error - Lower is better")
        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}",
                     help="Root Mean Squared Error - Lower is better")
        with col3:
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%",
                     help="Mean Absolute Percentage Error - Lower is better")
        with col4:
            st.metric("R² Score", f"{metrics['R2']:.4f}",
                     help="Coefficient of Determination - Higher is better (max 1.0)")
        
        st.markdown("---")
        
        # Model comparison guide
        st.markdown("##### 📊 Compare All Models")
        st.info("💡 Tip: Use 'All Models (Compare)' option in Forecasting tab for comprehensive comparison")
        
        # Visual comparison guide
        comparison_data = {
            'Model': ['XGBoost', 'Gradient Boosting', 'Random Forest', 'Linear Regression', 'SVR'],
            'Speed': ['⚡⚡⚡', '⚡⚡', '⚡⚡', '⚡⚡⚡⚡', '⚡'],
            'Accuracy': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐'],
            'Complexity': ['High', 'High', 'Medium', 'Low', 'Medium'],
            'Best For': [
                'Large datasets, complex patterns',
                'Gradient-based optimization',
                'Interpretable, robust predictions',
                'Linear trends, simple relationships',
                'Non-linear patterns, small datasets'
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        # Performance interpretation
        st.markdown("---")
        st.markdown("##### 📈 Performance Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Good Performance Indicators:**")
            st.markdown("""
            - ✅ R² Score > 0.7 (Excellent fit)
            - ✅ MAPE < 10% (High accuracy)
            - ✅ Low RMSE relative to demand range
            - ✅ MAE close to expected variation
            """)
        
        with col2:
            st.markdown("**Warning Signs:**")
            st.markdown("""
            - ⚠️ R² Score < 0.5 (Poor fit)
            - ⚠️ MAPE > 20% (Low accuracy)
            - ⚠️ High RMSE (Large errors)
            - ⚠️ MAE > 20% of average demand
            """)
        
        # Current model assessment
        r2 = metrics['R2']
        mape = metrics['MAPE']
        
        st.markdown("##### 🎯 Your Model Assessment")
        
        if r2 > 0.7 and mape < 10:
            st.success("🎉 Excellent! Your model shows strong performance.")
        elif r2 > 0.5 and mape < 20:
            st.info("👍 Good! Your model is performing reasonably well.")
        else:
            st.warning("⚠️ Consider: More data, feature engineering, or trying different models.")
    
    else:
        st.info("🎯 Train a model first to see performance comparison")


def show_feature_importance_tab(has_model):
    """Display feature importance analysis"""
    
    st.markdown("#### 💎 Feature Importance & Impact Analysis")
    
    if has_model and st.session_state.best_model:
        
        # Try to get feature importance
        model = st.session_state.best_model.model
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                feature_cols = st.session_state.best_model.feature_cols
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_imp = go.Figure(data=[
                        go.Bar(
                            x=importance_df['Importance'][:15],
                            y=importance_df['Feature'][:15],
                            orientation='h',
                            marker=dict(
                                color=importance_df['Importance'][:15],
                                colorscale='Purples',
                                showscale=True
                            )
                        )
                    ])
                    fig_imp.update_layout(
                        title='Top 15 Most Important Features',
                        xaxis_title='Importance Score',
                        yaxis_title='Feature',
                        plot_bgcolor='#1f2937',
                        paper_bgcolor='#111827',
                        font=dict(color='#ffffff'),
                        height=500,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                with col2:
                    st.markdown("##### Top 10 Features")
                    st.dataframe(
                        importance_df.head(10).style.format({'Importance': '{:.4f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Feature categories
                st.markdown("---")
                st.markdown("##### 📊 Feature Categories Impact")
                
                # Categorize features
                lag_features = [f for f in feature_cols if 'lag' in f.lower()]
                rolling_features = [f for f in feature_cols if 'rolling' in f.lower()]
                time_features = [f for f in feature_cols if any(t in f.lower() for t in ['day', 'month', 'week'])]
                nlp_features = [f for f in feature_cols if any(n in f.lower() for n in ['sentiment', 'review'])]
                
                categories = []
                if lag_features:
                    lag_imp = importance_df[importance_df['Feature'].isin(lag_features)]['Importance'].sum()
                    categories.append({'Category': 'Lag Features', 'Importance': lag_imp, 'Count': len(lag_features)})
                
                if rolling_features:
                    roll_imp = importance_df[importance_df['Feature'].isin(rolling_features)]['Importance'].sum()
                    categories.append({'Category': 'Rolling Features', 'Importance': roll_imp, 'Count': len(rolling_features)})
                
                if time_features:
                    time_imp = importance_df[importance_df['Feature'].isin(time_features)]['Importance'].sum()
                    categories.append({'Category': 'Time Features', 'Importance': time_imp, 'Count': len(time_features)})
                
                if nlp_features:
                    nlp_imp = importance_df[importance_df['Feature'].isin(nlp_features)]['Importance'].sum()
                    categories.append({'Category': 'NLP Features', 'Importance': nlp_imp, 'Count': len(nlp_features)})
                
                if categories:
                    cat_df = pd.DataFrame(categories)
                    
                    fig_cat = go.Figure(data=[
                        go.Pie(
                            labels=cat_df['Category'],
                            values=cat_df['Importance'],
                            hole=0.4,
                            marker=dict(colors=['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd'])
                        )
                    ])
                    fig_cat.update_layout(
                        title='Feature Category Contribution',
                        plot_bgcolor='#1f2937',
                        paper_bgcolor='#111827',
                        font=dict(color='#ffffff'),
                        height=400
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
                    
                    st.dataframe(cat_df, use_container_width=True, hide_index=True)
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = model.coef_
                feature_cols = st.session_state.best_model.feature_cols
                
                coef_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Coefficient': coefficients,
                    'Abs_Coefficient': np.abs(coefficients)
                }).sort_values('Abs_Coefficient', ascending=False)
                
                fig_coef = go.Figure(data=[
                    go.Bar(
                        x=coef_df['Coefficient'][:15],
                        y=coef_df['Feature'][:15],
                        orientation='h',
                        marker=dict(
                            color=coef_df['Coefficient'][:15],
                            colorscale='RdBu',
                            showscale=True
                        )
                    )
                ])
                fig_coef.update_layout(
                    title='Top 15 Feature Coefficients',
                    xaxis_title='Coefficient Value',
                    yaxis_title='Feature',
                    plot_bgcolor='#1f2937',
                    paper_bgcolor='#111827',
                    font=dict(color='#ffffff'),
                    height=500,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_coef, use_container_width=True)
                
                st.markdown("##### Interpretation")
                st.info("""
                - **Positive coefficients**: Feature increases → Demand increases
                - **Negative coefficients**: Feature increases → Demand decreases
                - **Larger absolute values**: Stronger impact on demand
                """)
                
                st.dataframe(coef_df.head(10), use_container_width=True, hide_index=True)
            
            else:
                st.info("Feature importance not available for this model type (SVR)")
                st.markdown("""
                **Note**: Support Vector Regression (SVR) doesn't provide explicit feature importance scores.
                Consider using tree-based models (XGBoost, Random Forest) for feature importance analysis.
                """)
        
        except Exception as e:
            st.warning(f"Could not extract feature importance: {str(e)}")
            st.info("Feature importance is available for tree-based models (XGBoost, Gradient Boosting, Random Forest)")
    
    else:
        st.info("💎 Train a model first to see feature importance")
        st.markdown("""
        **What you'll see here:**
        - Which features have the most impact on predictions
        - Relative importance of different feature categories
        - How lag features compare to time features
        - Impact of NLP sentiment features (if available)
        """)
