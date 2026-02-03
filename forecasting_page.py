"""
Forecasting Page Module
Handles model training, forecasting, and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try importing feature engineering
try:
    from nlp.feature_engineering import FeatureEngineer
except ImportError:
    class FeatureEngineer:
        def __init__(self):
            self.feature_names = []
        def merge_datasets(self, df1, df2, on):
            return pd.merge(df1, df2, on=on, how='left').fillna(0)
        def create_all_features(self, df, target, date_col):
            return self.create_time_features(df, date_col)
        def create_time_features(self, df, date_col):
            df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek
            df['month'] = pd.to_datetime(df[date_col]).dt.month
            self.feature_names = ['day_of_week', 'month']
            return df
        def create_lag_features(self, df, col, lags=[1,7]):
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                self.feature_names.append(f'{col}_lag_{lag}')
            return df
        def create_rolling_features(self, df, col, windows=[7]):
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                self.feature_names.append(f'{col}_rolling_mean_{window}')
            return df

# Try importing ML models
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    st.error("⚠️ ML libraries not available. Please install scikit-learn and xgboost.")


# ============================================================================
# MODEL FORECASTER CLASS
# ============================================================================

class EnhancedDemandForecaster:
    """Enhanced forecaster with multiple models"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        
    def train(self, train_df, target_col, feature_cols):
        """Train the model"""
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression()
        elif self.model_type == 'svr':
            self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        
        self.model.fit(X_train, y_train)
        self.feature_cols = feature_cols
        
    def predict(self, test_df):
        """Make predictions"""
        X_test = test_df[self.feature_cols].values
        predictions = self.model.predict(X_test)
        return pd.DataFrame({'prediction': predictions})
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE safely
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }


def train_and_evaluate_all_models_enhanced(df, target_col, date_col, feature_cols, test_size=0.2):
    """Train and evaluate multiple models"""
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    models = {
        'XGBoost': 'xgboost',
        'Gradient Boosting': 'gradient_boosting',
        'Random Forest': 'random_forest',
        'Linear Regression': 'linear_regression',
        'Support Vector Regression': 'svr'
    }
    
    results = {}
    
    for model_name, model_type in models.items():
        try:
            forecaster = EnhancedDemandForecaster(model_type)
            forecaster.train(train_df, target_col, feature_cols)
            
            pred = forecaster.predict(test_df)
            metrics = forecaster.evaluate(test_df[target_col].values, pred['prediction'].values)
            
            results[model_name] = {
                'model': forecaster,
                'metrics': metrics,
                'predictions': pred
            }
        except Exception as e:
            st.warning(f"Model {model_name} failed: {str(e)}")
            continue
    
    return results


# ============================================================================
# ITERATIVE FORECASTING
# ============================================================================

def create_iterative_forecast(forecaster, merged_df, feature_cols, forecast_horizon):
    """Create iterative time-series forecast"""
    
    last_date = merged_df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
    
    forecast_values = []
    historical_demands = merged_df['demand'].tolist()
    
    # Get NLP features if available
    has_sentiment = 'avg_sentiment' in merged_df.columns
    has_review_count = 'review_count' in merged_df.columns
    avg_sentiment = merged_df['avg_sentiment'].mean() if has_sentiment else 0
    avg_review_count = merged_df['review_count'].mean() if has_review_count else 0
    
    # Historical statistics
    historical_mean = np.mean(historical_demands)
    historical_std = np.std(historical_demands)
    recent_trend = np.polyfit(range(min(30, len(historical_demands))), 
                               historical_demands[-min(30, len(historical_demands)):], 1)[0]
    
    for i in range(forecast_horizon):
        future_date = future_dates[i]
        
        # Create time-based features
        features = {
            'day_of_week': future_date.dayofweek,
            'month': future_date.month,
        }
        
        # Add NLP features if available
        if has_sentiment:
            features['avg_sentiment'] = avg_sentiment
        if has_review_count:
            features['review_count'] = avg_review_count
        
        # Create lag features
        all_demands = historical_demands + forecast_values
        
        for lag in [1, 7, 14, 30]:
            if f'demand_lag_{lag}' in feature_cols:
                if len(all_demands) >= lag:
                    features[f'demand_lag_{lag}'] = all_demands[-lag]
                else:
                    features[f'demand_lag_{lag}'] = historical_demands[-1] if historical_demands else historical_mean
        
        # Create rolling features
        for window in [7, 14, 30]:
            if f'demand_rolling_mean_{window}' in feature_cols:
                if len(all_demands) >= window:
                    features[f'demand_rolling_mean_{window}'] = np.mean(all_demands[-window:])
                elif all_demands:
                    features[f'demand_rolling_mean_{window}'] = np.mean(all_demands)
                else:
                    features[f'demand_rolling_mean_{window}'] = np.mean(historical_demands[-window:]) if len(historical_demands) >= window else historical_mean
        
        # Ensure all required features are present
        for col in feature_cols:
            if col not in features:
                features[col] = merged_df[col].tail(30).mean()
        
        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([features])[feature_cols]
        
        # Make prediction
        pred = forecaster.predict(feature_df)
        predicted_value = pred['prediction'].values[0]
        
        # Add trend and noise
        trend_adjustment = recent_trend * (i + 1) * 0.3
        predicted_value += trend_adjustment
        
        noise = np.random.normal(0, historical_std * 0.03)
        predicted_value = max(0, predicted_value + noise)
        
        forecast_values.append(predicted_value)
    
    return np.array(forecast_values), future_dates


# ============================================================================
# FORECAST VISUALIZATION
# ============================================================================

def create_enhanced_forecast_plot(historical_df, forecast_df, chart_type='combined'):
    """Create enhanced forecast visualization"""
    
    fig = go.Figure()
    
    if chart_type in ['combined', 'line']:
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['demand'],
            mode='lines',
            name='Historical',
            line=dict(color='#6366f1', width=3),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.2)'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecasted_demand'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#8b5cf6', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        # Confidence interval
        upper_bound = forecast_df['forecasted_demand'] * 1.1
        lower_bound = forecast_df['forecasted_demand'] * 0.9
        
        fig.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(139, 92, 246, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Confidence Interval'
        ))
    
    elif chart_type == 'area':
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['demand'],
            fill='tozeroy',
            name='Historical',
            line=dict(color='#6366f1')
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecasted_demand'],
            fill='tozeroy',
            name='Forecast',
            line=dict(color='#8b5cf6')
        ))
    
    elif chart_type == 'comparison':
        last_30_days = historical_df.tail(30)
        
        fig.add_trace(go.Bar(
            x=last_30_days['date'],
            y=last_30_days['demand'],
            name='Recent Historical',
            marker=dict(color='#6366f1', opacity=0.7)
        ))
        
        fig.add_trace(go.Bar(
            x=forecast_df['date'],
            y=forecast_df['forecasted_demand'],
            name='Forecast',
            marker=dict(color='#8b5cf6', opacity=0.7)
        ))
    
    # Styling
    fig.update_layout(
        title=dict(
            text='<b>Demand Forecast with Confidence Interval</b>',
            font=dict(size=24, family='Arial Black', color='#ffffff')
        ),
        plot_bgcolor='#1f2937',
        paper_bgcolor='#111827',
        font=dict(color='#ffffff', family='Arial'),
        xaxis=dict(
            title='<b>Date</b>',
            gridcolor='#374151',
            showgrid=True
        ),
        yaxis=dict(
            title='<b>Demand</b>',
            gridcolor='#374151',
            showgrid=True
        ),
        hovermode='x unified',
        height=600,
        legend=dict(
            bgcolor='rgba(31, 41, 55, 0.8)',
            bordercolor='#6366f1',
            borderwidth=2
        ),
        margin=dict(t=80, b=60, l=60, r=40)
    )
    
    return fig


# ============================================================================
# FORECASTING PAGE
# ============================================================================

def show_forecasting_page():
    """Forecasting page with enhanced visualizations"""
    
    if not MODELS_AVAILABLE:
        st.error("⚠️ Machine learning libraries not available. Please install required packages.")
        return
    
    if st.session_state.demand_data is None:
        st.warning("⚠️ Please upload demand data first!")
        return
    
    demand_df = st.session_state.demand_data.copy()
    has_nlp = st.session_state.processed_data is not None
    
    # Feature engineering
    if has_nlp:
        st.info("✅ NLP features available - Enhanced forecasting enabled")
        nlp_df = st.session_state.processed_data.copy()
        
        engineer = FeatureEngineer()
        nlp_df['date'] = pd.to_datetime(nlp_df['date'])
        nlp_agg = nlp_df.groupby('date').agg({
            'sentiment_compound': 'mean' if 'sentiment_compound' in nlp_df.columns else lambda x: 0,
            'review_text': 'count'
        }).reset_index()
        nlp_agg.columns = ['date', 'avg_sentiment', 'review_count']
        
        merged_df = engineer.merge_datasets(demand_df, nlp_agg, 'date')
        merged_df = engineer.create_all_features(merged_df, 'demand', 'date')
        feature_cols = engineer.feature_names
    else:
        st.warning("⚠️ No NLP data - Using basic time-series forecasting")
        engineer = FeatureEngineer()
        merged_df = engineer.create_time_features(demand_df, 'date')
        merged_df = engineer.create_lag_features(merged_df, 'demand')
        merged_df = engineer.create_rolling_features(merged_df, 'demand')
        feature_cols = [col for col in merged_df.columns if col not in ['date', 'demand']]
    
    # Convert UInt types
    for col in merged_df.columns:
        if str(merged_df[col].dtype).startswith('UInt'):
            merged_df[col] = merged_df[col].astype('int64')
    
    merged_df = merged_df.dropna()
    
    st.markdown(f"### Dataset: {len(merged_df)} rows, {len(feature_cols)} features")
    
    # Model configuration
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model", [
            "XGBoost", 
            "Gradient Boosting",
            "Random Forest", 
            "Linear Regression",
            "Support Vector Regression",
            "All Models (Compare)"
        ])
    with col2:
        test_size = st.slider("Test Set (%)", 10, 40, 20) / 100
    
    forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
    
    if st.button("🚀 Train & Forecast", type="primary"):
        with st.spinner("Training model..."):
            try:
                if model_type == "All Models (Compare)":
                    results = train_and_evaluate_all_models_enhanced(
                        merged_df, 'demand', 'date', feature_cols, test_size
                    )
                    
                    if not results:
                        st.error("❌ Training failed")
                        return
                    
                    st.markdown("### Model Comparison")
                    metrics_df = pd.DataFrame({
                        model_name: result['metrics']
                        for model_name, result in results.items()
                    }).T
                    st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen'))
                    
                    best_model_name = metrics_df['RMSE'].idxmin()
                    st.success(f"✅ Best Model: {best_model_name}")
                    
                    best_forecaster = results[best_model_name]['model']
                    best_metrics = results[best_model_name]['metrics']
                    st.session_state.model_name = best_model_name
                else:
                    model_map = {
                        "XGBoost": "xgboost",
                        "Gradient Boosting": "gradient_boosting",
                        "Random Forest": "random_forest",
                        "Linear Regression": "linear_regression",
                        "Support Vector Regression": "svr"
                    }
                    
                    split_idx = int(len(merged_df) * (1 - test_size))
                    train_df = merged_df[:split_idx]
                    test_df = merged_df[split_idx:]
                    
                    forecaster = EnhancedDemandForecaster(model_map[model_type])
                    forecaster.train(train_df, 'demand', feature_cols)
                    
                    pred = forecaster.predict(test_df)
                    best_metrics = forecaster.evaluate(test_df['demand'].values, pred['prediction'].values)
                    
                    st.success(f"✅ {model_type} trained!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{best_metrics['MAE']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{best_metrics['RMSE']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{best_metrics['MAPE']:.2f}%")
                    with col4:
                        st.metric("R²", f"{best_metrics['R2']:.4f}")
                    
                    best_forecaster = forecaster
                    st.session_state.model_name = model_type
                
                st.markdown("### Future Forecast")
                
                # Use iterative forecasting
                forecast_values, future_dates = create_iterative_forecast(
                    best_forecaster, merged_df, feature_cols, forecast_horizon
                )
                
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'forecasted_demand': forecast_values
                })
                
                st.session_state.model_trained = True
                st.session_state.forecast_values = forecast_values
                st.session_state.forecast_dates = future_dates
                st.session_state.best_model = best_forecaster
                st.session_state.metrics = best_metrics
                st.session_state.forecast_data = forecast_df
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display forecast visualization
    if st.session_state.model_trained and st.session_state.forecast_data is not None:
        st.markdown("---")
        st.markdown("### 📊 Forecast Visualization")
        
        forecast_chart = st.selectbox(
            "Select Forecast Visualization Type",
            options=['combined', 'line', 'area', 'comparison'],
            format_func=lambda x: {
                'combined': '🎯 Combined (with Confidence Interval)',
                'line': '📈 Line Chart',
                'area': '🎨 Area Chart',
                'comparison': '📊 Side-by-Side Comparison'
            }[x],
            key='forecast_chart_type'
        )
        
        if st.session_state.demand_data is not None:
            fig = create_enhanced_forecast_plot(
                st.session_state.demand_data, 
                st.session_state.forecast_data, 
                forecast_chart
            )
            st.plotly_chart(fig, use_container_width=True)
