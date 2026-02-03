"""
Custom BI Page Module
Build custom visualizations with intelligent column detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
from typing import List, Dict


# ============================================================================
# SMART COLUMN DETECTOR
# ============================================================================

class SmartColumnDetector:
    """Intelligent column detection and categorization for Custom BI"""
    
    COLUMN_PATTERNS = {
        'ORDER_TRANSACTION': {
            'patterns': ['orderid', 'ordernumber', 'invoicenumber', 'transactionid', 'orderdate', 
                        'shipdate', 'deliverydate', 'paymentdate', 'status', 'orderchannel', 
                        'paymentmethod', 'currency', 'exchangerate', 'discount', 'tax', 
                        'shippingcost', 'totalamount', 'refundamount', 'returnflag', 'cancellationreason'],
            'icon': '🧾',
            'display_name': 'Order/Transaction'
        },
        'PRODUCT': {
            'patterns': ['productid', 'productcode', 'productname', 'productline', 'category', 
                        'subcategory', 'brand', 'msrp', 'priceeach', 'costprice', 'margin', 
                        'weight', 'dimensions', 'color', 'size', 'warranty', 'batchnumber', 
                        'expirydate', 'stockquantity'],
            'icon': '📦',
            'display_name': 'Product'
        },
        'QUANTITY_LINE': {
            'patterns': ['orderlinenumber', 'quantityordered', 'backorderedquantity', 
                        'fulfilledquantity', 'unitofmeasure', 'sales', 'grosssales', 
                        'netsales', 'profit', 'loss'],
            'icon': '🔢',
            'display_name': 'Quantity/Line Details'
        },
        'CUSTOMER': {
            'patterns': ['customerid', 'customername', 'customertype', 'customersegment', 
                        'loyaltyid', 'loyaltystatus', 'contactfirstname', 'contactlastname', 
                        'email', 'phone', 'alternatephone', 'fax', 'dateofbirth', 'gender', 
                        'age', 'incomelevel', 'occupation'],
            'icon': '👤',
            'display_name': 'Customer'
        },
        'ADDRESS_LOCATION': {
            'patterns': ['addressline1', 'addressline2', 'city', 'state', 'postalcode', 
                        'country', 'territory', 'region', 'latitude', 'longitude', 
                        'timezone', 'shippingzone', 'warehouselocation'],
            'icon': '📍',
            'display_name': 'Address/Location'
        },
        'TIME_CALENDAR': {
            'patterns': ['year_id', 'month_id', 'qtr_id', 'week_id', 'day', 'dayofweek', 
                        'isweekend', 'fiscalyear', 'fiscalquarter', 'season', 'holidayflag',
                        'date', 'time', 'datetime', 'timestamp'],
            'icon': '📅',
            'display_name': 'Time/Calendar'
        },
        'SHIPPING_LOGISTICS': {
            'patterns': ['shippingmethod', 'carrier', 'trackingnumber', 'deliverystatus', 
                        'estimateddeliverydate', 'actualdeliverydate'],
            'icon': '🚚',
            'display_name': 'Shipping/Logistics'
        },
        'SALES_MARKETING': {
            'patterns': ['salesperson', 'salesregion', 'campaignid', 'promotionid', 
                        'promotiontype', 'leadsource'],
            'icon': '📢',
            'display_name': 'Sales/Marketing'
        },
        'SYSTEM_METADATA': {
            'patterns': ['created_at', 'updated_at', 'source_system', 'recordstatus', 
                        'isactive', 'dataqualityflag'],
            'icon': '⚙️',
            'display_name': 'System/Metadata'
        }
    }
    
    @classmethod
    def categorize_column(cls, column_name: str) -> str:
        """Categorize a column based on its name"""
        col_lower = column_name.lower().replace('_', '').replace(' ', '')
        
        for category, info in cls.COLUMN_PATTERNS.items():
            for pattern in info['patterns']:
                pattern_clean = pattern.lower().replace('_', '')
                if pattern_clean in col_lower or col_lower in pattern_clean:
                    return category
        
        return 'UNCATEGORIZED'
    
    @classmethod
    def categorize_dataframe(cls, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize all columns in a dataframe"""
        categorized = {category: [] for category in cls.COLUMN_PATTERNS.keys()}
        categorized['UNCATEGORIZED'] = []
        
        for column in df.columns:
            category = cls.categorize_column(column)
            categorized[category].append(column)
        
        categorized = {k: v for k, v in categorized.items() if v}
        
        return categorized
    
    @classmethod
    def get_numeric_columns(cls, df: pd.DataFrame) -> List[str]:
        """Get all numeric columns from dataframe"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    @classmethod
    def get_categorical_columns(cls, df: pd.DataFrame, max_unique: int = 50) -> List[str]:
        """Get categorical columns"""
        categorical = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < max_unique:
                categorical.append(col)
        return categorical


# ============================================================================
# CUSTOM GRAPH BUILDER
# ============================================================================

def create_custom_graph(df: pd.DataFrame, graph_type: str, x_col: str = None, 
                       y_col: str = None, color_col: str = None, 
                       aggregation: str = 'sum', title: str = None) -> go.Figure:
    """Create custom graph based on user specifications"""
    
    if not title:
        title = f"{graph_type.title()} Chart"
        if x_col and y_col:
            title = f"{y_col} by {x_col}"
    
    fig = None
    
    try:
        if graph_type == 'line':
            if x_col and y_col:
                df_plot = df.groupby(x_col)[y_col].agg(aggregation).reset_index()
                fig = px.line(df_plot, x=x_col, y=y_col, title=title,
                             color=color_col if color_col else None)
        
        elif graph_type == 'bar':
            if x_col and y_col:
                df_plot = df.groupby(x_col)[y_col].agg(aggregation).reset_index()
                fig = px.bar(df_plot, x=x_col, y=y_col, title=title,
                            color=color_col if color_col else None)
        
        elif graph_type == 'scatter':
            if x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col, title=title,
                               color=color_col if color_col else None)
        
        elif graph_type == 'area':
            if x_col and y_col:
                df_plot = df.groupby(x_col)[y_col].agg(aggregation).reset_index()
                fig = px.area(df_plot, x=x_col, y=y_col, title=title,
                             color=color_col if color_col else None)
        
        elif graph_type == 'box':
            if x_col and y_col:
                fig = px.box(df, x=x_col, y=y_col, title=title,
                            color=color_col if color_col else None)
        
        elif graph_type == 'violin':
            if x_col and y_col:
                fig = px.violin(df, x=x_col, y=y_col, title=title,
                               color=color_col if color_col else None)
        
        elif graph_type == 'pie':
            if x_col and y_col:
                df_plot = df.groupby(x_col)[y_col].agg(aggregation).reset_index()
                fig = px.pie(df_plot, names=x_col, values=y_col, title=title)
        
        elif graph_type == 'heatmap':
            if x_col and y_col:
                pivot = df.pivot_table(values=y_col, index=x_col, 
                                      columns=color_col if color_col else None,
                                      aggfunc=aggregation)
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='Purples'
                ))
                fig.update_layout(title=title, xaxis_title=color_col, yaxis_title=x_col)
        
        elif graph_type == 'histogram':
            if x_col:
                fig = px.histogram(df, x=x_col, title=title,
                                  color=color_col if color_col else None)
        
        elif graph_type == 'sunburst':
            if x_col and y_col:
                fig = px.sunburst(df, path=[x_col], values=y_col, title=title)
        
        elif graph_type == 'treemap':
            if x_col and y_col:
                fig = px.treemap(df, path=[x_col], values=y_col, title=title)
        
        # Apply consistent styling
        if fig:
            fig.update_layout(
                plot_bgcolor='#1f2937',
                paper_bgcolor='#111827',
                font=dict(color='#ffffff', family='Arial'),
                xaxis=dict(gridcolor='#374151', showgrid=True),
                yaxis=dict(gridcolor='#374151', showgrid=True),
                hovermode='closest',
                height=500
            )
    
    except Exception as e:
        st.error(f"Error creating graph: {str(e)}")
        return None
    
    return fig


# ============================================================================
# CUSTOM BI BUILDER UI
# ============================================================================

def show_custom_bi_builder(df: pd.DataFrame, data_name: str = "Data"):
    """Display the Custom BI graph builder interface"""
    
    st.markdown(f"### 📊 Custom BI Builder - {data_name}")
    st.info("💡 Build your own custom visualizations using available columns from your data")
    
    # Column detection
    detector = SmartColumnDetector()
    categorized_cols = detector.categorize_dataframe(df)
    numeric_cols = detector.get_numeric_columns(df)
    categorical_cols = detector.get_categorical_columns(df)
    
    # Show available column categories
    with st.expander("📋 Available Columns by Category", expanded=False):
        for category, columns in categorized_cols.items():
            if columns:
                category_info = detector.COLUMN_PATTERNS.get(category, {})
                icon = category_info.get('icon', '📊')
                display_name = category_info.get('display_name', category)
                
                st.markdown(f"**{icon} {display_name}** ({len(columns)} columns)")
                st.caption(", ".join(columns))
    
    # Graph builder
    st.markdown("---")
    st.markdown("#### 🎨 Build Your Graph")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Graph type selection
        graph_types = [
            'line', 'bar', 'scatter', 'area', 'pie', 
            'box', 'violin', 'histogram', 'heatmap',
            'sunburst', 'treemap'
        ]
        
        graph_type = st.selectbox(
            "Graph Type",
            options=graph_types,
            format_func=lambda x: {
                'line': '📈 Line Chart',
                'bar': '📊 Bar Chart',
                'scatter': '🔵 Scatter Plot',
                'area': '🎨 Area Chart',
                'pie': '🥧 Pie Chart',
                'box': '📦 Box Plot',
                'violin': '🎻 Violin Plot',
                'histogram': '📉 Histogram',
                'heatmap': '🔥 Heatmap',
                'sunburst': '☀️ Sunburst',
                'treemap': '🌳 Treemap'
            }[x]
        )
    
    with col2:
        # Aggregation method
        aggregation = st.selectbox(
            "Aggregation Method",
            options=['sum', 'mean', 'count', 'min', 'max', 'median', 'std'],
            index=0
        )
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        all_columns = ['-- None --'] + list(df.columns)
        x_col = st.selectbox("X-Axis (Category/Date)", options=all_columns)
        x_col = None if x_col == '-- None --' else x_col
    
    with col4:
        y_options = ['-- None --'] + numeric_cols if numeric_cols else all_columns
        y_col = st.selectbox("Y-Axis (Numeric)", options=y_options)
        y_col = None if y_col == '-- None --' else y_col
    
    with col5:
        color_col = st.selectbox("Color By (Optional)", options=all_columns)
        color_col = None if color_col == '-- None --' else color_col
    
    # Custom title
    custom_title = st.text_input("Custom Graph Title (Optional)", 
                                 placeholder="Leave empty for auto-generated title")
    
    # Generate button
    if st.button("🎨 Generate Graph", type="primary", use_container_width=True):
        if x_col or y_col:
            with st.spinner("Creating your custom visualization..."):
                fig = create_custom_graph(
                    df=df,
                    graph_type=graph_type,
                    x_col=x_col,
                    y_col=y_col,
                    color_col=color_col,
                    aggregation=aggregation,
                    title=custom_title if custom_title else None
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download options
                    st.markdown("#### 💾 Export Options")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        html_buffer = io.StringIO()
                        fig.write_html(html_buffer)
                        st.download_button(
                            "📥 Download as HTML",
                            data=html_buffer.getvalue(),
                            file_name=f"custom_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                    
                    with col_b:
                        if x_col and y_col:
                            export_cols = [c for c in [x_col, y_col, color_col] if c]
                            export_df = df[export_cols].copy()
                            csv_export = export_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Data (CSV)",
                                data=csv_export,
                                file_name=f"graph_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("Could not generate graph. Please check your column selections.")
        else:
            st.warning("⚠️ Please select at least one column for X or Y axis")
    
    # Quick insights
    if x_col or y_col:
        st.markdown("---")
        st.markdown("#### 🔍 Quick Data Insights")
        
        insight_cols = st.columns(4)
        
        if x_col:
            with insight_cols[0]:
                unique_x = df[x_col].nunique()
                st.metric(f"{x_col} (unique)", unique_x)
        
        if y_col and y_col in numeric_cols:
            with insight_cols[1]:
                avg_y = df[y_col].mean()
                st.metric(f"{y_col} (avg)", f"{avg_y:.2f}")
            
            with insight_cols[2]:
                sum_y = df[y_col].sum()
                st.metric(f"{y_col} (sum)", f"{sum_y:.2f}")
            
            with insight_cols[3]:
                max_y = df[y_col].max()
                st.metric(f"{y_col} (max)", f"{max_y:.2f}")


# ============================================================================
# CUSTOM BI PAGE
# ============================================================================

def show_custom_bi_page():
    """Custom BI page for building custom visualizations"""
    
    st.markdown("### 🎨 Custom Business Intelligence Builder")
    st.markdown("Build custom visualizations from your uploaded data using any available columns")
    
    # Check available data sources
    available_data = []
    
    if st.session_state.original_csv_data is not None:
        available_data.append(("Original CSV Data (All Columns)", st.session_state.original_csv_data))
    
    if st.session_state.demand_data is not None:
        available_data.append(("Demand Data (Processed)", st.session_state.demand_data))
    
    if st.session_state.scraped_data is not None:
        available_data.append(("Review Data", st.session_state.scraped_data))
    
    if st.session_state.processed_data is not None:
        available_data.append(("Processed NLP Data", st.session_state.processed_data))
    
    if not available_data:
        st.warning("⚠️ No data available. Please upload data first in the 'Data Input' tab.")
        
        # Show example capabilities
        st.markdown("---")
        st.markdown("#### 📚 What You Can Do with Custom BI:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Order Analysis**
            - Sales by region
            - Revenue trends
            - Order status distribution
            - Payment method analysis
            """)
        
        with col2:
            st.markdown("""
            **📦 Product Insights**
            - Top products by sales
            - Category performance
            - Price vs. demand
            - Inventory analysis
            """)
        
        with col3:
            st.markdown("""
            **👤 Customer Analytics**
            - Customer segmentation
            - Geographic distribution
            - Loyalty trends
            - Purchase patterns
            """)
        
        return
    
    # Data source selector
    st.markdown("#### 📂 Select Data Source")
    
    data_source_names = [name for name, _ in available_data]
    
    default_index = 0
    for idx, name in enumerate(data_source_names):
        if "Original CSV" in name:
            default_index = idx
            break
    
    selected_source = st.selectbox(
        "Choose dataset to visualize",
        options=data_source_names,
        index=default_index,
        help="💡 Select 'Original CSV Data' to access all columns from your uploaded file"
    )
    
    # Get selected dataframe
    selected_df = None
    for name, df in available_data:
        if name == selected_source:
            selected_df = df.copy()
            break
    
    if selected_df is not None:
        # Show data preview
        with st.expander(f"📊 {selected_source} Preview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", len(selected_df))
            with col2:
                st.metric("Total Columns", len(selected_df.columns))
            with col3:
                numeric_count = len(selected_df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_count)
            with col4:
                memory_usage = selected_df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory", f"{memory_usage:.2f} MB")
            
            st.markdown("**Column Names:**")
            st.caption(", ".join(selected_df.columns.tolist()))
            
            st.dataframe(selected_df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Show the custom BI builder
        show_custom_bi_builder(selected_df, selected_source)
        
        # Multi-graph builder
        st.markdown("---")
        st.markdown("### 📊 Multi-Graph Dashboard")
        st.info("💡 Create multiple graphs side-by-side for comprehensive analysis")
        
        num_graphs = st.slider("Number of graphs to create", 1, 4, 2)
        
        if st.button("🎯 Build Multi-Graph Dashboard", use_container_width=True):
            st.session_state['multi_graph_mode'] = True
            st.session_state['num_multi_graphs'] = num_graphs
        
        if st.session_state.get('multi_graph_mode', False):
            num_graphs = st.session_state.get('num_multi_graphs', 2)
            
            # Create columns based on number of graphs
            if num_graphs == 1:
                cols = [st.container()]
            elif num_graphs == 2:
                cols = st.columns(2)
            elif num_graphs == 3:
                cols = st.columns(3)
            else:
                cols = st.columns(2)
            
            for idx in range(num_graphs):
                with cols[idx % len(cols)]:
                    st.markdown(f"#### Graph {idx + 1}")
                    
                    graph_type = st.selectbox(
                        f"Type",
                        options=['line', 'bar', 'scatter', 'pie', 'box'],
                        key=f"multi_graph_type_{idx}"
                    )
                    
                    x_col = st.selectbox(
                        f"X-Axis",
                        options=['-- None --'] + list(selected_df.columns),
                        key=f"multi_x_{idx}"
                    )
                    
                    y_col = st.selectbox(
                        f"Y-Axis",
                        options=['-- None --'] + list(selected_df.columns),
                        key=f"multi_y_{idx}"
                    )
                    
                    if x_col != '-- None --' and y_col != '-- None --':
                        try:
                            fig = create_custom_graph(
                                df=selected_df,
                                graph_type=graph_type,
                                x_col=x_col,
                                y_col=y_col,
                                aggregation='sum',
                                title=f"{y_col} by {x_col}"
                            )
                            
                            if fig:
                                fig.update_layout(height=350)
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
