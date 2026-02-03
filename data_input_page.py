# """
# Data Input Page Module
# Handles CSV uploads, web scraping, and intelligent column mapping
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# import re
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse, urljoin
# from datetime import datetime, timedelta
# import json
# import hashlib
# import random
# from typing import Dict, List, Optional
# import plotly.express as px
# import plotly.graph_objects as go


# # ============================================================================
# # UNIVERSAL WEB SCRAPER
# # ============================================================================

# def scrape_url(url: str, max_items: int = 200) -> pd.DataFrame:
#     """Universal web scraper for extracting reviews and data from any URL"""
    
#     HEADERS = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
#         "Accept-Language": "en-US,en;q=0.9",
#     }

#     PRICE_REGEX = re.compile(r'[₹$€£¥]\s*[\d,.]+')
#     RATING_REGEX = re.compile(r'(\d+(\.\d+)?)\s*(/5|stars?)', re.I)
#     DATE_REGEX = re.compile(r'\d{4}-\d{2}-\d{2}')

#     def clean(text):
#         return re.sub(r'\s+', ' ', text).strip() if text else None

#     try:
#         r = requests.get(url, headers=HEADERS, timeout=30)
#         r.raise_for_status()
#     except Exception as e:
#         return pd.DataFrame({"error": [str(e)]})

#     soup = BeautifulSoup(r.text, "html.parser")
#     domain = urlparse(url).netloc

#     page_title = clean(soup.title.text) if soup.title else None

#     # Extract JSON-LD
#     records = []

#     for script in soup.find_all("script", type="application/ld+json"):
#         try:
#             data = json.loads(script.string)
#             if isinstance(data, dict):
#                 records.append({
#                     "title": data.get("name"),
#                     "review_text": data.get("description"),
#                     "rating": data.get("aggregateRating", {}).get("ratingValue"),
#                     "price": data.get("offers", {}).get("price"),
#                     "date": data.get("datePublished"),
#                     "url": data.get("url"),
#                     "source_url": url,
#                     "domain": domain,
#                     "page_title": page_title,
#                     "scraped_at": datetime.utcnow()
#                 })
#         except:
#             pass

#     # HTML Content Blocks
#     blocks = soup.find_all(["article", "section", "div", "li"])

#     seen = set()

#     for block in blocks:
#         text = clean(block.get_text(" ", strip=True))
#         if not text or len(text) < 40:
#             continue

#         signature = hashlib.md5(text[:300].encode()).hexdigest()
#         if signature in seen:
#             continue
#         seen.add(signature)

#         price = None
#         rating = None

#         price_match = PRICE_REGEX.search(text)
#         if price_match:
#             price = float(re.sub(r"[^\d.]", "", price_match.group()))

#         rating_match = RATING_REGEX.search(text)
#         if rating_match:
#             rating = float(rating_match.group(1))

#         link = block.find("a", href=True)
#         image = block.find("img")

#         records.append({
#             "title": clean(block.find(["h1","h2","h3","h4","h5","h6"]).text) 
#                      if block.find(["h1","h2","h3","h4","h5","h6"]) else None,
#             "review_text": text[:500],
#             "full_text": text,
#             "rating": rating,
#             "price": price,
#             "url": urljoin(url, link["href"]) if link else None,
#             "image_url": urljoin(url, image["src"]) if image and image.get("src") else None,
#             "source_url": url,
#             "domain": domain,
#             "page_title": page_title,
#             "scraped_at": datetime.utcnow(),
#             "word_count": len(text.split())
#         })

#         if len(records) >= max_items:
#             break

#     if not records:
#         return pd.DataFrame(columns=[
#             "title","review_text","rating","price","url","domain"
#         ])

#     df = pd.DataFrame(records)

#     df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
#     df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
#     df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

#     return df


# # ============================================================================
# # SAMPLE DATA GENERATION
# # ============================================================================

# def create_sample_review_data(num_samples: int = 50) -> pd.DataFrame:
#     """Create sample review data for demonstration"""
    
#     sample_reviews = [
#         "Great product! Exactly what I needed. Fast shipping too.",
#         "Not bad but could be better. The quality is okay for the price.",
#         "Terrible experience. Product broke after one week.",
#         "Love it! Best purchase I've made in a while.",
#         "Decent product but overpriced in my opinion.",
#         "Five stars! Excellent quality and great customer service.",
#         "Would not recommend. Very disappointed with this purchase.",
#         "Pretty good overall. Met my expectations.",
#         "Amazing! Worth every penny. Highly recommend!",
#         "Meh, it's okay. Nothing special but does the job.",
#     ]
    
#     reviews = []
#     base_date = datetime.now()
    
#     for i in range(num_samples):
#         date = base_date - timedelta(days=random.randint(0, 90))
#         rating = random.choice([1, 2, 3, 4, 5])
        
#         if rating >= 4:
#             text = random.choice([r for r in sample_reviews if any(word in r.lower() for word in ['great', 'excellent', 'love', 'amazing'])])
#         elif rating <= 2:
#             text = random.choice([r for r in sample_reviews if any(word in r.lower() for word in ['terrible', 'poor', 'disappointed'])])
#         else:
#             text = random.choice([r for r in sample_reviews if any(word in r.lower() for word in ['okay', 'decent', 'meh'])])
        
#         reviews.append({
#             'review_text': text,
#             'date': date.strftime('%Y-%m-%d'),
#             'rating': rating,
#             'source': 'sample_data'
#         })
    
#     return pd.DataFrame(reviews)


# # ============================================================================
# # COLUMN DETECTION AND MAPPING
# # ============================================================================

# class ColumnMapper:
#     """Intelligent column detection and mapping for various CSV formats"""
    
#     DATE_PATTERNS = [
#         r'date', r'time', r'datetime', r'timestamp', r'created', r'published',
#         r'posted', r'review_date', r'purchase_date', r'order_date', r'day'
#     ]
    
#     TEXT_PATTERNS = [
#         r'review', r'text', r'comment', r'feedback', r'description', r'content',
#         r'message', r'body', r'summary', r'opinion', r'note'
#     ]
    
#     RATING_PATTERNS = [
#         r'rating', r'score', r'stars', r'rate', r'grade', r'rank'
#     ]
    
#     DEMAND_PATTERNS = [
#         r'demand', r'sales', r'quantity', r'units', r'volume', r'orders',
#         r'sold', r'revenue', r'count', r'total'
#     ]
    
#     @staticmethod
#     def fuzzy_match(column_name: str, patterns: List[str]) -> float:
#         """Calculate fuzzy match score between column name and patterns"""
#         column_lower = column_name.lower().strip()
#         best_score = 0
        
#         for pattern in patterns:
#             if re.search(pattern, column_lower):
#                 best_score = max(best_score, 1.0)
#             elif pattern in column_lower or column_lower in pattern:
#                 best_score = max(best_score, 0.7)
        
#         return best_score
    
#     @classmethod
#     def detect_column_type(cls, df: pd.DataFrame, column: str) -> Dict[str, float]:
#         """Detect what type of data a column likely contains"""
#         scores = {
#             'date': 0,
#             'text': 0,
#             'rating': 0,
#             'demand': 0
#         }
        
#         scores['date'] = cls.fuzzy_match(column, cls.DATE_PATTERNS)
#         scores['text'] = cls.fuzzy_match(column, cls.TEXT_PATTERNS)
#         scores['rating'] = cls.fuzzy_match(column, cls.RATING_PATTERNS)
#         scores['demand'] = cls.fuzzy_match(column, cls.DEMAND_PATTERNS)
        
#         sample = df[column].dropna().head(20)
        
#         if len(sample) > 0:
#             try:
#                 pd.to_datetime(sample)
#                 scores['date'] += 0.5
#             except:
#                 pass
            
#             if pd.api.types.is_numeric_dtype(df[column]):
#                 if df[column].max() <= 10 and df[column].min() >= 0:
#                     scores['rating'] += 0.3
#                 scores['demand'] += 0.2
            
#             if pd.api.types.is_string_dtype(df[column]):
#                 avg_length = sample.astype(str).str.len().mean()
#                 if avg_length > 20:
#                     scores['text'] += 0.5
        
#         return scores
    
#     @classmethod
#     def auto_detect_columns(cls, df: pd.DataFrame) -> Dict[str, Optional[str]]:
#         """Automatically detect and map columns to required fields"""
#         mapping = {
#             'date': None,
#             'text': None,
#             'rating': None,
#             'demand': None
#         }
        
#         column_scores = {}
#         for column in df.columns:
#             column_scores[column] = cls.detect_column_type(df, column)
        
#         for col_type in mapping.keys():
#             best_col = None
#             best_score = 0
            
#             for column, scores in column_scores.items():
#                 if scores[col_type] > best_score:
#                     best_score = scores[col_type]
#                     best_col = column
            
#             if best_score > 0.3:
#                 mapping[col_type] = best_col
        
#         return mapping
    
#     @classmethod
#     def get_mapping_confidence(cls, df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Dict[str, float]:
#         """Get confidence scores for each mapped column"""
#         confidence = {}
        
#         for col_type, column in mapping.items():
#             if column is not None:
#                 scores = cls.detect_column_type(df, column)
#                 confidence[col_type] = scores[col_type]
#             else:
#                 confidence[col_type] = 0.0
        
#         return confidence


# def show_column_mapper_ui(df: pd.DataFrame, data_type: str = 'review', context: str = 'main') -> Dict[str, str]:
#     """Display interactive UI for column mapping"""
    
#     mapper = ColumnMapper()
#     auto_mapping = mapper.auto_detect_columns(df)
#     confidence = mapper.get_mapping_confidence(df, auto_mapping)
    
#     st.markdown("#### 🎯 Smart Column Detection")
    
#     cols = st.columns(4)
#     for idx, (field, column) in enumerate(auto_mapping.items()):
#         with cols[idx % 4]:
#             if column:
#                 conf_pct = confidence[field] * 100
#                 color = "🟢" if conf_pct > 70 else "🟡" if conf_pct > 40 else "🔴"
#                 st.metric(field.title(), column, f"{color} {conf_pct:.0f}%")
#             else:
#                 st.metric(field.title(), "Not detected", "🔴 0%")
    
#     st.markdown("---")
#     st.markdown("#### Adjust Mappings (if needed)")
    
#     final_mapping = {}
    
#     if data_type == 'review':
#         required_fields = {
#             'date': 'Date/Time column',
#             'text': 'Review/Comment text',
#             'rating': 'Rating (optional)'
#         }
#     else:
#         required_fields = {
#             'date': 'Date/Time column',
#             'demand': 'Demand/Sales/Quantity'
#         }
    
#     col1, col2 = st.columns(2)
    
#     for idx, (field, description) in enumerate(required_fields.items()):
#         with col1 if idx % 2 == 0 else col2:
#             default_val = auto_mapping.get(field)
#             default_idx = 0
            
#             options = ['-- None --'] + list(df.columns)
            
#             if default_val and default_val in df.columns:
#                 default_idx = options.index(default_val)
            
#             selected = st.selectbox(
#                 f"**{description}**",
#                 options=options,
#                 index=default_idx,
#                 key=f"{context}_map_{field}"
#             )
            
#             if selected != '-- None --':
#                 final_mapping[field] = selected
    
#     return final_mapping


# # ============================================================================
# # ENHANCED VISUALIZATIONS
# # ============================================================================

# def create_enhanced_demand_plot(df, chart_type='line'):
#     """Create enhanced demand visualization with multiple chart types"""
    
#     if chart_type == 'line':
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=df['date'],
#             y=df['demand'],
#             mode='lines',
#             name='Demand',
#             line=dict(color='#6366f1', width=3),
#             fill='tozeroy',
#             fillcolor='rgba(99, 102, 241, 0.1)'
#         ))
        
#     elif chart_type == 'area':
#         fig = px.area(df, x='date', y='demand', 
#                      color_discrete_sequence=['#6366f1'])
        
#     elif chart_type == 'bar':
#         fig = go.Figure()
#         fig.add_trace(go.Bar(
#             x=df['date'],
#             y=df['demand'],
#             marker=dict(
#                 color=df['demand'],
#                 colorscale='Purples',
#                 showscale=True,
#                 colorbar=dict(title="Demand")
#             )
#         ))
        
#     elif chart_type == 'scatter':
#         fig = px.scatter(df, x='date', y='demand',
#                         size='demand',
#                         color='demand',
#                         color_continuous_scale='Purples')
        
#     elif chart_type == 'candlestick':
#         df_roll = df.copy()
#         df_roll['high'] = df_roll['demand'].rolling(7, center=True).max()
#         df_roll['low'] = df_roll['demand'].rolling(7, center=True).min()
#         df_roll['open'] = df_roll['demand'].shift(1)
#         df_roll['close'] = df_roll['demand']
        
#         fig = go.Figure(data=[go.Candlestick(
#             x=df_roll['date'],
#             open=df_roll['open'],
#             high=df_roll['high'],
#             low=df_roll['low'],
#             close=df_roll['close'],
#             increasing_line_color='#6366f1',
#             decreasing_line_color='#ef4444'
#         )])
        
#     elif chart_type == 'heatmap':
#         df_heat = df.copy()
#         df_heat['week'] = df_heat['date'].dt.isocalendar().week
#         df_heat['dow'] = df_heat['date'].dt.dayofweek
#         pivot = df_heat.pivot_table(values='demand', index='dow', columns='week', aggfunc='mean')
        
#         fig = go.Figure(data=go.Heatmap(
#             z=pivot.values,
#             x=pivot.columns,
#             y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
#             colorscale='Purples',
#             showscale=True
#         ))
#         fig.update_layout(
#             xaxis_title='Week of Year',
#             yaxis_title='Day of Week'
#         )
        
#     elif chart_type == 'box':
#         df_box = df.copy()
#         df_box['month'] = df_box['date'].dt.to_period('M').astype(str)
#         fig = px.box(df_box, x='month', y='demand',
#                     color_discrete_sequence=['#6366f1'])
#         fig.update_traces(marker=dict(opacity=0.7))
        
#     elif chart_type == 'violin':
#         df_vio = df.copy()
#         df_vio['month'] = df_vio['date'].dt.to_period('M').astype(str)
#         fig = px.violin(df_vio, x='month', y='demand',
#                        box=True, points='all',
#                        color_discrete_sequence=['#6366f1'])
    
#     # Common styling
#     fig.update_layout(
#         title=dict(
#             text='<b>Historical Demand Trend</b>',
#             font=dict(size=24, family='Arial Black', color='#ffffff')
#         ),
#         plot_bgcolor='#1f2937',
#         paper_bgcolor='#111827',
#         font=dict(color='#ffffff', family='Arial'),
#         xaxis=dict(
#             title='<b>Date</b>',
#             gridcolor='#374151',
#             showgrid=True,
#             zeroline=False
#         ),
#         yaxis=dict(
#             title='<b>Demand</b>',
#             gridcolor='#374151',
#             showgrid=True,
#             zeroline=False
#         ),
#         hovermode='x unified',
#         height=500,
#         margin=dict(t=80, b=60, l=60, r=40)
#     )
    
#     return fig


# # ============================================================================
# # DATA INPUT PAGE
# # ============================================================================

# def show_data_input_page():
#     """Data input page with intelligent column mapping"""
    
#     tab1, tab2, tab3 = st.tabs(["📁 Upload Demand Data", "🌐 Scrape Reviews", "📋 Upload Review Data"])
    
#     with tab1:
#         st.markdown("### Upload Historical Demand Data")
        
#         uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key='demand_upload')
        
#         if uploaded_file:
#             try:
#                 df = pd.read_csv(uploaded_file)
#                 st.success("✅ Data uploaded successfully!")
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Total Records", len(df))
#                 with col2:
#                     st.metric("Columns", len(df.columns))
#                 with col3:
#                     st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
#                 st.markdown("#### Raw Data Preview")
#                 st.dataframe(df.head(10), use_container_width=True)
                
#                 st.markdown("---")
                
#                 column_mapping = show_column_mapper_ui(df, data_type='demand', context='demand')
                
#                 if column_mapping.get('date') and column_mapping.get('demand'):
#                     if st.button("✅ Confirm and Process Data", type="primary"):
#                         try:
#                             # Store the FULL original CSV for Custom BI
#                             st.session_state.original_csv_data = df.copy()
                            
#                             # Process only date and demand for forecasting
#                             df_processed = df[[column_mapping['date'], column_mapping['demand']]].copy()
#                             df_processed.columns = ['date', 'demand']
                            
#                             df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
#                             df_processed['demand'] = pd.to_numeric(df_processed['demand'], errors='coerce')
#                             df_processed = df_processed.dropna(subset=['date', 'demand'])
#                             df_processed = df_processed.sort_values('date').reset_index(drop=True)
                            
#                             st.session_state.demand_data = df_processed
#                             st.session_state.data_loaded = True
                            
#                             st.success(f"✅ Processed {len(df_processed)} records successfully!")
                            
#                             col1, col2, col3, col4 = st.columns(4)
#                             with col1:
#                                 st.metric("Start Date", df_processed['date'].min().strftime('%Y-%m-%d'))
#                             with col2:
#                                 st.metric("End Date", df_processed['date'].max().strftime('%Y-%m-%d'))
#                             with col3:
#                                 st.metric("Avg Demand", f"{df_processed['demand'].mean():.2f}")
#                             with col4:
#                                 st.metric("Total Demand", f"{df_processed['demand'].sum():.0f}")
                            
#                         except Exception as e:
#                             st.error(f"Error processing data: {str(e)}")
                    
#                     # Show chart selector
#                     if st.session_state.data_loaded and st.session_state.demand_data is not None:
#                         st.markdown("---")
#                         st.markdown("### 📊 Visualize Your Data")
                        
#                         chart_type = st.selectbox(
#                             "Select Chart Type",
#                             options=['line', 'area', 'bar', 'scatter', 'candlestick', 'heatmap', 'box', 'violin'],
#                             format_func=lambda x: {
#                                 'line': '📈 Line Chart',
#                                 'area': '🎨 Area Chart',
#                                 'bar': '📊 Bar Chart',
#                                 'scatter': '🔵 Scatter Plot',
#                                 'candlestick': '🕯️ Candlestick',
#                                 'heatmap': '🔥 Heatmap',
#                                 'box': '📦 Box Plot',
#                                 'violin': '🎻 Violin Plot'
#                             }[x],
#                             key='demand_chart_type'
#                         )
                        
#                         fig = create_enhanced_demand_plot(st.session_state.demand_data, chart_type)
#                         st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.warning("⚠️ Please map at least Date and Demand columns")
            
#             except Exception as e:
#                 st.error(f"Error loading file: {str(e)}")
    
#     with tab2:
#         st.markdown("### Scrape Customer Reviews from URL")
#         st.info("🌟 Universal Web Scraper - Works on ANY website!")
        
#         url_input = st.text_input("Enter URL", placeholder="https://example.com/reviews")
#         max_reviews = st.slider("Max Reviews", 10, 200, 50)
        
#         if st.button("🔍 Start Scraping", type="primary"):
#             if url_input:
#                 with st.spinner("🌐 Scraping website... This may take a moment..."):
#                     try:
#                         scraped_df = scrape_url(url_input, max_items=max_reviews)
                        
#                         if 'error' in scraped_df.columns:
#                             st.error(f"Error scraping URL: {scraped_df['error'].iloc[0]}")
#                             st.info("Generating sample data for demonstration...")
#                             scraped_df = create_sample_review_data(max_reviews)
#                         elif scraped_df.empty:
#                             st.warning("No data found on the website. Generating sample data...")
#                             scraped_df = create_sample_review_data(max_reviews)
                        
#                         if not scraped_df.empty:
#                             st.success(f"✅ Collected {len(scraped_df)} reviews!")
#                             st.session_state.scraped_data = scraped_df
                            
#                             col1, col2, col3, col4 = st.columns(4)
#                             with col1:
#                                 st.metric("Total Reviews", len(scraped_df))
#                             with col2:
#                                 if 'rating' in scraped_df.columns:
#                                     avg_rating = scraped_df['rating'].mean()
#                                     st.metric("Avg Rating", f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A")
#                                 else:
#                                     st.metric("Avg Rating", "N/A")
#                             with col3:
#                                 if 'author' in scraped_df.columns:
#                                     unique_authors = scraped_df['author'].nunique()
#                                     st.metric("Unique Authors", unique_authors)
#                                 else:
#                                     st.metric("Unique Authors", "N/A")
#                             with col4:
#                                 if 'domain' in scraped_df.columns:
#                                     st.metric("Source", scraped_df['domain'].iloc[0] if len(scraped_df) > 0 else "N/A")
#                                 else:
#                                     st.metric("Source", "N/A")
                            
#                             st.dataframe(scraped_df.head(10), use_container_width=True)
                            
#                             st.markdown("### 📥 Download Scraped Data")
#                             csv_data = scraped_df.to_csv(index=False)
#                             domain = urlparse(url_input).netloc.replace(".", "_")
#                             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                             file_name = f"scraped_{domain}_{timestamp}.csv"

#                             st.download_button(
#                                 label="⬇️ Download Scraped CSV",
#                                 data=csv_data,
#                                 file_name=file_name,
#                                 mime="text/csv",
#                                 use_container_width=True
#                             )

#                     except Exception as e:
#                         st.error(f"Error: {str(e)}")
#                         st.info("Generating sample data for demonstration...")
#                         scraped_df = create_sample_review_data(max_reviews)
#                         st.session_state.scraped_data = scraped_df
#                         st.dataframe(scraped_df.head(10), use_container_width=True)
#             else:
#                 st.warning("Please enter a URL")
    
#     with tab3:
#         st.markdown("### Upload Existing Review Data")
        
#         review_file = st.file_uploader("Choose CSV file", type=['csv'], key='review_upload')
        
#         if review_file:
#             try:
#                 review_df = pd.read_csv(review_file)
#                 st.success("✅ Review data uploaded!")
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Total Reviews", len(review_df))
#                 with col2:
#                     st.metric("Columns", len(review_df.columns))
                
#                 st.dataframe(review_df.head(10), use_container_width=True)
                
#                 st.markdown("---")
                
#                 column_mapping = show_column_mapper_ui(review_df, data_type='review', context='review')
                
#                 if column_mapping.get('date') and column_mapping.get('text'):
#                     if st.button("✅ Confirm Review Data", type="primary"):
#                         try:
#                             cols_to_extract = [column_mapping['date'], column_mapping['text']]
#                             new_col_names = ['date', 'review_text']
                            
#                             if column_mapping.get('rating'):
#                                 cols_to_extract.append(column_mapping['rating'])
#                                 new_col_names.append('rating')
                            
#                             processed_review_df = review_df[cols_to_extract].copy()
#                             processed_review_df.columns = new_col_names
                            
#                             processed_review_df['date'] = pd.to_datetime(
#                                 processed_review_df['date'], errors='coerce'
#                             )
                            
#                             processed_review_df = processed_review_df.dropna(subset=['date'])
                            
#                             st.session_state.scraped_data = processed_review_df
                            
#                             st.success(f"✅ Processed {len(processed_review_df)} reviews!")
#                             st.dataframe(processed_review_df.head(10), use_container_width=True)
                            
#                         except Exception as e:
#                             st.error(f"Error: {str(e)}")
#                 else:
#                     st.warning("⚠️ Please map at least Date and Text columns")
            
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")






"""
Data Input Page Module - IMPROVED FOR LARGE DATASETS
Handles CSV uploads, web scraping, and intelligent column mapping
Optimized for datasets with 1M+ rows
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import datetime, timedelta
import json
import hashlib
import random
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go


# ============================================================================
# UNIVERSAL WEB SCRAPER
# ============================================================================

def scrape_url(url: str, max_items: int = 200) -> pd.DataFrame:
    """Universal web scraper for extracting reviews and data from any URL"""
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    PRICE_REGEX = re.compile(r'[₹$€£¥]\s*[\d,.]+')
    RATING_REGEX = re.compile(r'(\d+(\.\d+)?)\s*(/5|stars?)', re.I)
    DATE_REGEX = re.compile(r'\d{4}-\d{2}-\d{2}')

    def clean(text):
        return re.sub(r'\s+', ' ', text).strip() if text else None

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

    soup = BeautifulSoup(r.text, "html.parser")
    domain = urlparse(url).netloc

    page_title = clean(soup.title.text) if soup.title else None

    # Extract JSON-LD
    records = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                records.append({
                    "title": data.get("name"),
                    "review_text": data.get("description"),
                    "rating": data.get("aggregateRating", {}).get("ratingValue"),
                    "price": data.get("offers", {}).get("price"),
                    "date": data.get("datePublished"),
                    "url": data.get("url"),
                    "source_url": url,
                    "domain": domain,
                    "page_title": page_title,
                    "scraped_at": datetime.utcnow()
                })
        except:
            pass

    # HTML Content Blocks
    blocks = soup.find_all(["article", "section", "div", "li"])

    seen = set()

    for block in blocks:
        text = clean(block.get_text(" ", strip=True))
        if not text or len(text) < 40:
            continue

        signature = hashlib.md5(text[:300].encode()).hexdigest()
        if signature in seen:
            continue
        seen.add(signature)

        price = None
        rating = None

        price_match = PRICE_REGEX.search(text)
        if price_match:
            price = float(re.sub(r"[^\d.]", "", price_match.group()))

        rating_match = RATING_REGEX.search(text)
        if rating_match:
            rating = float(rating_match.group(1))

        link = block.find("a", href=True)
        image = block.find("img")

        records.append({
            "title": clean(block.find(["h1","h2","h3","h4","h5","h6"]).text) 
                     if block.find(["h1","h2","h3","h4","h5","h6"]) else None,
            "review_text": text[:500],
            "full_text": text,
            "rating": rating,
            "price": price,
            "url": urljoin(url, link["href"]) if link else None,
            "image_url": urljoin(url, image["src"]) if image and image.get("src") else None,
            "source_url": url,
            "domain": domain,
            "page_title": page_title,
            "scraped_at": datetime.utcnow(),
            "word_count": len(text.split())
        })

        if len(records) >= max_items:
            break

    if not records:
        return pd.DataFrame(columns=[
            "title","review_text","rating","price","url","domain"
        ])

    df = pd.DataFrame(records)

    df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")

    return df


# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================

def create_sample_review_data(num_samples: int = 50) -> pd.DataFrame:
    """Create sample review data for demonstration"""
    
    sample_reviews = [
        "Great product! Exactly what I needed. Fast shipping too.",
        "Not bad but could be better. The quality is okay for the price.",
        "Terrible experience. Product broke after one week.",
        "Love it! Best purchase I've made in a while.",
        "Decent product but overpriced in my opinion.",
        "Five stars! Excellent quality and great customer service.",
        "Would not recommend. Very disappointed with this purchase.",
        "Pretty good overall. Met my expectations.",
        "Amazing! Worth every penny. Highly recommend!",
        "Meh, it's okay. Nothing special but does the job.",
    ]
    
    reviews = []
    base_date = datetime.now()
    
    for i in range(num_samples):
        date = base_date - timedelta(days=random.randint(0, 90))
        rating = random.choice([1, 2, 3, 4, 5])
        
        if rating >= 4:
            text = random.choice([r for r in sample_reviews if any(word in r.lower() for word in ['great', 'excellent', 'love', 'amazing'])])
        elif rating <= 2:
            text = random.choice([r for r in sample_reviews if any(word in r.lower() for word in ['terrible', 'poor', 'disappointed'])])
        else:
            text = random.choice([r for r in sample_reviews if any(word in r.lower() for word in ['okay', 'decent', 'meh'])])
        
        reviews.append({
            'review_text': text,
            'date': date.strftime('%Y-%m-%d'),
            'rating': rating,
            'source': 'sample_data'
        })
    
    return pd.DataFrame(reviews)


# ============================================================================
# COLUMN DETECTION AND MAPPING
# ============================================================================

class ColumnMapper:
    """Intelligent column detection and mapping for various CSV formats"""
    
    DATE_PATTERNS = [
        r'date', r'time', r'datetime', r'timestamp', r'created', r'published',
        r'posted', r'review_date', r'purchase_date', r'order_date', r'day'
    ]
    
    TEXT_PATTERNS = [
        r'review', r'text', r'comment', r'feedback', r'description', r'content',
        r'message', r'body', r'summary', r'opinion', r'note'
    ]
    
    RATING_PATTERNS = [
        r'rating', r'score', r'stars', r'rate', r'grade', r'rank'
    ]
    
    DEMAND_PATTERNS = [
        r'demand', r'sales', r'quantity', r'units', r'volume', r'orders',
        r'sold', r'revenue', r'count', r'total'
    ]
    
    @staticmethod
    def fuzzy_match(column_name: str, patterns: List[str]) -> float:
        """Calculate fuzzy match score between column name and patterns"""
        column_lower = column_name.lower().strip()
        best_score = 0
        
        for pattern in patterns:
            if re.search(pattern, column_lower):
                best_score = max(best_score, 1.0)
            elif pattern in column_lower or column_lower in pattern:
                best_score = max(best_score, 0.7)
        
        return best_score
    
    @classmethod
    def detect_column_type(cls, df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Detect what type of data a column likely contains"""
        scores = {
            'date': 0,
            'text': 0,
            'rating': 0,
            'demand': 0
        }
        
        scores['date'] = cls.fuzzy_match(column, cls.DATE_PATTERNS)
        scores['text'] = cls.fuzzy_match(column, cls.TEXT_PATTERNS)
        scores['rating'] = cls.fuzzy_match(column, cls.RATING_PATTERNS)
        scores['demand'] = cls.fuzzy_match(column, cls.DEMAND_PATTERNS)
        
        sample = df[column].dropna().head(20)
        
        if len(sample) > 0:
            try:
                pd.to_datetime(sample)
                scores['date'] += 0.5
            except:
                pass
            
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].max() <= 10 and df[column].min() >= 0:
                    scores['rating'] += 0.3
                scores['demand'] += 0.2
            
            if pd.api.types.is_string_dtype(df[column]):
                avg_length = sample.astype(str).str.len().mean()
                if avg_length > 20:
                    scores['text'] += 0.5
        
        return scores
    
    @classmethod
    def auto_detect_columns(cls, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Automatically detect and map columns to required fields"""
        mapping = {
            'date': None,
            'text': None,
            'rating': None,
            'demand': None
        }
        
        column_scores = {}
        for column in df.columns:
            column_scores[column] = cls.detect_column_type(df, column)
        
        for col_type in mapping.keys():
            best_col = None
            best_score = 0
            
            for column, scores in column_scores.items():
                if scores[col_type] > best_score:
                    best_score = scores[col_type]
                    best_col = column
            
            if best_score > 0.3:
                mapping[col_type] = best_col
        
        return mapping
    
    @classmethod
    def get_mapping_confidence(cls, df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Dict[str, float]:
        """Get confidence scores for each mapped column"""
        confidence = {}
        
        for col_type, column in mapping.items():
            if column is not None:
                scores = cls.detect_column_type(df, column)
                confidence[col_type] = scores[col_type]
            else:
                confidence[col_type] = 0.0
        
        return confidence


def show_column_mapper_ui(df: pd.DataFrame, data_type: str = 'review', context: str = 'main') -> Dict[str, str]:
    """Display interactive UI for column mapping"""
    
    mapper = ColumnMapper()
    auto_mapping = mapper.auto_detect_columns(df)
    confidence = mapper.get_mapping_confidence(df, auto_mapping)
    
    st.markdown("#### 🎯 Smart Column Detection")
    
    cols = st.columns(4)
    for idx, (field, column) in enumerate(auto_mapping.items()):
        with cols[idx % 4]:
            if column:
                conf_pct = confidence[field] * 100
                color = "🟢" if conf_pct > 70 else "🟡" if conf_pct > 40 else "🔴"
                st.metric(field.title(), column, f"{color} {conf_pct:.0f}%")
            else:
                st.metric(field.title(), "Not detected", "🔴 0%")
    
    st.markdown("---")
    st.markdown("#### Adjust Mappings (if needed)")
    
    final_mapping = {}
    
    if data_type == 'review':
        required_fields = {
            'date': 'Date/Time column',
            'text': 'Review/Comment text',
            'rating': 'Rating (optional)'
        }
    else:
        required_fields = {
            'date': 'Date/Time column',
            'demand': 'Demand/Sales/Quantity'
        }
    
    col1, col2 = st.columns(2)
    
    for idx, (field, description) in enumerate(required_fields.items()):
        with col1 if idx % 2 == 0 else col2:
            default_val = auto_mapping.get(field)
            default_idx = 0
            
            options = ['-- None --'] + list(df.columns)
            
            if default_val and default_val in df.columns:
                default_idx = options.index(default_val)
            
            selected = st.selectbox(
                f"**{description}**",
                options=options,
                index=default_idx,
                key=f"{context}_map_{field}"
            )
            
            if selected != '-- None --':
                final_mapping[field] = selected
    
    return final_mapping


# ============================================================================
# LARGE DATASET OPTIMIZATION
# ============================================================================

def prepare_large_dataset(df, max_points=10000, aggregation_method='adaptive'):
    """
    Optimize large datasets for visualization by intelligent sampling/aggregation
    
    Args:
        df: DataFrame with 'date' and 'demand' columns
        max_points: Maximum number of points to display
        aggregation_method: 'adaptive', 'sample', 'aggregate', 'lttb', 'minmax'
    
    Returns:
        Optimized DataFrame suitable for visualization
    """
    if len(df) <= max_points:
        return df, 'original'
    
    df_viz = df.copy()
    
    if aggregation_method == 'sample':
        # Stratified random sampling - preserves distribution
        return df_viz.sample(n=max_points, random_state=42).sort_values('date').reset_index(drop=True), 'sampled'
    
    elif aggregation_method == 'minmax':
        # Min-Max downsampling - preserves peaks and valleys
        n_bins = max_points // 2
        df_viz['bin'] = pd.cut(range(len(df_viz)), bins=n_bins, labels=False)
        
        mins = df_viz.groupby('bin').apply(lambda x: x.loc[x['demand'].idxmin()]).reset_index(drop=True)
        maxs = df_viz.groupby('bin').apply(lambda x: x.loc[x['demand'].idxmax()]).reset_index(drop=True)
        
        combined = pd.concat([mins, maxs]).drop_duplicates().sort_values('date').reset_index(drop=True)
        return combined.drop(columns=['bin'], errors='ignore'), 'minmax'
    
    elif aggregation_method == 'aggregate':
        # Time-based aggregation with statistics
        n_bins = min(max_points, len(df))
        df_viz['time_bin'] = pd.cut(range(len(df_viz)), bins=n_bins, labels=False)
        
        aggregated = df_viz.groupby('time_bin').agg({
            'date': 'first',
            'demand': ['mean', 'min', 'max', 'std', 'count']
        }).reset_index()
        
        aggregated.columns = ['time_bin', 'date', 'demand', 'demand_min', 'demand_max', 'demand_std', 'point_count']
        return aggregated, 'aggregated'
    
    elif aggregation_method == 'lttb':
        # Largest Triangle Three Buckets - preserves visual shape
        return lttb_downsample(df_viz, max_points), 'lttb'
    
    else:  # 'adaptive'
        # Adaptive: aggregate by time period based on data range
        date_range = (df_viz['date'].max() - df_viz['date'].min()).days
        
        if date_range > 365 * 3:  # > 3 years
            freq = 'W'  # Weekly
            method_used = 'weekly_avg'
        elif date_range > 365:  # > 1 year
            freq = 'D'  # Daily
            method_used = 'daily_avg'
        elif date_range > 90:  # > 3 months
            freq = '6H'  # 6-hourly
            method_used = '6hour_avg'
        else:
            freq = 'H'  # Hourly
            method_used = 'hourly_avg'
        
        df_viz = df_viz.set_index('date').resample(freq).agg({
            'demand': ['mean', 'min', 'max', 'std', 'count']
        }).reset_index()
        
        df_viz.columns = ['date', 'demand', 'demand_min', 'demand_max', 'demand_std', 'point_count']
        return df_viz, method_used


def lttb_downsample(df, threshold):
    """
    Largest Triangle Three Buckets (LTTB) downsampling algorithm
    Preserves visual appearance better than random sampling
    """
    if len(df) <= threshold:
        return df
    
    data = df[['date', 'demand']].copy()
    data['x'] = range(len(data))
    
    # Convert to numpy for faster processing
    sampled_indices = [0]  # Always keep first point
    
    bucket_size = (len(data) - 2) / (threshold - 2)
    
    a = 0  # Initially point a is first point
    
    for i in range(threshold - 2):
        # Calculate point average for next bucket
        avg_range_start = int((i + 1) * bucket_size) + 1
        avg_range_end = int((i + 2) * bucket_size) + 1
        avg_range_end = min(avg_range_end, len(data))
        
        avg_x = data.iloc[avg_range_start:avg_range_end]['x'].mean()
        avg_y = data.iloc[avg_range_start:avg_range_end]['demand'].mean()
        
        # Get current bucket range
        range_start = int(i * bucket_size) + 1
        range_end = int((i + 1) * bucket_size) + 1
        
        # Find point in current bucket with largest triangle
        max_area = -1
        max_area_point = range_start
        
        for idx in range(range_start, range_end):
            # Calculate triangle area
            area = abs(
                (data.iloc[a]['x'] - avg_x) * (data.iloc[idx]['demand'] - data.iloc[a]['demand']) -
                (data.iloc[a]['x'] - data.iloc[idx]['x']) * (avg_y - data.iloc[a]['demand'])
            ) * 0.5
            
            if area > max_area:
                max_area = area
                max_area_point = idx
        
        sampled_indices.append(max_area_point)
        a = max_area_point
    
    sampled_indices.append(len(data) - 1)  # Always keep last point
    
    return df.iloc[sampled_indices].reset_index(drop=True)


# ============================================================================
# ENHANCED VISUALIZATIONS FOR LARGE DATASETS
# ============================================================================

def create_enhanced_demand_plot(df, chart_type='line', max_points=10000, aggregation_method='adaptive'):
    """Create enhanced demand visualization optimized for large datasets"""
    
    # Optimize dataset if needed
    original_size = len(df)
    df_viz, optimization_method = prepare_large_dataset(df, max_points, aggregation_method)
    optimized_size = len(df_viz)
    
    # Check if we have aggregated data with statistical columns
    has_stats = all(col in df_viz.columns for col in ['demand_min', 'demand_max'])
    
    if chart_type == 'line':
        fig = go.Figure()
        
        if has_stats:
            # Show uncertainty band
            fig.add_trace(go.Scatter(
                x=df_viz['date'],
                y=df_viz['demand_max'],
                mode='lines',
                name='Max',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=df_viz['date'],
                y=df_viz['demand_min'],
                mode='lines',
                name='Range',
                fill='tonexty',
                fillcolor='rgba(99, 102, 241, 0.2)',
                line=dict(width=0),
                showlegend=True
            ))
        
        fig.add_trace(go.Scatter(
            x=df_viz['date'],
            y=df_viz['demand'],
            mode='lines',
            name='Demand',
            line=dict(color='#6366f1', width=2),
        ))
        
    elif chart_type == 'scatter':
        # Scattergl for large datasets (WebGL acceleration)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=df_viz['date'],
            y=df_viz['demand'],
            mode='markers',
            marker=dict(
                size=3,
                color=df_viz['demand'],
                colorscale='Purples',
                showscale=True,
                colorbar=dict(title="Demand"),
                opacity=0.6
            ),
            name='Demand'
        ))
        
    elif chart_type == 'heatmap_calendar':
        # Calendar heatmap - great for large temporal datasets
        df_heat = df_viz.copy()
        df_heat['year'] = df_heat['date'].dt.year
        df_heat['month'] = df_heat['date'].dt.month
        df_heat['day'] = df_heat['date'].dt.day
        
        pivot = df_heat.pivot_table(
            values='demand',
            index='day',
            columns=['year', 'month'],
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{y}-{m:02d}" for y, m in pivot.columns],
            y=pivot.index,
            colorscale='Purples',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            xaxis_title='Year-Month',
            yaxis_title='Day of Month'
        )
        
    elif chart_type == 'density_heatmap':
        # 2D histogram for density visualization
        fig = go.Figure()
        
        # Create time-based bins
        time_numeric = (df_viz['date'] - df_viz['date'].min()).dt.total_seconds()
        
        fig.add_trace(go.Histogram2d(
            x=time_numeric,
            y=df_viz['demand'],
            colorscale='Purples',
            showscale=True,
            nbinsx=100,
            nbinsy=50
        ))
        
        fig.update_layout(
            xaxis_title='Time →',
            yaxis_title='Demand'
        )
        
    elif chart_type == 'candlestick':
        if has_stats:
            # Use pre-computed stats
            df_candle = df_viz.copy()
            df_candle['open'] = df_candle['demand']
            df_candle['close'] = df_candle['demand']
            df_candle['high'] = df_candle['demand_max']
            df_candle['low'] = df_candle['demand_min']
        else:
            # Compute rolling stats
            df_candle = df_viz.copy()
            window = max(7, len(df_viz) // 100)
            df_candle['high'] = df_candle['demand'].rolling(window, center=True).max()
            df_candle['low'] = df_candle['demand'].rolling(window, center=True).min()
            df_candle['open'] = df_candle['demand'].shift(1)
            df_candle['close'] = df_candle['demand']
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_candle['date'],
            open=df_candle['open'],
            high=df_candle['high'],
            low=df_candle['low'],
            close=df_candle['close'],
            increasing_line_color='#6366f1',
            decreasing_line_color='#ef4444'
        )])
        
    elif chart_type == 'area':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_viz['date'],
            y=df_viz['demand'],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.3)',
            line=dict(color='#6366f1', width=2),
            name='Demand'
        ))
        
    elif chart_type == 'bar':
        # Use aggregated data for bars
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_viz['date'],
            y=df_viz['demand'],
            marker=dict(
                color=df_viz['demand'],
                colorscale='Purples',
                showscale=True,
                colorbar=dict(title="Demand")
            ),
            name='Demand'
        ))
        
    elif chart_type == 'box':
        # Monthly box plots
        df_box = df_viz.copy()
        df_box['month'] = df_box['date'].dt.to_period('M').astype(str)
        
        fig = px.box(
            df_box,
            x='month',
            y='demand',
            color_discrete_sequence=['#6366f1']
        )
        fig.update_traces(marker=dict(opacity=0.7))
        
    elif chart_type == 'violin':
        # Monthly violin plots
        df_vio = df_viz.copy()
        df_vio['month'] = df_vio['date'].dt.to_period('M').astype(str)
        
        fig = px.violin(
            df_vio,
            x='month',
            y='demand',
            box=True,
            points='outliers',  # Only show outliers for large datasets
            color_discrete_sequence=['#6366f1']
        )
        
    elif chart_type == 'rolling_stats':
        # Rolling statistics view
        window = max(7, len(df_viz) // 100)
        df_roll = df_viz.copy()
        df_roll['rolling_mean'] = df_roll['demand'].rolling(window, center=True).mean()
        df_roll['rolling_std'] = df_roll['demand'].rolling(window, center=True).std()
        df_roll['upper_band'] = df_roll['rolling_mean'] + 2 * df_roll['rolling_std']
        df_roll['lower_band'] = df_roll['rolling_mean'] - 2 * df_roll['rolling_std']
        
        fig = go.Figure()
        
        # Confidence band
        fig.add_trace(go.Scatter(
            x=df_roll['date'],
            y=df_roll['upper_band'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df_roll['date'],
            y=df_roll['lower_band'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(99, 102, 241, 0.2)',
            line=dict(width=0),
            name='±2σ Band'
        ))
        
        # Rolling mean
        fig.add_trace(go.Scatter(
            x=df_roll['date'],
            y=df_roll['rolling_mean'],
            mode='lines',
            line=dict(color='#6366f1', width=3),
            name='Rolling Mean'
        ))
        
        # Actual data (sampled)
        sample_indices = np.random.choice(len(df_roll), size=min(1000, len(df_roll)), replace=False)
        sample_data = df_roll.iloc[sorted(sample_indices)]
        
        fig.add_trace(go.Scattergl(
            x=sample_data['date'],
            y=sample_data['demand'],
            mode='markers',
            marker=dict(size=3, color='#9333ea', opacity=0.3),
            name='Sample Points'
        ))
    
    else:  # Default to line
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_viz['date'],
            y=df_viz['demand'],
            mode='lines',
            line=dict(color='#6366f1', width=2),
            name='Demand'
        ))
    
    # Common styling
    title_text = '<b>Historical Demand Trend</b>'
    if original_size != optimized_size:
        title_text += f'<br><sub>Showing {optimized_size:,} of {original_size:,} points ({optimization_method})</sub>'
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=24, family='Arial Black', color='#ffffff')
        ),
        plot_bgcolor='#1f2937',
        paper_bgcolor='#111827',
        font=dict(color='#ffffff', family='Arial'),
        xaxis=dict(
            title='<b>Date</b>',
            gridcolor='#374151',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='<b>Demand</b>',
            gridcolor='#374151',
            showgrid=True,
            zeroline=False
        ),
        hovermode='x unified',
        height=500,
        margin=dict(t=100, b=60, l=60, r=40)
    )
    
    return fig


# ============================================================================
# DATA INPUT PAGE
# ============================================================================

def show_data_input_page():
    """Data input page with intelligent column mapping and large dataset support"""
    
    # Add performance settings in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Visualization Settings")
        
        max_viz_points = st.select_slider(
            "Max Visualization Points",
            options=[1000, 5000, 10000, 25000, 50000, 100000],
            value=10000,
            help="Reduce for faster rendering with large datasets"
        )
        
        agg_method = st.selectbox(
            "Aggregation Method",
            options=['adaptive', 'lttb', 'minmax', 'aggregate', 'sample'],
            format_func=lambda x: {
                'adaptive': '🎯 Adaptive (Recommended)',
                'lttb': '📐 LTTB (Shape Preserving)',
                'minmax': '📊 Min-Max (Peak Preserving)',
                'aggregate': '📈 Time Aggregation',
                'sample': '🎲 Random Sample'
            }[x],
            help="Choose how to optimize large datasets"
        )
        
        st.session_state.max_viz_points = max_viz_points
        st.session_state.agg_method = agg_method
    
    tab1, tab2, tab3 = st.tabs(["📁 Upload Demand Data", "🌐 Scrape Reviews", "📋 Upload Review Data"])
    
    with tab1:
        st.markdown("### Upload Historical Demand Data")
        st.info("💡 **Large Dataset Support**: Optimized for datasets with millions of rows!")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key='demand_upload')
        
        if uploaded_file:
            try:
                # Read with progress indicator
                with st.spinner("📥 Loading CSV file..."):
                    df = pd.read_csv(uploaded_file)
                
                st.success("✅ Data uploaded successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.metric("Memory", f"{memory_mb:.1f} MB")
                with col4:
                    if len(df) > 100000:
                        st.metric("Dataset Size", "🔥 Large", help="Optimizations enabled")
                    else:
                        st.metric("Dataset Size", "✅ Normal")
                
                st.markdown("#### Raw Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown("---")
                
                column_mapping = show_column_mapper_ui(df, data_type='demand', context='demand')
                
                if column_mapping.get('date') and column_mapping.get('demand'):
                    if st.button("✅ Confirm and Process Data", type="primary"):
                        try:
                            with st.spinner("🔄 Processing data..."):
                                # Store the FULL original CSV for Custom BI
                                st.session_state.original_csv_data = df.copy()
                                
                                # Process only date and demand for forecasting
                                df_processed = df[[column_mapping['date'], column_mapping['demand']]].copy()
                                df_processed.columns = ['date', 'demand']
                                
                                df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
                                df_processed['demand'] = pd.to_numeric(df_processed['demand'], errors='coerce')
                                df_processed = df_processed.dropna(subset=['date', 'demand'])
                                df_processed = df_processed.sort_values('date').reset_index(drop=True)
                                
                                st.session_state.demand_data = df_processed
                                st.session_state.data_loaded = True
                            
                            st.success(f"✅ Processed {len(df_processed):,} records successfully!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Start Date", df_processed['date'].min().strftime('%Y-%m-%d'))
                            with col2:
                                st.metric("End Date", df_processed['date'].max().strftime('%Y-%m-%d'))
                            with col3:
                                st.metric("Avg Demand", f"{df_processed['demand'].mean():.2f}")
                            with col4:
                                st.metric("Total Demand", f"{df_processed['demand'].sum():.0f}")
                            
                        except Exception as e:
                            st.error(f"Error processing data: {str(e)}")
                    
                    # Show chart selector
                    if st.session_state.get('data_loaded') and st.session_state.get('demand_data') is not None:
                        st.markdown("---")
                        st.markdown("### 📊 Visualize Your Data")
                        
                        # Recommend chart types based on dataset size
                        dataset_size = len(st.session_state.demand_data)
                        
                        if dataset_size > 100000:
                            recommended_charts = ['scatter', 'heatmap_calendar', 'density_heatmap', 'rolling_stats']
                            st.info(f"📊 **Large Dataset Detected** ({dataset_size:,} rows): Recommended chart types are optimized for performance")
                        else:
                            recommended_charts = ['line', 'area', 'candlestick', 'bar']
                        
                        chart_options = {
                            'line': '📈 Line Chart',
                            'area': '🎨 Area Chart',
                            'scatter': '🔵 Scatter Plot (WebGL)',
                            'bar': '📊 Bar Chart',
                            'candlestick': '🕯️ Candlestick',
                            'heatmap_calendar': '📅 Calendar Heatmap',
                            'density_heatmap': '🔥 Density Heatmap',
                            'box': '📦 Box Plot',
                            'violin': '🎻 Violin Plot',
                            'rolling_stats': '📉 Rolling Statistics'
                        }
                        
                        chart_type = st.selectbox(
                            "Select Chart Type",
                            options=list(chart_options.keys()),
                            format_func=lambda x: chart_options[x] + (' ⭐ Recommended' if x in recommended_charts else ''),
                            key='demand_chart_type'
                        )
                        
                        with st.spinner("🎨 Rendering chart..."):
                            fig = create_enhanced_demand_plot(
                                st.session_state.demand_data,
                                chart_type,
                                max_points=st.session_state.get('max_viz_points', 10000),
                                aggregation_method=st.session_state.get('agg_method', 'adaptive')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Please map at least Date and Demand columns")
            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.markdown("### Scrape Customer Reviews from URL")
        st.info("🌟 Universal Web Scraper - Works on ANY website!")
        
        url_input = st.text_input("Enter URL", placeholder="https://example.com/reviews")
        max_reviews = st.slider("Max Reviews", 10, 200, 50)
        
        if st.button("🔍 Start Scraping", type="primary"):
            if url_input:
                with st.spinner("🌐 Scraping website... This may take a moment..."):
                    try:
                        scraped_df = scrape_url(url_input, max_items=max_reviews)
                        
                        if 'error' in scraped_df.columns:
                            st.error(f"Error scraping URL: {scraped_df['error'].iloc[0]}")
                            st.info("Generating sample data for demonstration...")
                            scraped_df = create_sample_review_data(max_reviews)
                        elif scraped_df.empty:
                            st.warning("No data found on the website. Generating sample data...")
                            scraped_df = create_sample_review_data(max_reviews)
                        
                        if not scraped_df.empty:
                            st.success(f"✅ Collected {len(scraped_df)} reviews!")
                            st.session_state.scraped_data = scraped_df
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Reviews", len(scraped_df))
                            with col2:
                                if 'rating' in scraped_df.columns:
                                    avg_rating = scraped_df['rating'].mean()
                                    st.metric("Avg Rating", f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A")
                                else:
                                    st.metric("Avg Rating", "N/A")
                            with col3:
                                if 'author' in scraped_df.columns:
                                    unique_authors = scraped_df['author'].nunique()
                                    st.metric("Unique Authors", unique_authors)
                                else:
                                    st.metric("Unique Authors", "N/A")
                            with col4:
                                if 'domain' in scraped_df.columns:
                                    st.metric("Source", scraped_df['domain'].iloc[0] if len(scraped_df) > 0 else "N/A")
                                else:
                                    st.metric("Source", "N/A")
                            
                            st.dataframe(scraped_df.head(10), use_container_width=True)
                            
                            st.markdown("### 📥 Download Scraped Data")
                            csv_data = scraped_df.to_csv(index=False)
                            domain = urlparse(url_input).netloc.replace(".", "_")
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_name = f"scraped_{domain}_{timestamp}.csv"

                            st.download_button(
                                label="⬇️ Download Scraped CSV",
                                data=csv_data,
                                file_name=file_name,
                                mime="text/csv",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Generating sample data for demonstration...")
                        scraped_df = create_sample_review_data(max_reviews)
                        st.session_state.scraped_data = scraped_df
                        st.dataframe(scraped_df.head(10), use_container_width=True)
            else:
                st.warning("Please enter a URL")
    
    with tab3:
        st.markdown("### Upload Existing Review Data")
        
        review_file = st.file_uploader("Choose CSV file", type=['csv'], key='review_upload')
        
        if review_file:
            try:
                review_df = pd.read_csv(review_file)
                st.success("✅ Review data uploaded!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Reviews", f"{len(review_df):,}")
                with col2:
                    st.metric("Columns", len(review_df.columns))
                
                st.dataframe(review_df.head(10), use_container_width=True)
                
                st.markdown("---")
                
                column_mapping = show_column_mapper_ui(review_df, data_type='review', context='review')
                
                if column_mapping.get('date') and column_mapping.get('text'):
                    if st.button("✅ Confirm Review Data", type="primary"):
                        try:
                            cols_to_extract = [column_mapping['date'], column_mapping['text']]
                            new_col_names = ['date', 'review_text']
                            
                            if column_mapping.get('rating'):
                                cols_to_extract.append(column_mapping['rating'])
                                new_col_names.append('rating')
                            
                            processed_review_df = review_df[cols_to_extract].copy()
                            processed_review_df.columns = new_col_names
                            
                            processed_review_df['date'] = pd.to_datetime(
                                processed_review_df['date'], errors='coerce'
                            )
                            
                            processed_review_df = processed_review_df.dropna(subset=['date'])
                            
                            st.session_state.scraped_data = processed_review_df
                            
                            st.success(f"✅ Processed {len(processed_review_df):,} reviews!")
                            st.dataframe(processed_review_df.head(10), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("⚠️ Please map at least Date and Text columns")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # For standalone testing
    import streamlit as st
    
    st.set_page_config(page_title="Data Input - Large Dataset Demo", layout="wide")
    
    if 'demand_data' not in st.session_state:
        st.session_state.demand_data = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    
    show_data_input_page()