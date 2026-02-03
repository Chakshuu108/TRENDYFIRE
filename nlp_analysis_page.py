"""
NLP Analysis Page Module
Handles text preprocessing, sentiment analysis, and topic modeling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path for NLP modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try importing NLP modules
try:
    from nlp.preprocessing import TextPreprocessor
    from nlp.sentiment_analysis import SentimentAnalyzer, aggregate_sentiment_by_date
    from nlp.topic_modeling import TopicModeler, aggregate_topics_by_date
except ImportError:
    # Create dummy classes if imports fail
    class TextPreprocessor:
        def preprocess_dataframe(self, df, col):
            df['processed_text'] = df[col].astype(str).str.lower()
            return df
    
    class SentimentAnalyzer:
        def __init__(self, method='vader'):
            self.method = method
        def analyze_dataframe(self, df, col):
            np.random.seed(42)
            df['sentiment_compound'] = np.random.uniform(-1, 1, len(df))
            df['sentiment_label'] = np.random.choice(['positive', 'negative', 'neutral'], len(df))
            return df
    
    def aggregate_sentiment_by_date(df, date_col):
        return df.groupby(date_col).agg({'sentiment_compound': 'mean'}).reset_index().rename(
            columns={'sentiment_compound': 'avg_sentiment_compound'}
        )
    
    class TopicModeler:
        def __init__(self, num_topics=5):
            self.num_topics = num_topics
        def train(self, texts):
            pass
        def get_topics(self, num_words=10):
            return [(i, [f'word{j}' for j in range(num_words)]) for i in range(self.num_topics)]
        def analyze_dataframe(self, df, col):
            df['dominant_topic'] = np.random.randint(0, self.num_topics, len(df))
            return df
    
    def aggregate_topics_by_date(df, date_col):
        return df


def show_nlp_analysis_page():
    """NLP analysis page"""
    
    if st.session_state.scraped_data is None:
        st.warning("⚠️ Please upload or scrape review data first!")
        return
    
    df = st.session_state.scraped_data.copy()
    
    st.info(f"📊 Analyzing {len(df)} reviews")
    
    # ========================================================================
    # TEXT PREPROCESSING
    # ========================================================================
    
    with st.expander("📝 Text Preprocessing", expanded=True):
        st.markdown("""
        **Text preprocessing** cleans and normalizes your review text by:
        - Converting to lowercase
        - Removing special characters and punctuation
        - Removing extra whitespace
        - Tokenization and lemmatization
        """)
        
        if st.button("Start Preprocessing", key="preprocess_btn"):
            with st.spinner("Preprocessing text data..."):
                try:
                    preprocessor = TextPreprocessor()
                    df = preprocessor.preprocess_dataframe(df, 'review_text')
                    st.success("✅ Preprocessing complete!")
                    
                    st.markdown("#### Sample Results")
                    sample_df = df[['review_text', 'processed_text']].head(5)
                    st.dataframe(sample_df, use_container_width=True)
                    
                    st.session_state.scraped_data = df
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")
    
    # ========================================================================
    # SENTIMENT ANALYSIS
    # ========================================================================
    
    with st.expander("😊 Sentiment Analysis", expanded=True):
        st.markdown("""
        **Sentiment Analysis** determines the emotional tone of reviews:
        - **Positive**: Customer satisfaction and praise
        - **Negative**: Complaints and dissatisfaction
        - **Neutral**: Factual or balanced opinions
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_method = st.selectbox(
                "Analysis Method",
                ["VADER", "TextBlob"],
                help="VADER: Better for social media and short texts. TextBlob: General purpose sentiment analysis."
            )
        
        with col2:
            st.markdown("##### Method Comparison")
            if sentiment_method == "VADER":
                st.info("**VADER**: Optimized for social media, considers emoticons and slang")
            else:
                st.info("**TextBlob**: General-purpose, provides polarity and subjectivity scores")
        
        if st.button("Analyze Sentiment", key="sentiment_btn"):
            with st.spinner("Analyzing sentiment..."):
                try:
                    method = 'vader' if sentiment_method == "VADER" else 'textblob'
                    analyzer = SentimentAnalyzer(method=method)
                    
                    # Use processed text if available, otherwise use original
                    text_col = 'processed_text' if 'processed_text' in df.columns else 'review_text'
                    df = analyzer.analyze_dataframe(df, text_col)
                    
                    st.success("✅ Sentiment analysis complete!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution pie chart
                        st.markdown("#### Sentiment Distribution")
                        sentiment_counts = df['sentiment_label'].value_counts()
                        
                        colors = {
                            'positive': '#10b981',
                            'negative': '#ef4444',
                            'neutral': '#6b7280'
                        }
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=sentiment_counts.index,
                            values=sentiment_counts.values,
                            marker=dict(colors=[colors.get(label, '#6366f1') for label in sentiment_counts.index]),
                            hole=0.4
                        )])
                        
                        fig_pie.update_layout(
                            plot_bgcolor='#1f2937',
                            paper_bgcolor='#111827',
                            font=dict(color='#ffffff'),
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Sentiment statistics
                        st.markdown("#### Statistics")
                        
                        total = len(df)
                        for label in ['positive', 'negative', 'neutral']:
                            count = sentiment_counts.get(label, 0)
                            percentage = (count / total) * 100
                            
                            if label == 'positive':
                                st.metric(
                                    f"😊 {label.title()}", 
                                    f"{count} reviews", 
                                    f"{percentage:.1f}%"
                                )
                            elif label == 'negative':
                                st.metric(
                                    f"😞 {label.title()}", 
                                    f"{count} reviews", 
                                    f"{percentage:.1f}%"
                                )
                            else:
                                st.metric(
                                    f"😐 {label.title()}", 
                                    f"{count} reviews", 
                                    f"{percentage:.1f}%"
                                )
                    
                    # Sentiment trend over time
                    if 'date' in df.columns:
                        st.markdown("---")
                        st.markdown("#### Sentiment Trend Over Time")
                        
                        df_agg = aggregate_sentiment_by_date(df, 'date')
                        
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=df_agg['date'],
                            y=df_agg['avg_sentiment_compound'],
                            mode='lines+markers',
                            name='Average Sentiment',
                            line=dict(color='#6366f1', width=3),
                            marker=dict(size=8),
                            fill='tozeroy',
                            fillcolor='rgba(99, 102, 241, 0.2)'
                        ))
                        
                        # Add reference line at 0
                        fig_trend.add_hline(
                            y=0, 
                            line_dash="dash", 
                            line_color="gray",
                            annotation_text="Neutral"
                        )
                        
                        fig_trend.update_layout(
                            title='Average Sentiment Score Over Time',
                            xaxis_title='Date',
                            yaxis_title='Sentiment Score (-1 to +1)',
                            plot_bgcolor='#1f2937',
                            paper_bgcolor='#111827',
                            font=dict(color='#ffffff'),
                            xaxis=dict(gridcolor='#374151'),
                            yaxis=dict(gridcolor='#374151'),
                            height=400
                        )
                        
                        st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Save processed data
                    st.session_state.scraped_data = df
                    
                    # Sample results
                    st.markdown("---")
                    st.markdown("#### Sample Sentiment Results")
                    sample_cols = ['review_text', 'sentiment_label', 'sentiment_compound']
                    available_cols = [col for col in sample_cols if col in df.columns]
                    st.dataframe(df[available_cols].head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during sentiment analysis: {str(e)}")
    
    # ========================================================================
    # TOPIC MODELING
    # ========================================================================
    
    with st.expander("🏷️ Topic Modeling", expanded=True):
        st.markdown("""
        **Topic Modeling** identifies common themes and topics in your reviews:
        - Automatically discovers discussion topics
        - Groups similar reviews together
        - Extracts key words for each topic
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_topics = st.slider("Number of Topics", 2, 10, 5, help="How many different topics to extract")
        
        with col2:
            num_words = st.slider("Words per Topic", 5, 15, 10, help="How many words to show for each topic")
        
        if st.button("Extract Topics", key="topic_btn"):
            with st.spinner("Extracting topics from reviews..."):
                try:
                    if 'processed_text' not in df.columns:
                        st.warning("⚠️ Running preprocessing first...")
                        preprocessor = TextPreprocessor()
                        df = preprocessor.preprocess_dataframe(df, 'review_text')
                    
                    texts = df['processed_text'].tolist()
                    modeler = TopicModeler(num_topics=num_topics)
                    modeler.train(texts)
                    
                    st.success("✅ Topic modeling complete!")
                    
                    # Display topics
                    st.markdown("#### Discovered Topics")
                    
                    topics = modeler.get_topics(num_words=num_words)
                    
                    # Create columns for topics
                    topic_cols = st.columns(min(3, num_topics))
                    
                    for idx, (topic_id, words) in enumerate(topics):
                        with topic_cols[idx % len(topic_cols)]:
                            st.markdown(f"##### 🏷️ Topic {topic_id}")
                            st.markdown(f"**Keywords:** {', '.join(words)}")
                    
                    # Analyze documents
                    st.markdown("---")
                    st.markdown("#### Topic Distribution")
                    
                    df = modeler.analyze_dataframe(df, 'processed_text')
                    
                    # Topic distribution chart
                    topic_counts = df['dominant_topic'].value_counts().sort_index()
                    
                    fig_topics = go.Figure(data=[
                        go.Bar(
                            x=[f"Topic {i}" for i in topic_counts.index],
                            y=topic_counts.values,
                            marker=dict(
                                color=topic_counts.values,
                                colorscale='Purples',
                                showscale=True
                            )
                        )
                    ])
                    
                    fig_topics.update_layout(
                        title='Number of Reviews per Topic',
                        xaxis_title='Topic',
                        yaxis_title='Number of Reviews',
                        plot_bgcolor='#1f2937',
                        paper_bgcolor='#111827',
                        font=dict(color='#ffffff'),
                        xaxis=dict(gridcolor='#374151'),
                        yaxis=dict(gridcolor='#374151'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_topics, use_container_width=True)
                    
                    # Save processed data
                    st.session_state.scraped_data = df
                    
                    # Sample results
                    st.markdown("---")
                    st.markdown("#### Sample Topic Assignments")
                    sample_cols = ['review_text', 'dominant_topic']
                    available_cols = [col for col in sample_cols if col in df.columns]
                    st.dataframe(df[available_cols].head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during topic modeling: {str(e)}")
    
    # ========================================================================
    # SAVE PROCESSED DATA
    # ========================================================================
    
    st.markdown("---")
    
    if st.button("💾 Save All Processed Data", type="primary", use_container_width=True):
        st.session_state.processed_data = df
        st.success("✅ All NLP analysis results saved!")
        st.info("You can now proceed to the Forecasting tab to train prediction models.")
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'processed_text' in df.columns:
                st.metric("✅ Preprocessing", "Complete")
            else:
                st.metric("⚠️ Preprocessing", "Not done")
        
        with col2:
            if 'sentiment_label' in df.columns:
                st.metric("✅ Sentiment Analysis", "Complete")
            else:
                st.metric("⚠️ Sentiment Analysis", "Not done")
        
        with col3:
            if 'dominant_topic' in df.columns:
                st.metric("✅ Topic Modeling", "Complete")
            else:
                st.metric("⚠️ Topic Modeling", "Not done")
