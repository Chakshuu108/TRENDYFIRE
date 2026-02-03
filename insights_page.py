"""
Insights Page Module
Generates strategic recommendations and comprehensive PDF reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# Try importing PDF generation libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ============================================================================
# PDF REPORT GENERATION
# ============================================================================

def generate_comprehensive_pdf_report(demand_data, forecast_data, metrics, sentiment_data=None, 
                                     recommendations=None, model_name=None):
    """Generate comprehensive PDF report with graphs and detailed analysis"""
    if not PDF_AVAILABLE:
        st.error("PDF generation requires reportlab and matplotlib libraries")
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#3b82f6'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )
        
        # Title page
        story.append(Paragraph("Demand Forecasting Report", title_style))
        story.append(Paragraph(f"with NLP Insights & Predictive Analytics", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        summary_data = []
        if demand_data is not None:
            summary_data.append(['Total Historical Records', str(len(demand_data))])
            summary_data.append(['Date Range', f"{demand_data['date'].min().strftime('%Y-%m-%d')} to {demand_data['date'].max().strftime('%Y-%m-%d')}"])
            summary_data.append(['Average Demand', f"{demand_data['demand'].mean():.2f}"])
            summary_data.append(['Total Demand', f"{demand_data['demand'].sum():.0f}"])
        
        if forecast_data is not None:
            summary_data.append(['Forecast Horizon', f"{len(forecast_data)} days"])
            summary_data.append(['Predicted Avg Demand', f"{forecast_data['forecasted_demand'].mean():.2f}"])
        
        if model_name:
            summary_data.append(['Model Used', model_name])
        
        if sentiment_data is not None:
            summary_data.append(['Reviews Analyzed', str(len(sentiment_data))])
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f0f9ff')])
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Model Performance Metrics
        if metrics:
            story.append(Paragraph("Model Performance Metrics", heading_style))
            
            metrics_data = [['Metric', 'Value', 'Interpretation']]
            
            interpretations = {
                'MAE': 'Lower is better - Average prediction error',
                'RMSE': 'Lower is better - Root mean squared error',
                'R2': 'Higher is better (max 1.0) - Model fit quality',
                'MAPE': 'Lower is better - Percentage error'
            }
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                    interpretation = interpretations.get(key, 'N/A')
                    metrics_data.append([key, formatted_value, interpretation])
            
            metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f9ff')])
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Generate graphs
        story.append(PageBreak())
        story.append(Paragraph("Visualizations & Analysis", heading_style))
        
        # Graph 1: Historical Demand Trend
        if demand_data is not None:
            story.append(Paragraph("Historical Demand Trend", subheading_style))
            
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(demand_data['date'], demand_data['demand'], color='#1f77b4', linewidth=2)
            ax1.fill_between(demand_data['date'], demand_data['demand'], alpha=0.3, color='#1f77b4')
            ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Demand', fontsize=12, fontweight='bold')
            ax1.set_title('Historical Demand Trend', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img_buffer1 = io.BytesIO()
            plt.savefig(img_buffer1, format='png', dpi=300, bbox_inches='tight')
            img_buffer1.seek(0)
            plt.close()
            
            img1 = Image(img_buffer1, width=6*inch, height=3*inch)
            story.append(img1)
            story.append(Spacer(1, 0.2*inch))
        
        # Graph 2: Forecast with Confidence Interval
        if demand_data is not None and forecast_data is not None:
            story.append(Paragraph("Demand Forecast", subheading_style))
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            
            ax2.plot(demand_data['date'], demand_data['demand'], 
                    color='#1f77b4', linewidth=2, label='Historical')
            
            ax2.plot(forecast_data['date'], forecast_data['forecasted_demand'], 
                    color='#ff7f0e', linewidth=2, linestyle='--', label='Forecast')
            
            upper = forecast_data['forecasted_demand'] * 1.1
            lower = forecast_data['forecasted_demand'] * 0.9
            ax2.fill_between(forecast_data['date'], lower, upper, 
                            alpha=0.2, color='#ff7f0e', label='Confidence Interval')
            
            ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Demand', fontsize=12, fontweight='bold')
            ax2.set_title('Demand Forecast with 90% Confidence Interval', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img_buffer2 = io.BytesIO()
            plt.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight')
            img_buffer2.seek(0)
            plt.close()
            
            img2 = Image(img_buffer2, width=6*inch, height=3*inch)
            story.append(img2)
            story.append(Spacer(1, 0.2*inch))
        
        # Graph 3: Sentiment Analysis
        if sentiment_data is not None and 'sentiment_label' in sentiment_data.columns:
            story.append(PageBreak())
            story.append(Paragraph("Sentiment Analysis", subheading_style))
            
            sentiment_counts = sentiment_data['sentiment_label'].value_counts()
            
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            colors_pie = ['#10b981', '#ef4444', '#6b7280']
            ax3.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=colors_pie, startangle=90)
            ax3.set_title('Customer Sentiment Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            img_buffer3 = io.BytesIO()
            plt.savefig(img_buffer3, format='png', dpi=300, bbox_inches='tight')
            img_buffer3.seek(0)
            plt.close()
            
            img3 = Image(img_buffer3, width=5*inch, height=4*inch)
            story.append(img3)
            story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        if recommendations:
            story.append(PageBreak())
            story.append(Paragraph("Strategic Recommendations", heading_style))
            
            rec_data = [['Priority', 'Category', 'Recommendation']]
            for rec in recommendations:
                rec_data.append([rec['Priority'], rec['Category'], rec['Recommendation']])
            
            rec_table = Table(rec_data, colWidths=[1*inch, 1.5*inch, 3.5*inch])
            rec_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f9ff')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(rec_table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


# ============================================================================
# INSIGHTS PAGE
# ============================================================================

def show_insights_page():
    """Insights page with comprehensive PDF report"""
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first!")
        return
    
    st.success("✅ Model trained successfully!")
    
    st.markdown("### 🎯 Key Insights")
    
    # Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Demand Trends")
        if st.session_state.demand_data is not None:
            recent_demand = st.session_state.demand_data['demand'].tail(30).mean()
            forecast_demand = np.mean(st.session_state.forecast_values)
            change = ((forecast_demand - recent_demand) / recent_demand) * 100
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Recent Avg", f"{recent_demand:.0f}")
            with col_b:
                st.metric("Forecast Avg", f"{forecast_demand:.0f}", f"{change:+.1f}%")
    
    with col2:
        st.markdown("#### 😊 Sentiment Analysis")
        if st.session_state.processed_data is not None and 'sentiment_label' in st.session_state.processed_data.columns:
            sentiment_dist = st.session_state.processed_data['sentiment_label'].value_counts()
            positive_pct = (sentiment_dist.get('positive', 0) / len(st.session_state.processed_data)) * 100
            
            st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            if positive_pct > 60:
                st.success("🟢 High positive sentiment detected")
            elif positive_pct < 40:
                st.warning("🟡 Low positive sentiment - consider improvements")
            else:
                st.info("🔵 Neutral sentiment balance")
    
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    recommendations = []
    
    if st.session_state.processed_data is not None and 'sentiment_label' in st.session_state.processed_data.columns:
        negative_pct = (st.session_state.processed_data['sentiment_label'].value_counts().get('negative', 0) / 
                       len(st.session_state.processed_data)) * 100
        
        if negative_pct > 30:
            recommendations.append({
                'Priority': 'High',
                'Category': 'Customer Satisfaction',
                'Recommendation': 'Address customer complaints - high negative sentiment detected'
            })
    
    if 'forecast_values' in st.session_state and st.session_state.forecast_values is not None:
        forecast_trend = np.polyfit(range(len(st.session_state.forecast_values)), 
                                    st.session_state.forecast_values, 1)[0]
        
        if forecast_trend > 0:
            recommendations.append({
                'Priority': 'Medium',
                'Category': 'Inventory',
                'Recommendation': 'Increase inventory levels - demand trending upward'
            })
        else:
            recommendations.append({
                'Priority': 'Low',
                'Category': 'Inventory',
                'Recommendation': 'Optimize inventory - demand stable or declining'
            })
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        
        def highlight_priority(row):
            if row['Priority'] == 'High':
                return ['background-color: #dc2626; color: white'] * len(row)
            elif row['Priority'] == 'Medium':
                return ['background-color: #f59e0b; color: white'] * len(row)
            else:
                return ['background-color: #6366f1; color: white'] * len(row)
        
        styled_df = rec_df.style.apply(highlight_priority, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    
    # PDF generation section
    st.markdown("### 📄 Generate Comprehensive PDF Report")
    st.info("📊 The PDF report includes all graphs, metrics, sentiment analysis, and recommendations")
    
    if PDF_AVAILABLE:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Report Contents:**
            - Executive Summary
            - Model Performance Metrics
            - Historical Demand Trend (Graph)
            - Demand Forecast with Confidence Interval (Graph)
            - Sentiment Analysis Distribution (Graph)
            - Strategic Recommendations
            - Detailed Forecast Data Table
            """)
        
        with col2:
            if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive PDF report with graphs..."):
                    try:
                        pdf_buffer = generate_comprehensive_pdf_report(
                            demand_data=st.session_state.demand_data,
                            forecast_data=st.session_state.forecast_data,
                            metrics=st.session_state.metrics,
                            sentiment_data=st.session_state.processed_data,
                            recommendations=recommendations,
                            model_name=st.session_state.model_name
                        )
                        
                        if pdf_buffer:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"demand_forecast_report_{timestamp}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success("✅ PDF report generated successfully with all graphs and data!")
                    
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
    else:
        st.error("📦 PDF generation requires additional libraries. Please install: `pip install reportlab matplotlib`")
