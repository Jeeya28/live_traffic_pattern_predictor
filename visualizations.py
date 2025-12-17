"""
Enhanced Visualizations Module
All chart creation functions for the dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_gauge_chart(congestion_value):
    """Create animated gauge chart for current congestion"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=congestion_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Congestion Level", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#10b981'},
                {'range': [25, 50], 'color': '#fbbf24'},
                {'range': [50, 75], 'color': '#f97316'},
                {'range': [75, 100], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"},
        height=300
    )
    
    return fig

def create_heatmap_calendar(hourly_predictions):
    """Create weekly traffic heatmap"""
    # Simulate weekly data
    weekly_data = []
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for day in range(7):
        if day < 5:  # Weekday
            daily_data = [x * (1.0 + np.random.uniform(-0.1, 0.1)) for x in hourly_predictions]
        else:  # Weekend
            daily_data = [x * 0.7 * (1.0 + np.random.uniform(-0.1, 0.1)) for x in hourly_predictions]
        weekly_data.append(daily_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=weekly_data,
        x=[f"{h}:00" for h in range(24)],
        y=days,
        colorscale='RdYlGn_r',
        text=np.array(weekly_data).round(0),
        texttemplate='%{text:.0f}',
        textfont={"size": 8},
        colorbar=dict(
            title=dict(text="Congestion %", side="right")
        )
    ))
    
    fig.update_layout(
        title="Weekly Traffic Pattern Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=350
    )
    
    return fig
    
    fig.update_layout(
        title="Weekly Traffic Pattern Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=350
    )
    
    return fig

def create_comparison_chart(rf_r2, lgb_r2, lr_r2, rf_rmse, lgb_rmse, lr_rmse):
    """Create model comparison bar chart"""
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'LightGBM', 'Linear Regression', 'Ensemble'],
        'R² Score': [rf_r2, lgb_r2, lr_r2, (rf_r2 + lgb_r2 + lr_r2) / 3],
        'RMSE': [rf_rmse, lgb_rmse, lr_rmse, (rf_rmse + lgb_rmse + lr_rmse) / 3]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='R² Score',
        x=comparison_df['Model'],
        y=comparison_df['R² Score'],
        marker_color='#3b82f6',
        text=comparison_df['R² Score'].round(4),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='RMSE',
        x=comparison_df['Model'],
        y=comparison_df['RMSE'],
        marker_color='#ef4444',
        text=comparison_df['RMSE'].round(2),
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig
def create_speed_chart(hourly_predictions, current_hour):
    """Create speed vs congestion chart"""
    # Calculate corresponding speeds (inverse relationship)
    speeds = [60 - (p * 0.5) for p in hourly_predictions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=speeds,
        mode='lines+markers',
        name='Average Speed',
        line=dict(color='#10b981', width=3),
        marker=dict(size=6),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=hourly_predictions,
        mode='lines+markers',
        name='Congestion',
        line=dict(color='#ef4444', width=3, dash='dash'),
        marker=dict(size=6),
        yaxis='y2'
    ))
    
    fig.add_vline(x=current_hour, line_dash="dot", line_color="purple", 
                  annotation_text=f"Now", annotation_position="top")
    
    fig.update_layout(
        title="Speed vs Congestion Analysis",
        xaxis_title="Hour of Day",
        yaxis=dict(title="Speed (km/h)", side="left", range=[0, 70]),
        yaxis2=dict(title="Congestion (%)", side="right", overlaying="y", range=[0, 100]),
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_route_comparison(source, destination, current_congestion):
    """Create alternative routes comparison chart"""
    routes_data = [
        {
            'name': f'{source} -> {destination} (Main)',
            'time': 35 * (1.6 if current_congestion >= 75 else 1.3 if current_congestion >= 50 else 1.0),
            'distance': 18,
            'congestion': current_congestion,
            'cost': 150
        },
        {
            'name': f'{source} -> {destination} (Alt 1)',
            'time': 35 * 1.2,
            'distance': 22,
            'congestion': current_congestion * 0.7,
            'cost': 180
        },
        {
            'name': f'{source} -> {destination} (Alt 2)',
            'time': 35 * 1.1,
            'distance': 20,
            'congestion': current_congestion * 0.8,
            'cost': 165
        }
    ]
    
    df = pd.DataFrame(routes_data)
    
    fig = go.Figure()
    
    metrics = ['time', 'distance', 'congestion']
    colors = ['#3b82f6', '#10b981', '#f97316']
    titles = ['Time (min)', 'Distance (km)', 'Congestion (%)']
    
    for i, (metric, color, title) in enumerate(zip(metrics, colors, titles)):
        fig.add_trace(go.Bar(
            name=title,
            x=df['name'],
            y=df[metric],
            marker_color=color,
            text=df[metric].round(1),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Alternative Routes Comparison",
        xaxis_title="Route",
        yaxis_title="Value",
        barmode='group',
        height=400,
        showlegend=True
    )
    
    return fig