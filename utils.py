"""
Utility Functions
Helper functions for data export and calculations
"""

import pandas as pd
from datetime import datetime
import streamlit as st

def export_prediction_data(source, destination, prediction, features, model_used):
    """Export prediction data as CSV"""
    export_data = {
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Source': [source],
        'Destination': [destination],
        'Route': [f"{source} -> {destination}"],
        'Predicted_Congestion (%)': [f"{prediction:.1f}"],
        'Traffic_Volume': [f"{features['traffic_volume']:,.0f}"],
        'Average_Speed (km/h)': [f"{features['avg_speed']:.1f}"],
        'Travel_Time_Index': [f"{features['travel_time_index']:.2f}"],
        'Capacity_Utilization (%)': [f"{features['capacity_util']:.1f}"],
        'Incidents_Reported': [features['incidents']],
        'Environmental_Impact': [f"{features['env_impact']:.1f}"],
        'Public_Transport_Usage (%)': [f"{features['public_transport']:.1f}"],
        'Signal_Compliance (%)': [f"{features['signal_compliance']:.1f}"],
        'Parking_Usage (%)': [f"{features['parking_usage']:.1f}"],
        'Pedestrian_Count': [f"{features['pedestrian_count']:.0f}"],
        'Model_Used': [model_used],
        'Status': ['High' if prediction >= 75 else 'Medium' if prediction >= 50 else 'Low']
    }
    
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8')

def calculate_route_stats(congestion, base_time, features):
    """Calculate comprehensive route statistics"""
    # Calculate travel times
    normal_time = base_time
    
    if congestion >= 75:
        multiplier = 1.6
    elif congestion >= 50:
        multiplier = 1.3
    else:
        multiplier = 1.0
    
    traffic_time = normal_time * multiplier
    extra_time = traffic_time - normal_time
    
    # Calculate distance (approximate)
    distance = 18  # km (default)
    
    # Calculate costs
    fuel_cost_per_km = 8.5  # INR
    toll_cost = 50  # INR
    time_value_per_min = 5  # INR (opportunity cost)
    
    total_fuel_cost = distance * fuel_cost_per_km
    time_cost = traffic_time * time_value_per_min
    total_cost = total_fuel_cost + toll_cost + time_cost
    
    return {
        'distance': f"{distance} km",
        'normal_time': f"{normal_time:.0f} min",
        'traffic_time': f"{traffic_time:.0f} min",
        'extra_time': f"+{extra_time:.0f} min",
        'fuel_cost': f"Rs.{total_fuel_cost:.0f}",
        'total_cost': f"Rs.{total_cost:.0f}",
        'avg_speed': f"{features['avg_speed']:.1f} km/h"
    }

def format_time_difference(minutes):
    """Format time difference in human-readable format"""
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:.0f}h {mins:.0f}m"

def get_traffic_status_emoji(congestion):
    """Get emoji based on congestion level"""
    if congestion >= 90:
        return "🔴"
    elif congestion >= 75:
        return "🟠"
    elif congestion >= 50:
        return "🟡"
    elif congestion >= 25:
        return "🟢"
    else:
        return "✅"

def get_color_for_congestion(congestion):
    """Get color code based on congestion level"""
    if congestion >= 75:
        return "#ef4444"  # Red
    elif congestion >= 50:
        return "#f97316"  # Orange
    elif congestion >= 25:
        return "#fbbf24"  # Yellow
    else:
        return "#10b981"  # Green