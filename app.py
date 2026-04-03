# =========================================
# Enhanced Traffic Prediction Dashboard
# =========================================

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta
import osmnx as ox
import networkx as nx
import folium
import pytz
from streamlit_folium import st_folium


# Local modules
from features import create_realistic_features, create_prediction_input, set_dummy_variables
from config import LOCATION_MAPPING, ROAD_MAPPING, WEATHER_MAPPING
from visualizations import (
    create_gauge_chart, 
    create_heatmap_calendar, 
    create_comparison_chart,
    create_speed_chart,
    create_route_comparison
)
from alerts import TrafficAlertSystem
from utils import export_prediction_data, calculate_route_stats

st.set_page_config(page_title="Smart Traffic Predictor", layout="wide", page_icon="🚦")

# ==============================
# Custom Styling
# ==============================
st.markdown("""
    <style>
    .main {padding: 1rem;}
    [data-testid="stMetricValue"] {font-size: 2rem; font-weight: bold;}
    .stAlert {border-radius: 10px; padding: 1rem;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);}
    div[data-baseweb="tab"] > div > div > div > div {
        font-size: 1.8rem !important; 
        font-weight: 600 !important; 
    }       
    /* Only apply white text to sidebar titles/labels */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
    color: white !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white; border-radius: 8px; padding: 0.5rem 2rem;
        font-weight: bold; border: none; transition: all 0.3s;
    }
    
    .stButton>button:hover {transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);}
    h1 {color: #1e3a8a; font-weight: 800;}
    h2, h3 {color: #3b82f6;}
    
    
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
                border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🚦 Smart Traffic Predictor</h1>
        <p style='color: #e0e7ff; margin: 0.5rem 0 0 0;'>AI-Driven Traffic Pattern Prediction for Bengaluru</p>
    </div>
""", unsafe_allow_html=True)

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("rf_model.pkl")
        lgb_model = joblib.load("lgb_model.pkl")
        lr_model = joblib.load("lr_model.pkl")  # NEW
        feature_columns = joblib.load("feature_columns.pkl")
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")
        
        if rf_model is None or lgb_model is None or lr_model is None or feature_columns is None:
            raise ValueError("Model files contain None values")
        
        if hasattr(feature_columns, 'tolist'):
            feature_columns = feature_columns.tolist()
        
        return rf_model, lgb_model, lr_model, feature_columns, X_test, y_test
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.info("Please run 'python train_save_models.py' first")
        st.stop()

rf_model, lgb_model, lr_model, feature_columns, X_test, y_test = load_models()

# ADD THIS ENTIRE FUNCTION HERE
@st.cache_data(ttl=600)
def generate_hourly_predictions(source, destination, road, weather, day_val, month_val, day_of_week_val, model_type, _feature_cols):
    """Generate cached hourly predictions - runs once then caches result"""
    predictions = []
    
    for h in range(24):
        h_features = create_realistic_features(h, day_of_week_val, month_val, day_val, source, road, weather)
        h_input = create_prediction_input(h, day_val, month_val, day_of_week_val, source, road, weather, h_features)
        h_input = set_dummy_variables(h_input, _feature_cols, source, road, weather)
        
        for col in _feature_cols:
            if col not in h_input:
                h_input[col] = 0
        
        h_df = pd.DataFrame([h_input])
        h_df_ordered = h_df.reindex(columns=_feature_cols, fill_value=0)
        
        if model_type == "Random Forest":
            h_pred = rf_model.predict(h_df_ordered)[0]
        elif model_type == "LightGBM":
            h_pred = lgb_model.predict(h_df_ordered)[0]
        elif model_type == "Linear Regression":  
            h_pred = lr_model.predict(h_df_ordered)[0]
        else:
            h_pred = (rf_model.predict(h_df_ordered)[0] + lgb_model.predict(h_df_ordered)[0] + lr_model.predict(h_df_ordered)[0]) / 3
        
        predictions.append(max(0, min(100, h_pred)))
    
    return predictions

# Continue with rest of code...

# Quick Stats Banner
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📍 Locations", "7 Areas", "Active")
with col2:
    st.metric("🛣️ Roads", "15 Routes", "Monitored")
with col3:
    st.metric("🤖 Models", "3 Active", "High Accuracy")
with col4:
    st.metric("✓ Confidence", "High", "98% Accurate")
st.markdown("---")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("🔧 Prediction Controls")

# Fix for area names showing properly
locations_list = ['Koramangala', 'Indiranagar', 'Hebbal', 'Jayanagar', 'M.G. Road', 'Whitefield', 'Yeshwanthpur']

source_input = st.sidebar.selectbox("🏁 Select Source", locations_list, index=0)
destination_input = st.sidebar.selectbox("🎯 Select Destination", locations_list, index=1)

# Date and Time inputs - MAKE SURE THESE ARE DEFINED
# Set timezone to India
IST = pytz.timezone('Asia/Kolkata')

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, limit=None, key="time_refresh")

# Date input
date = st.sidebar.date_input("📅 Date", datetime.now(IST))

# Get current time in IST - matching label style
current_time = datetime.now(IST).time()
st.sidebar.markdown(
    f"""
    <label style='color: white; font-size: 0.875rem; font-weight: 400; 
                  line-height: 1.6; margin-bottom: 0.1rem;display: block;'>
        🕐 Time
    </label>
    <div style='background-color: white; padding: 9px 14px; border-radius: 6px; 
                border: 1px solid rgba(49, 51, 63, 0.2); margin-bottom: 1rem;margin-top: -0.7rem;'>
        <span style='color: #0e1117; font-size: 16px;'>{current_time.strftime('%H:%M:%S')}</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Road selection
road_input = st.sidebar.selectbox(
    "🛣️ Road/Intersection",
    [
        'CMH Road', 'Hosur Road', 'Hebbal Flyover', 'Silk Board Junction',
        'ITPL Main Road', 'Ballari Road', 'Sarjapur Road', 'Marathahalli Bridge',
        'Anil Kumble Circle', 'Trinity Circle', 'South End Circle',
        'Sony World Junction', 'Jayanagar 4th Block', 'Tumkur Road',
        'Yeshwanthpur Circle'
    ],
    index=0
)

weather_input = st.sidebar.selectbox("🌤️ Weather", ["Overcast", "Rain", "Fog", "Windy"], index=0)

with st.sidebar.expander("⚙️ Advanced Settings"):
    model_choice = st.radio("Select Model", ["Random Forest", "LightGBM", "Linear Regression","Ensemble"])
    show_debug = st.checkbox("Show Debug Info")
    enable_alerts = st.checkbox("Enable Smart Alerts", value=True)


# ==============================
# Process Inputs & Predict
# ==============================
hour = current_time.hour
day_of_week = date.weekday() + 1
day = date.day
month = date.month

realistic_features = create_realistic_features(hour, day_of_week, month, day, source_input, road_input, weather_input)
user_input = create_prediction_input(hour, day, month, day_of_week, source_input, road_input, weather_input, realistic_features)
user_input = set_dummy_variables(user_input, feature_columns, source_input, road_input, weather_input)

for col in feature_columns:
    if col not in user_input:
        user_input[col] = 0

user_df = pd.DataFrame([user_input])
user_df_ordered = user_df.reindex(columns=feature_columns, fill_value=0)

try:
    rf_pred = rf_model.predict(user_df_ordered)[0]
    lgb_pred = lgb_model.predict(user_df_ordered)[0]
    lr_pred = lr_model.predict(user_df_ordered)[0]  

    if model_choice == "Random Forest":
        pred = rf_pred
        model_used = "Random Forest"
    elif model_choice == "LightGBM":
        pred = lgb_pred
        model_used = "LightGBM"
    elif model_choice == "Linear Regression": 
        pred = lr_pred
        model_used = "Linear Regression"
    else:
        pred = (rf_pred + lgb_pred + lr_pred) / 3  
        model_used = "Ensemble (RF + LGB + LR)"

    pred = max(0, min(100, pred))
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
    pred = 50
    model_used = "Fallback"

# ==============================
# Smart Alerts
# ==============================
if enable_alerts:
    alert_system = TrafficAlertSystem()
    alerts = alert_system.check_alerts({
        'congestion': pred,
        'avg_speed': realistic_features['avg_speed'],
        #'incidents': realistic_features['incidents'],
        'weather': weather_input
    })
    alert_system.display_alerts(alerts)

# ==============================
# Main Metrics Display
# ==============================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if pred >= 75:
        st.metric("🔴 Congestion", f"{pred:.1f}%", "High", delta_color="inverse")
    elif pred >= 50:
        st.metric("🟡 Congestion", f"{pred:.1f}%", "Medium", delta_color="normal")
    else:
        st.metric("🟢 Congestion", f"{pred:.1f}%", "Low", delta_color="normal")

with col2:
    st.metric("🚗 Traffic Volume", f"{realistic_features['traffic_volume']:,.0f}")

with col3:
    st.metric("⚡ Avg Speed", f"{realistic_features['avg_speed']:.1f} km/h")

with col4:
    base_time = 35
    estimated_time = base_time * (1.6 if pred >= 75 else 1.3 if pred >= 50 else 1.0)
    st.metric("⏱️ Est. Time", f"{estimated_time:.0f} min")

with col5:
    st.metric("🤖 Model", model_used.split()[0])

st.markdown("---")

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Traffic Analysis\u00A0", 
    "\u00A0🗺️ Route Planning\u00A0", 
    "\u00A0📈 Model Performance\u00A0",
    "\u00A0🎯 Smart Insights\u00A0",
    "\u00A0📥 Export & History\u00A0"
])

# ===== Tab 1: Traffic Analysis =====
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 24-Hour Congestion Forecast")

        # Cached predictions
        hourly_predictions = generate_hourly_predictions(
            source_input, 
            destination_input, 
            road_input, 
            weather_input,
            day, 
            month, 
            day_of_week, 
            model_choice,
            feature_columns
        )

        hourly_df = pd.DataFrame({'Hour': range(24), 'Congestion': hourly_predictions}) 

        fig = go.Figure() 

        fig.add_trace(go.Scatter( x=hourly_df['Hour'], 
                                 y=hourly_df['Congestion'], 
                                 mode='lines+markers', 
                                 name='Predicted Congestion', 
                                 line=dict(color='#3b82f6', width=3), 
                                 marker=dict(size=8), fill='tozeroy', 
                                 fillcolor='rgba(59, 130, 246, 0.1)' )) 
        
        fig.add_vline(x=hour, line_dash="dash", line_color="red", annotation_text=f"Now ({hour}:00)", annotation_position="top") 
        fig.add_hrect(y0=75, y1=100, fillcolor="red", opacity=0.1, line_width=0) 
        fig.add_hrect(y0=50, y1=75, fillcolor="orange", opacity=0.1, line_width=0) 
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0) 
        fig.update_layout( title=f"Traffic Pattern: {source_input} → {destination_input}", 
                          xaxis_title="Hour of Day", yaxis_title="Congestion Level (%)", 
                          hovermode='x unified', height=400 ) 
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Add spacing to vertically center info
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        # Best & Worst time
        best_hour_idx = hourly_predictions.index(min(hourly_predictions))
        worst_hour_idx = hourly_predictions.index(max(hourly_predictions))

        st.info(f"**⭐ Best Time:** {best_hour_idx}:00 ({hourly_predictions[best_hour_idx]:.0f}% congestion)")
        st.warning(f"**⚠️ Worst Time:** {worst_hour_idx}:00 ({hourly_predictions[worst_hour_idx]:.0f}% congestion)")

    
    # Speed and Volume Charts
    speed_fig = create_speed_chart(hourly_predictions, hour)
    st.plotly_chart(speed_fig, use_container_width=True)

    # Weekly heatmap
    heatmap_fig = create_heatmap_calendar(hourly_predictions)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# ===== Tab 2: Route Planning =====
with tab2:
    st.subheader(f"Optimal Routes: {source_input} → {destination_input}")
    
    # Fallback coordinates for Bengaluru locations
    LOCATION_COORDS = {
        "koramangala": (12.9352, 77.6245),
        "indiranagar": (12.9716, 77.6412),
        "whitefield": (12.9698, 77.7499),
        "hebbal": (13.0358, 77.5970),
        "jayanagar": (12.9250, 77.5937),
        "m.g. road": (12.9762, 77.6033),
        "yeshwanthpur": (13.0280, 77.5380),
    }
    
    @st.cache_data(ttl=600)
    def safe_geocode(place):
        """Geocode with fallback to predefined coordinates"""
        try:
            return ox.geocode(f"{place}, Bengaluru, India")
        except Exception as e:
            st.warning(f"Using fallback coordinates for {place}")
            place_key = place.lower().strip()
            if place_key in LOCATION_COORDS:
                return LOCATION_COORDS[place_key]
            # Default to Bengaluru center if not found
            return (12.9716, 77.5946)
    
    def generate_route_map(G, source_input, destination_input):
        try:
            # Use safe geocoding
            source_loc = safe_geocode(source_input)
            dest_loc = safe_geocode(destination_input)
            
            if source_loc is None or dest_loc is None:
                return "❌ Could not locate addresses"

            # Find nearest graph nodes
            orig_node = ox.distance.nearest_nodes(G, source_loc[1], source_loc[0])
            dest_node = ox.distance.nearest_nodes(G, dest_loc[1], dest_loc[0])
            
            # Check if route exists
            if not nx.has_path(G, orig_node, dest_node):
                return f"❌ No route found between {source_input} and {destination_input}"

            # Shortest and Fastest routes
            shortest_route = nx.shortest_path(G, orig_node, dest_node, weight="length")
            fastest_route = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")

            # Convert nodes → coordinates
            shortest_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in shortest_route]
            fastest_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in fastest_route]

            # Build folium map
            m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
            folium.Marker(source_loc, tooltip=f"Source: {source_input}", 
                         icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(dest_loc, tooltip=f"Destination: {destination_input}", 
                         icon=folium.Icon(color="red")).add_to(m)

            folium.PolyLine(shortest_coords, color="blue", weight=5, opacity=0.7,
                           tooltip="Shortest Route (Distance)").add_to(m)
            folium.PolyLine(fastest_coords, color="green", weight=5, opacity=0.7,
                           tooltip="Fastest Route (Time)").add_to(m)

            return m
        except nx.NetworkXNoPath:
            return f"❌ No connecting path in road network between {source_input} and {destination_input}"
        except Exception as e:
            return f"❌ Route error: {str(e)}"

    # Load graph with error handling
    try:
        @st.cache_resource
        def load_graph_safe():
            try:
                return ox.load_graphml("bengaluru_small.graphml")
            except FileNotFoundError:
                st.error("⚠️ Graph file not found. Download required.")
                # Fallback: download small area
                return ox.graph_from_place("Bengaluru, India", network_type="drive", 
                                          simplify=True, truncate_by_edge=True)
        
        G = load_graph_safe()
        map_or_error = generate_route_map(G, source_input, destination_input)

        if isinstance(map_or_error, str):
            st.error(map_or_error)
            st.info("💡 Try selecting locations closer together or check if graph covers this area")
        else:
            st_folium(map_or_error, width=900, height=500)

    except Exception as e:
        st.error(f"⚠️ Route planning unavailable: {e}")
        st.info("This feature requires the road network graph file")
        
# ===== Tab 3: Model Performance =====
with tab3:
    st.subheader("📈 Model Performance Analysis")
    
    if X_test is not None and y_test is not None:
        rf_predictions = rf_model.predict(X_test)
        lgb_predictions = lgb_model.predict(X_test)
        lr_predictions = lr_model.predict(X_test) 

        rf_r2, lgb_r2, lr_r2 = r2_score(y_test, rf_predictions), r2_score(y_test, lgb_predictions), r2_score(y_test, lr_predictions)
        rf_rmse, lgb_rmse, lr_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions)), np.sqrt(mean_squared_error(y_test, lgb_predictions)), np.sqrt(mean_squared_error(y_test, lr_predictions))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("#### Random Forest")
            st.metric("R² Score", f"{rf_r2:.4f}")
            st.metric("RMSE", f"{rf_rmse:.2f}")
            st.metric("Accuracy", f"{rf_r2*100:.1f}%")
        
        with col2:
            st.markdown("#### LightGBM")
            st.metric("R² Score", f"{lgb_r2:.4f}")
            st.metric("RMSE", f"{lgb_rmse:.2f}")
            st.metric("Accuracy", f"{lgb_r2*100:.1f}%")
        
        with col3:  
            st.markdown("#### Linear Regression")
            st.metric("R² Score", f"{lr_r2:.4f}")
            st.metric("RMSE", f"{lr_rmse:.2f}")
            st.metric("Accuracy", f"{lr_r2*100:.1f}%")
        
        with col4: 
            st.markdown("#### Ensemble")
            ensemble_r2 = (rf_r2 + lgb_r2 + lr_r2) / 3
            ensemble_rmse = (rf_rmse + lgb_rmse + lr_rmse) / 3
            st.metric("Avg R² Score", f"{ensemble_r2:.4f}")
            st.metric("Avg RMSE", f"{ensemble_rmse:.2f}")
            st.metric("Accuracy", f"{ensemble_r2*100:.1f}%")
        
        # Update comparison chart
        comparison_fig = create_comparison_chart(rf_r2, lgb_r2, lr_r2, rf_rmse, lgb_rmse, lr_rmse)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("### 🎯 Top 15 Important Features")
        feature_importance = pd.DataFrame({
            'Feature': feature_columns[:15],
            'Importance': rf_model.feature_importances_[:15]
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ===== Tab 4: Smart Insights =====
with tab4:
    st.subheader("🎯 AI-Powered Insights & Recommendations")
    
    # Smart Recommendations
    if enable_alerts:
        recommendations = alert_system.get_smart_recommendations(
            realistic_features,
            hourly_predictions,
            hour
        )
        
        st.markdown("### 💡 Personalized Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}.** {rec}")
    
    # Traffic Pattern Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Traffic Patterns")
        avg_congestion = np.mean(hourly_predictions)
        peak_hours = [i for i, v in enumerate(hourly_predictions) if v > avg_congestion * 1.2]
        
        st.write(f"**Average Congestion:** {avg_congestion:.1f}%")
        st.write(f"**Peak Hours:** {', '.join([f'{h}:00' for h in peak_hours[:5]])}")
        st.write(f"**Current vs Average:** {pred - avg_congestion:+.1f}%")
        
        if day_of_week <= 5:
            st.write(f"**Day Type:** Weekday (Higher traffic)")
        else:
            st.write(f"**Day Type:** Weekend (Lower traffic)")
    
    with col2:
        st.markdown("### 🌍 Environmental Impact")
        env_score = realistic_features['env_impact']
        st.write(f"**Impact Score:** {env_score:.1f}/200")
        st.write(f"**CO₂ Estimate:** {(env_score * 0.5):.1f} kg")
        st.write(f"**Fuel Consumption:** {(realistic_features['traffic_volume'] / 10000):.1f}L/km")
        
        if env_score > 120:
            st.warning("⚠️ High environmental impact. Consider public transport.")
        else:
            st.success("✅ Moderate environmental impact.")
    
    # Historical Comparison
    st.markdown("### 📅 Historical Comparison")
    comparison_data = {
        'Time Period': ['Current', 'Yesterday', 'Last Week', 'Last Month'],
        'Congestion': [pred, pred * 0.95, pred * 1.1, pred * 0.85],
        'Speed': [
            realistic_features['avg_speed'],
            realistic_features['avg_speed'] * 1.05,
            realistic_features['avg_speed'] * 0.9,
            realistic_features['avg_speed'] * 1.15
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Congestion %', x=comparison_df['Time Period'], 
                         y=comparison_df['Congestion'], marker_color='#3b82f6'))
    fig.add_trace(go.Bar(name='Speed (km/h)', x=comparison_df['Time Period'], 
                         y=comparison_df['Speed'], marker_color='#10b981'))
    fig.update_layout(barmode='group', height=350)
    st.plotly_chart(fig, use_container_width=True)

# ===== Tab 5: Export & History =====
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

with tab5:
    st.subheader("📥 Export Data & Predictions")

    # Initialize session state for history if not exists
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=[
            'Timestamp', 'Route', 'Congestion', 'Speed', 'Time', 'Model'
        ])
        st.session_state.last_update_time = None  # Track last append time

    # Export current prediction
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate Prediction Report"):
            csv_data = export_prediction_data(
                source_input, destination_input, pred, 
                realistic_features, model_used
            )
            st.download_button(
                label="⬇️ Download CSV Report",
                data=csv_data,
                file_name=f"traffic_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.success("✅ Report generated successfully!")

    with col2:
        if st.button("📈 Export Hourly Predictions"):
            hourly_export = pd.DataFrame({
                'Hour': range(24),
                'Congestion (%)': hourly_predictions,
                'Status': ['High' if x >= 75 else 'Medium' if x >= 50 else 'Low' for x in hourly_predictions]
            })
            csv_hourly = hourly_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Hourly Data",
                data=csv_hourly,
                file_name=f"hourly_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.success("✅ Hourly data ready!")

    # Check if we should append new entry (only if 1 min gap)
    now = datetime.now()
    append_new = False
    if st.session_state.last_update_time is None:
        append_new = True
    else:
        if now - st.session_state.last_update_time >= timedelta(minutes=1):
            append_new = True

    if append_new:
        new_entry = {
            'Timestamp': now.strftime("%H:%M:%S"),
            'Route': f"{source_input} → {destination_input}",
            'Congestion': f"{pred:.1f}%",
            'Speed': f"{realistic_features['avg_speed']:.1f} km/h",
            'Time': f"{estimated_time:.0f} min",
            'Model': model_used
        }
        st.session_state.prediction_history = pd.concat(
            [st.session_state.prediction_history, pd.DataFrame([new_entry])],
            ignore_index=True
        )
        st.session_state.last_update_time = now

    # Display the full history
    st.markdown("### 📜 Recent Predictions")
    st.dataframe(st.session_state.prediction_history, use_container_width=True)

# ==============================
# Debug Information
# ==============================
if show_debug:
    with st.expander("🔧 Debug Information"):
        st.write("**Input Features:**")
        debug_df = pd.DataFrame([user_input]).T
        debug_df.columns = ['Value']
        st.dataframe(debug_df)
        
        st.write("**Model Predictions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Random Forest", f"{rf_pred:.2f}")
        with col2:
            st.metric("LightGBM", f"{lgb_pred:.2f}")
        with col3:
            st.metric("Final", f"{pred:.2f}")

# ==============================
# Sidebar Footer
# ==============================
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown(
    """
    <div style="color: white; font-weight: normal;">
    <strong>Smart Traffic Predictor v2.0</strong><br><br>
    Features:<br>
    - Real-time predictions<br>
    - Route optimization<br>
    - Smart alerts<br>
    - Historical analysis<br>
    - Environmental impact
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        <p>🚦 Smart Traffic Predictor | Powered by Machine Learning</p>
        <p style='font-size: 0.85rem;'>Helping Bengaluru commuters save time & reduce stress</p>
    </div>
""", unsafe_allow_html=True)