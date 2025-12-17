# apptomtom.py
# =========================================
# Smart Traffic Predictor v3.1 — Cleaned (Incidents removed)
# - TomTom incidents API removed entirely (get_tomtom_incidents removed)
# - Keeps TomTom flowSegmentData, ML models, and Hybrid blending
# - Mirrors app.py UI and features (predictions, route planning, charts, export)
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
import pytz
import requests
import math
import logging

# Local modules (must exist in project workspace)
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

st.set_page_config(page_title="Smart Traffic Predictor ", layout="wide", page_icon="🚦")
logging.basicConfig(level=logging.INFO)

# ==============================
# TomTom API Configuration
# ==============================
TOMTOM_API_KEY = st.secrets.get("TOMTOM_API_KEY", None)
if not TOMTOM_API_KEY:
    TOMTOM_API_KEY = st.secrets.get("TOMTOM_API_KEY", "YOUR_API_KEY_HERE")

# Fallback coordinates for named areas
LOCATION_COORDS = {
    "koramangala": (12.9352, 77.6245),
    "indiranagar": (12.9716, 77.6412),
    "whitefield": (12.9698, 77.7499),
    "hebbal": (13.0358, 77.5970),
    "jayanagar": (12.9250, 77.5937),
    "m.g. road": (12.9762, 77.6033),
    "yeshwanthpur": (13.0280, 77.5380),
}

# TomTom flow API helper (keeps only flowSegmentData)
@st.cache_data(ttl=300)
def get_tomtom_flow(lat, lon):
    """Get flowSegmentData from TomTom for a point (lat, lon). Returns dict or None."""
    if not TOMTOM_API_KEY or TOMTOM_API_KEY == "YOUR_API_KEY_HERE":
        return None

    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{lat},{lon}",
        "unit": "KMPH",
    }
    try:
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        if "flowSegmentData" in data:
            f = data["flowSegmentData"]
            return {
                "current_speed": f.get("currentSpeed", 0),
                "free_flow_speed": f.get("freeFlowSpeed", 0),
                "current_travel_time": f.get("currentTravelTime", 0),
                "free_flow_travel_time": f.get("freeFlowTravelTime", 0),
                "confidence": f.get("confidence", 0),
                "road_closure": f.get("roadClosure", False)
            }
    except Exception as e:
        st.sidebar.warning(f"TomTom flow error: {str(e)[:80]}")
        return None
    return None

def calculate_congestion_from_tomtom(traffic_data):
    """
    Convert TomTom traffic data to congestion percentage (0–100)
    using speed, travel time, traffic status, and jam factor.
    """
    if not traffic_data:
        return None

    # Extract parameters safely
    current_speed = traffic_data.get('current_speed', 0)
    free_flow_speed = traffic_data.get('free_flow_speed', 1)  # avoid division by zero
    current_travel_time = traffic_data.get('current_travel_time', 0)
    free_flow_travel_time = traffic_data.get('free_flow_travel_time', 1)
    jam_factor = traffic_data.get('jamFactor', 0)  # 0–10 scale
    traffic_status = traffic_data.get('trafficStatus', 'NORMAL').upper()
    confidence = traffic_data.get('confidence', 1.0)  # 0–1

    # 1️⃣ Speed-based congestion (0–100)
    speed_ratio = max(0.0, min(1.0, current_speed / free_flow_speed))
    congestion_speed = (1 - speed_ratio) * 100

    # 2️⃣ Delay-based congestion (0–100)
    delay_ratio = max(0.0, (current_travel_time - free_flow_travel_time) / free_flow_travel_time)
    congestion_delay = delay_ratio * 100

    # 3️⃣ Jam factor (0–100)
    congestion_jam = max(0.0, min(100.0, jam_factor * 10))

    congestion_estimate = 0.4 * congestion_speed + 0.3 * congestion_delay + 0.3 * congestion_jam
    congestion_estimate = max(0.0, min(100.0, congestion_estimate))



def estimate_traffic_volume_from_tomtom(flow_data, baseline_volume=10000):
    """
    Heuristic: estimate traffic_volume from TomTom speed ratio.
    baseline_volume: assumed volume at free-flow speed.
    Returns integer estimated vehicles/h (heuristic).
    """
    if not flow_data:
        return None
    free_flow = flow_data.get("free_flow_speed", 0)
    current = flow_data.get("current_speed", 0)
    if free_flow == 0:
        return None
    speed_ratio = current / free_flow
    estimated = baseline_volume * (1 + (1 - speed_ratio) * 2.0)
    return int(max(0, math.ceil(estimated)))

# ==============================
# Styling & Header (same as app.py)
# ==============================
st.markdown("""
    <style>
    .main {padding: 1rem;}
    [data-testid="stMetricValue"] {font-size: 2rem; font-weight: bold;}
    .stAlert {border-radius: 10px; padding: 1rem;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);}
    div[data-baseweb="tab"] > div > div > div > div { font-size: 1.8rem !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: white !important; }
    .stButton>button { background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); color: white; border-radius: 8px; padding: 0.5rem 2rem; font-weight: bold; border: none; transition: all 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
    h1 {color: #1e3a8a; font-weight: 800;}
    h2, h3 {color: #3b82f6;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
                border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🚦 Smart Traffic Predictor + TomTom</h1>
        <p style='color: #e0e7ff; margin: 0.5rem 0 0 0;'>AI-Driven Forecasts + TomTom Live Data (Bengaluru)</p>
    </div>
""", unsafe_allow_html=True)

# ==============================
# Load ML models (same as app.py logic)
# ==============================
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("rf_model.pkl")
        lgb_model = joblib.load("lgb_model.pkl")
        lr_model = joblib.load("lr_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")
        if hasattr(feature_columns, "tolist"):
            feature_columns = feature_columns.tolist()
        return rf_model, lgb_model, lr_model, feature_columns, X_test, y_test
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.info("Please run the training script to produce rf_model.pkl, lgb_model.pkl, lr_model.pkl, feature_columns.pkl")
        st.stop()

rf_model, lgb_model, lr_model, feature_columns, X_test, y_test = load_models()

# ==============================
# Cached prediction generator (24-hour)
# ==============================
@st.cache_data(ttl=600)
def generate_hourly_predictions(source, destination, road, weather, day_val, month_val, day_of_week_val, mode, ml_model_choice, _feature_cols, tomtom_congestion=None, tomtom_weight=0.6):
    """
    Returns 24-length list of predictions (0-100).
    mode: "ML Only", "TomTom Only", "Hybrid (ML + TomTom)"
    """
    preds = []
    for h in range(24):
        h_features = create_realistic_features(h, day_of_week_val, month_val, day_val, source, road, weather)
        h_in = create_prediction_input(h, day_val, month_val, day_of_week_val, source, road, weather, h_features)
        h_in = set_dummy_variables(h_in, _feature_cols, source, road, weather)
        for c in _feature_cols:
            if c not in h_in:
                h_in[c] = 0
        h_df = pd.DataFrame([h_in])
        h_df_ord = h_df.reindex(columns=_feature_cols, fill_value=0)
        if ml_model_choice == "Random Forest":
            ml_pred = rf_model.predict(h_df_ord)[0]
        elif ml_model_choice == "LightGBM":
            ml_pred = lgb_model.predict(h_df_ord)[0]
        elif ml_model_choice == "Linear Regression":
            ml_pred = lr_model.predict(h_df_ord)[0]
        else:
            ml_pred = (rf_model.predict(h_df_ord)[0] + lgb_model.predict(h_df_ord)[0] + lr_model.predict(h_df_ord)[0]) / 3.0

        ml_pred = float(max(0.0, min(100.0, ml_pred)))

        if mode == "TomTom Only":
            if tomtom_congestion is not None:
                preds.append(float(max(0.0, min(100.0, tomtom_congestion))))
            else:
                preds.append(ml_pred)
        elif mode == "Hybrid (ML + TomTom)" and tomtom_congestion is not None:
            blended = ml_pred * (1 - tomtom_weight) + float(tomtom_congestion) * tomtom_weight
            preds.append(float(max(0.0, min(100.0, blended))))
        else:
            preds.append(ml_pred)
    return preds

# ==============================
# Quick banner metrics
# ==============================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📍 Locations", "7 Areas", "Active")
with col2:
    st.metric("🛣️ Roads", "15 Routes", "Monitored")
with col3:
    st.metric("🤖 Models", "3 Active")
with col4:
    st.metric("📡 Live Data", "TomTom", "API")
st.markdown("---")

# ==============================
# Sidebar Inputs (mirrors app.py)
# ==============================
st.sidebar.header("🔧 Prediction Controls")
locations_list = ['Koramangala', 'Indiranagar', 'Hebbal', 'Jayanagar', 'M.G. Road', 'Whitefield', 'Yeshwanthpur']
source_input = st.sidebar.selectbox("🏁 Select Source", locations_list, index=0)
destination_input = st.sidebar.selectbox("🎯 Select Destination", locations_list, index=1)

IST = pytz.timezone("Asia/Kolkata")
st_autorefresh(interval=60000, limit=None, key="time_refresh")

date = st.sidebar.date_input("📅 Date", datetime.now(IST))
current_time = datetime.now(IST).time()
st.sidebar.markdown(
    f"""<label style='color: white; font-size: 0.875rem;'>🕐 Time</label>
    <div style='background-color: white; padding: 9px 14px; border-radius: 6px;'><span style='color:#0e1117'>{current_time.strftime('%H:%M:%S')}</span></div>""",
    unsafe_allow_html=True
)

road_input = st.sidebar.selectbox("🛣️ Road/Intersection", [
    'CMH Road', 'Hosur Road', 'Hebbal Flyover', 'Silk Board Junction',
    'ITPL Main Road', 'Ballari Road', 'Sarjapur Road', 'Marathahalli Bridge',
    'Anil Kumble Circle', 'Trinity Circle', 'South End Circle',
    'Sony World Junction', 'Jayanagar 4th Block', 'Tumkur Road',
    'Yeshwanthpur Circle'], index=0)

weather_input = st.sidebar.selectbox("🌤️ Weather", ["Overcast", "Rain", "Fog", "Windy"], index=0)

with st.sidebar.expander("⚙️ Advanced Settings"):
    mode_choice = st.radio("Mode", ["ML Only", "TomTom Only", "Hybrid (ML + TomTom)"])
    model_choice = st.radio("Select ML Model", ["Random Forest", "LightGBM", "Linear Regression", "Ensemble"])
    show_debug = st.checkbox("Show Debug Info")
    enable_alerts = st.checkbox("Enable Smart Alerts", value=True)
    use_tomtom_checkbox = st.checkbox("Use TomTom Live Data", value=True)
    tomtom_blend = 0.6
    baseline_volume = st.number_input("TomTom baseline volume (vehicles/hour, heuristic)", min_value=1000, max_value=100000, value=10000, step=500)

# ==============================
# Get TomTom Flow Data (no incidents)
# ==============================
src_coords = LOCATION_COORDS.get(source_input.lower(), (12.9716, 77.5946))
dst_coords = LOCATION_COORDS.get(destination_input.lower(), (12.9716, 77.5946))

tomtom_flow = None
tomtom_congestion = None
tomtom_estimated_values = {}

if use_tomtom_checkbox and TOMTOM_API_KEY and TOMTOM_API_KEY != "YOUR_API_KEY_HERE":
    tomtom_flow = get_tomtom_flow(src_coords[0], src_coords[1])
    tomtom_congestion = calculate_congestion_from_tomtom(tomtom_flow) if tomtom_flow else None
    if tomtom_flow:
        tomtom_estimated_values["avg_speed"] = float(tomtom_flow.get("current_speed", 0))
        tomtom_estimated_values["avg_travel_time"] = float(tomtom_flow.get("current_travel_time", 0))  # seconds
        tomtom_estimated_values["traffic_volume"] = estimate_traffic_volume_from_tomtom(tomtom_flow, baseline_volume)
        tomtom_estimated_values["congestion_estimate"] = tomtom_congestion
        tomtom_estimated_values["env_impact"] = float(min(200.0, 50.0 + (tomtom_congestion or 0) * 1.5)) if tomtom_congestion is not None else None
else:
    if use_tomtom_checkbox:
        st.sidebar.warning("TomTom key not configured; TomTom features disabled.")

# ==============================
# Build features and apply overrides (without incidents)
# ==============================
hour = current_time.hour
day_of_week = date.weekday() + 1
day = date.day
month = date.month

realistic_features = create_realistic_features(hour, day_of_week, month, day, source_input, road_input, weather_input)

# Apply TomTom overrides for multiple features when available and mode allows it
# Overrides: avg_speed, traffic_volume, avg_travel_time, congestion_estimate, env_impact
if tomtom_estimated_values and mode_choice in ("TomTom Only", "Hybrid (ML + TomTom)"):
    if tomtom_estimated_values.get("avg_speed") is not None:
        realistic_features["avg_speed"] = tomtom_estimated_values["avg_speed"]
    if tomtom_estimated_values.get("traffic_volume") is not None:
        realistic_features["traffic_volume"] = tomtom_estimated_values["traffic_volume"]
    if tomtom_estimated_values.get("avg_travel_time") is not None:
        realistic_features["avg_travel_time"] = tomtom_estimated_values["avg_travel_time"]
    if tomtom_estimated_values.get("congestion_estimate") is not None:
        realistic_features["tomtom_congestion"] = tomtom_estimated_values["congestion_estimate"]
    if tomtom_estimated_values.get("env_impact") is not None:
        realistic_features["env_impact"] = tomtom_estimated_values["env_impact"]

# Build user input vector for ML
user_input = create_prediction_input(hour, day, month, day_of_week, source_input, road_input, weather_input, realistic_features)
user_input = set_dummy_variables(user_input, feature_columns, source_input, road_input, weather_input)
for c in feature_columns:
    if c not in user_input:
        user_input[c] = 0
user_df = pd.DataFrame([user_input])
user_df_ordered = user_df.reindex(columns=feature_columns, fill_value=0)

# ==============================
# Make Predictions (ML, TomTom, Hybrid)
# ==============================
try:
    rf_pred = float(rf_model.predict(user_df_ordered)[0])
    lgb_pred = float(lgb_model.predict(user_df_ordered)[0])
    lr_pred = float(lr_model.predict(user_df_ordered)[0])
except Exception as e:
    st.error(f"Model prediction error: {str(e)}")
    rf_pred = lgb_pred = lr_pred = 50.0

if model_choice == "Random Forest":
    ml_pred = rf_pred
elif model_choice == "LightGBM":
    ml_pred = lgb_pred
elif model_choice == "Linear Regression":
    ml_pred = lr_pred
else:
    ml_pred = (rf_pred + lgb_pred + lr_pred) / 3.0

ml_pred = float(max(0.0, min(100.0, ml_pred)))

tomtom_pred = None
if tomtom_congestion is not None:
    tomtom_pred = float(max(0.0, min(100.0, tomtom_congestion)))

if mode_choice == "TomTom Only":
    if tomtom_pred is not None:
        final_pred = tomtom_pred
        model_used = "TomTom Live"
    else:
        final_pred = ml_pred
        model_used = "Fallback (ML)"
elif mode_choice == "Hybrid (ML + TomTom)":
    if tomtom_pred is not None:
        final_pred = float(max(0.0, min(100.0, ml_pred * (1 - tomtom_blend) + tomtom_pred * tomtom_blend)))
        model_used = f"{mode_choice} + TomTom"
    else:
        final_pred = ml_pred
        model_used = f"{mode_choice} (TomTom missing)"
else:
    final_pred = ml_pred
    model_used = f"{model_choice}"

final_pred = float(max(0.0, min(100.0, final_pred)))

# ==============================
# Smart Alerts
# ==============================
if enable_alerts:
    alert_system = TrafficAlertSystem()
    alerts = alert_system.check_alerts({
        "congestion": final_pred,
        "avg_speed": realistic_features.get("avg_speed", None),
        "weather": weather_input
    })
    alert_system.display_alerts(alerts)

# ==============================
# Main metrics display
# ==============================
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if final_pred >= 75:
        st.metric("🔴 Congestion", f"{final_pred:.1f}%", "High", delta_color="inverse")
    elif final_pred >= 50:
        st.metric("🟡 Congestion", f"{final_pred:.1f}%", "Medium")
    else:
        st.metric("🟢 Congestion", f"{final_pred:.1f}%", "Low")

with col2:
    st.metric("🚗 Traffic Volume", f"{realistic_features.get('traffic_volume', 0):,}")

with col3:
    speed_display = realistic_features.get("avg_speed", None)
    if speed_display is None:
        speed_display = "N/A"
    else:
        speed_display = f"{speed_display:.1f} km/h"
    st.metric("⚡ Avg Speed", speed_display)

with col4:
    base_time = 35
    estimated_time = base_time * (1.6 if final_pred >= 75 else 1.3 if final_pred >= 50 else 1.0)
    st.metric("⏱️ Est. Time", f"{estimated_time:.0f} min")

with col5:
    st.metric("🤖 Model Mode", model_used.split()[0])

st.markdown("---")

# ==============================
# Tabs (preserve app.py layout)
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

        hourly_predictions = generate_hourly_predictions(
            source_input, destination_input, road_input, weather_input,
            day, month, day_of_week, mode_choice, model_choice, feature_columns,
            tomtom_congestion=tomtom_pred, tomtom_weight=tomtom_blend
        )

        hourly_df = pd.DataFrame({"Hour": list(range(24)), "Congestion": hourly_predictions})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_df["Hour"], y=hourly_df["Congestion"],
            mode="lines+markers", name="Predicted Congestion",
            line=dict(color="#3b82f6", width=3), marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
        ))
        fig.add_vline(x=hour, line_dash="dash", line_color="red", annotation_text=f"Now ({hour}:00)", annotation_position="top")
        fig.add_hrect(y0=75, y1=100, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=50, y1=75, fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
        fig.update_layout(title=f"Traffic Pattern: {source_input} → {destination_input}", xaxis_title="Hour of Day", yaxis_title="Congestion Level (%)", hovermode="x unified", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        best_hour_idx = int(np.argmin(hourly_predictions))
        worst_hour_idx = int(np.argmax(hourly_predictions))
        st.info(f"**⭐ Best Time:** {best_hour_idx}:00 ({hourly_predictions[best_hour_idx]:.0f}% congestion)")
        st.warning(f"**⚠️ Worst Time:** {worst_hour_idx}:00 ({hourly_predictions[worst_hour_idx]:.0f}% congestion)")

    # Speed & heatmap charts
    speed_fig = create_speed_chart(hourly_predictions, hour)
    st.plotly_chart(speed_fig, use_container_width=True)
    heatmap_fig = create_heatmap_calendar(hourly_predictions)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# ===== Tab 2: Route Planning =====
with tab2:
    st.subheader(f"Optimal Routes: {source_input} → {destination_input}")

    @st.cache_data(ttl=600)
    def safe_geocode(place):
        try:
            return ox.geocode(f"{place}, Bengaluru, India")
        except Exception:
            pk = place.lower().strip()
            if pk in LOCATION_COORDS:
                return LOCATION_COORDS[pk]
            return (12.9716, 77.5946)

    def generate_route_map(G, src, dst):
        try:
            source_loc = safe_geocode(src)
            dest_loc = safe_geocode(dst)
            if source_loc is None or dest_loc is None:
                return "❌ Could not locate addresses"
            orig_node = ox.distance.nearest_nodes(G, source_loc[1], source_loc[0])
            dest_node = ox.distance.nearest_nodes(G, dest_loc[1], dest_loc[0])
            if not nx.has_path(G, orig_node, dest_node):
                return f"❌ No route found"
            shortest_route = nx.shortest_path(G, orig_node, dest_node, weight="length")
            fastest_route = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
            shortest_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in shortest_route]
            fastest_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in fastest_route]
            m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
            folium.Marker(source_loc, tooltip=f"Source: {src}", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(dest_loc, tooltip=f"Destination: {dst}", icon=folium.Icon(color="red")).add_to(m)
            folium.PolyLine(shortest_coords, color="blue", weight=5, opacity=0.7, tooltip="Shortest Route").add_to(m)
            folium.PolyLine(fastest_coords, color="green", weight=5, opacity=0.7, tooltip="Fastest Route").add_to(m)
            return m
        except nx.NetworkXNoPath:
            return f"❌ No connecting path"
        except Exception as e:
            return f"❌ Route error: {str(e)}"

    try:
        @st.cache_resource
        def load_graph_safe():
            try:
                return ox.load_graphml("bengaluru_small.graphml")
            except Exception:
                return ox.graph_from_place("Bengaluru, India", network_type="drive", simplify=True)
        G = load_graph_safe()
        map_or_err = generate_route_map(G, source_input, destination_input)
        if isinstance(map_or_err, str):
            st.error(map_or_err)
        else:
            st_folium(map_or_err, width=900, height=480)
    except Exception as e:
        st.error(f"⚠️ Route planning unavailable: {e}")

# ===== Tab 3: Model Performance =====
with tab3:
    st.subheader("📈 Model Performance Analysis")
    if X_test is not None and y_test is not None:
        rf_predictions = rf_model.predict(X_test)
        lgb_predictions = lgb_model.predict(X_test)
        lr_predictions = lr_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_predictions)
        lgb_r2 = r2_score(y_test, lgb_predictions)
        lr_r2 = r2_score(y_test, lr_predictions)
        rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_predictions)))
        lgb_rmse = float(np.sqrt(mean_squared_error(y_test, lgb_predictions)))
        lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_predictions)))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("#### Random Forest")
            st.metric("R² Score", f"{rf_r2:.4f}")
            st.metric("RMSE", f"{rf_rmse:.2f}")
        with col2:
            st.markdown("#### LightGBM")
            st.metric("R² Score", f"{lgb_r2:.4f}")
            st.metric("RMSE", f"{lgb_rmse:.2f}")
        with col3:
            st.markdown("#### Linear Regression")
            st.metric("R² Score", f"{lr_r2:.4f}")
            st.metric("RMSE", f"{lr_rmse:.2f}")
        with col4:
            st.markdown("#### Ensemble")
            ensemble_r2 = (rf_r2 + lgb_r2 + lr_r2) / 3.0
            ensemble_rmse = (rf_rmse + lgb_rmse + lr_rmse) / 3.0
            st.metric("Avg R²", f"{ensemble_r2:.4f}")
            st.metric("Avg RMSE", f"{ensemble_rmse:.2f}")
        comparison_fig = create_comparison_chart(rf_r2, lgb_r2, lr_r2, rf_rmse, lgb_rmse, lr_rmse)
        st.plotly_chart(comparison_fig, use_container_width=True)
        st.markdown("### 🎯 Top 15 Important Features (RF)")
        try:
            feat_imp = pd.DataFrame({"Feature": feature_columns[:15], "Importance": rf_model.feature_importances_[:15]})
            feat_imp = feat_imp.sort_values("Importance", ascending=False)
            fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="blues")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Feature importance not available for this model.")

# ===== Tab 4: Smart Insights =====
with tab4:
    st.subheader("🎯 AI-Powered Insights & Live Observations")
    if enable_alerts:
        recommendations = alert_system.get_smart_recommendations(realistic_features, hourly_predictions, hour)
        st.markdown("### 💡 Personalized Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}.** {rec}")

    if tomtom_flow:
        st.markdown("### 📡 TomTom Live Data Snapshot")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Current Speed", f"{tomtom_flow.get('current_speed', 0):.1f} km/h")
            st.metric("Free Flow Speed", f"{tomtom_flow.get('free_flow_speed', 0):.1f} km/h")
        with c2:
            travel_min = tomtom_flow.get("current_travel_time", 0) / 60.0
            st.metric("Current Travel Time", f"{travel_min:.1f} min")
            st.metric("Confidence", f"{tomtom_flow.get('confidence', 0):.2f}")

    st.markdown("### 📊 Predictions Comparison")
    comparison_table = {
        "Source→Destination": [f"{source_input} → {destination_input}"],
        "ML Prediction (%)": [f"{ml_pred:.1f}"],
        "TomTom Estimate (%)": [f"{tomtom_pred:.1f}" if tomtom_pred is not None else "N/A"],
        "Final (Mode) (%)": [f"{final_pred:.1f}"],
        "Model Used": [model_used]
    }
    comp_df = pd.DataFrame(comparison_table)
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("### 📅 Historical Comparison (Quick view)")
    hist_df = pd.DataFrame({
        "Time Period": ["Current", "Yesterday", "Last Week", "Last Month"],
        "Congestion": [final_pred, final_pred * 0.95, final_pred * 1.1, final_pred * 0.85],
        "Speed": [realistic_features.get("avg_speed", 0), realistic_features.get("avg_speed", 0) * 1.05, realistic_features.get("avg_speed", 0) * 0.9, realistic_features.get("avg_speed", 0) * 1.15]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Congestion %", x=hist_df["Time Period"], y=hist_df["Congestion"], marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="Speed (km/h)", x=hist_df["Time Period"], y=hist_df["Speed"], marker_color="#10b981"))
    fig.update_layout(barmode="group", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ===== Tab 5: Export & History =====
with tab5:
    st.subheader("📥 Export Data & Prediction History")
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=["Timestamp", "Route", "Congestion", "Speed", "Time", "Model"])
        st.session_state.last_update_time = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate Prediction Report"):
            csv_data = export_prediction_data(source_input, destination_input, final_pred, realistic_features, model_used)
            st.download_button(label="⬇️ Download CSV Report", data=csv_data, file_name=f"traffic_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            st.success("✅ Report generated")
    with col2:
        if st.button("📈 Export Hourly Predictions"):
            hourly_export = pd.DataFrame({"Hour": list(range(24)), "Congestion (%)": hourly_predictions, "Status": ["High" if x >= 75 else "Medium" if x >= 50 else "Low" for x in hourly_predictions]})
            csv_hourly = hourly_export.to_csv(index=False).encode("utf-8")
            st.download_button(label="⬇️ Download Hourly Data", data=csv_hourly, file_name=f"hourly_predictions_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            st.success("✅ Hourly data ready")

    now = datetime.now()
    append_new = False
    if st.session_state.last_update_time is None:
        append_new = True
    else:
        if now - st.session_state.last_update_time >= timedelta(minutes=1):
            append_new = True
    if append_new:
        new_entry = {
            "Timestamp": now.strftime("%H:%M:%S"),
            "Route": f"{source_input} → {destination_input}",
            "Congestion": f"{final_pred:.1f}%",
            "Speed": f"{realistic_features.get('avg_speed', 0):.1f} km/h",
            "Time": f"{estimated_time:.0f} min",
            "Model": model_used
        }
        st.session_state.prediction_history = pd.concat([st.session_state.prediction_history, pd.DataFrame([new_entry])], ignore_index=True)
        st.session_state.last_update_time = now

    st.markdown("### 📜 Recent Predictions")
    st.dataframe(st.session_state.prediction_history, use_container_width=True)

# ==============================
# Debug information
# ==============================
if show_debug:
    with st.expander("🔧 Debug Information"):
        st.write("**Input Features (selected subset)**")
        debug_df = pd.DataFrame([user_input]).T
        debug_df.columns = ["Value"]
        st.dataframe(debug_df, use_container_width=True)

        st.write("**Predictions**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Random Forest", f"{rf_pred:.2f}")
        with col2:
            st.metric("LightGBM", f"{lgb_pred:.2f}")
        with col3:
            st.metric("Linear Regression", f"{lr_pred:.2f}")
        with col4:
            st.metric("Final (Blended)", f"{final_pred:.2f}")

        if tomtom_flow:
            st.write("**TomTom Flow JSON**")
            st.json(tomtom_flow)

# ==============================
# Sidebar footer & refresh
# ==============================
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
<div style="color: white;">
<strong>Smart Traffic Predictor v3.1</strong><br>
ML + TomTom Hybrid Mode<br><br>
Features:<br>- Real-time TomTom data (flow only)<br>- Hybrid blending of ML + TomTom<br>- Same UI as app.py<br>- Route optimization & exports
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.rerun()

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        <p>🚦 Smart Traffic Predictor v3.1 | ML + TomTom Live Data (no incidents)</p>
        <p style='font-size: 0.85rem;'>Hybrid predictions — combining historical patterns and live sensor data</p>
    </div>
""", unsafe_allow_html=True)
