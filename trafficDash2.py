import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Configuration
# --------------------------
st.set_page_config(page_title="Traffic Predictor", layout="wide")

# Exact column mappings from your dataset
AREA_COLUMNS = [
    'Area Name_Hebbal',
    'Area Name_Indiranagar', 
    'Area Name_Jayanagar',
    'Area Name_Koramangala',
    'Area Name_M.G. Road',
    'Area Name_Whitefield',
    'Area Name_Yeshwanthpur'
]

ROAD_COLUMNS = [
    'Road/Intersection Name_Anil Kumble Circle',
    'Road/Intersection Name_Ballari Road',
    'Road/Intersection Name_CMH Road',
    'Road/Intersection Name_Hebbal Flyover',
    'Road/Intersection Name_Hosur Road',
    'Road/Intersection Name_ITPL Main Road',
    'Road/Intersection Name_Jayanagar 4th Block',
    'Road/Intersection Name_Marathahalli Bridge',
    'Road/Intersection Name_Sarjapur Road',
    'Road/Intersection Name_Silk Board Junction',
    'Road/Intersection Name_Sony World Junction',
    'Road/Intersection Name_South End Circle',
    'Road/Intersection Name_Trinity Circle',
    'Road/Intersection Name_Tumkur Road',
    'Road/Intersection Name_Yeshwanthpur Circle'
]

WEATHER_COLUMNS = [
    'Weather Conditions_Fog',
    'Weather Conditions_Overcast', 
    'Weather Conditions_Rain',
    'Weather Conditions_Windy'
]

# User-friendly mappings
LOCATION_MAPPING = {
    'Hebbal': 'Area Name_Hebbal',
    'Indiranagar': 'Area Name_Indiranagar',
    'Jayanagar': 'Area Name_Jayanagar', 
    'Koramangala': 'Area Name_Koramangala',
    'M.G. Road': 'Area Name_M.G. Road',
    'Whitefield': 'Area Name_Whitefield',
    'Yeshwanthpur': 'Area Name_Yeshwanthpur'
}

ROAD_MAPPING = {
    'Anil Kumble Circle': 'Road/Intersection Name_Anil Kumble Circle',
    'Ballari Road': 'Road/Intersection Name_Ballari Road',
    'CMH Road': 'Road/Intersection Name_CMH Road',
    'Hebbal Flyover': 'Road/Intersection Name_Hebbal Flyover',
    'Hosur Road': 'Road/Intersection Name_Hosur Road',
    'ITPL Main Road': 'Road/Intersection Name_ITPL Main Road',
    'Jayanagar 4th Block': 'Road/Intersection Name_Jayanagar 4th Block',
    'Marathahalli Bridge': 'Road/Intersection Name_Marathahalli Bridge',
    'Sarjapur Road': 'Road/Intersection Name_Sarjapur Road',
    'Silk Board Junction': 'Road/Intersection Name_Silk Board Junction',
    'Sony World Junction': 'Road/Intersection Name_Sony World Junction',
    'South End Circle': 'Road/Intersection Name_South End Circle',
    'Trinity Circle': 'Road/Intersection Name_Trinity Circle',
    'Tumkur Road': 'Road/Intersection Name_Tumkur Road',
    'Yeshwanthpur Circle': 'Road/Intersection Name_Yeshwanthpur Circle'
}

WEATHER_MAPPING = {
    'Fog': 'Weather Conditions_Fog',
    'Overcast': 'Weather Conditions_Overcast',
    'Rain': 'Weather Conditions_Rain', 
    'Windy': 'Weather Conditions_Windy'
}

# --------------------------
# Load and cache data
# --------------------------
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(r"C:\Users\Hp\Desktop\Bangalore_Traffic_Cleaned.csv")
        #st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please check the file path.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_models(df):
    """Train and cache the ML models"""
    if df is None or df.empty:
        return None, None, None, None, None, None
    
    try:
        # Prepare data
        target_column = 'Congestion Level'
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # Train LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42, 
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        return rf_model, lgb_model, X, X_test, y_test, X.columns
        
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        return None, None, None, None, None, None

def create_realistic_features(hour, day_of_week, month, day, area, road, weather="Overcast"):
    """Create realistic feature values based on time and location"""
    
    # Time-based patterns
    if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
        base_volume = np.random.normal(45000, 8000)
        base_speed = np.random.normal(20, 5)
        capacity_util = np.random.normal(85, 10)
        incidents = np.random.poisson(2)
    elif 10 <= hour <= 16:  # Day time
        base_volume = np.random.normal(30000, 5000)
        base_speed = np.random.normal(35, 7)
        capacity_util = np.random.normal(65, 15)
        incidents = np.random.poisson(1)
    else:  # Night/early morning
        base_volume = np.random.normal(15000, 3000)
        base_speed = np.random.normal(45, 8)
        capacity_util = np.random.normal(40, 10)
        incidents = np.random.poisson(0.5)
    
    # Weekend adjustment
    if day_of_week in [6, 7]:  # Weekend
        base_volume *= 0.7
        base_speed *= 1.2
        capacity_util *= 0.8
    
    # Weather adjustment
    if weather == "Rain":
        base_speed *= 0.7
        capacity_util *= 1.3
        incidents += 1
    elif weather == "Fog":
        base_speed *= 0.6
        incidents += 1
    
    # Clamp values to realistic ranges
    traffic_volume = max(5000, min(70000, base_volume))
    avg_speed = max(10, min(60, base_speed))
    capacity_util = max(20, min(100, capacity_util))
    incidents = max(0, min(5, int(incidents)))
    
    # Calculate derived features
    travel_time_index = max(1.0, 2.0 - (avg_speed / 30))
    env_impact = 50 + (traffic_volume / 1000) + (100 - avg_speed)
    
    return {
        'traffic_volume': traffic_volume,
        'avg_speed': avg_speed,
        'travel_time_index': travel_time_index,
        'capacity_util': capacity_util,
        'incidents': incidents,
        'env_impact': min(200, env_impact),
        'public_transport': 40 + (capacity_util * 0.3),  # More PT when congested
        'signal_compliance': max(70, 95 - (incidents * 5)),
        'parking_usage': min(95, 40 + (capacity_util * 0.6)),
        'pedestrian_count': max(20, 100 - (avg_speed * 1.5))
    }

def create_prediction_input(hour, day, month, weekday, area, road, weather, realistic_features):
    """Create properly formatted input for prediction"""
    
    # Start with base features
    user_input = {
        'Traffic Volume': realistic_features['traffic_volume'],
        'Average Speed': realistic_features['avg_speed'],
        'Travel Time Index': realistic_features['travel_time_index'],
        'Road Capacity Utilization': realistic_features['capacity_util'],
        'Incident Reports': realistic_features['incidents'],
        'Environmental Impact': realistic_features['env_impact'],
        'Public Transport Usage': realistic_features['public_transport'],
        'Traffic Signal Compliance': realistic_features['signal_compliance'],
        'Parking Usage': realistic_features['parking_usage'],
        'Pedestrian and Cyclist Count': realistic_features['pedestrian_count'],
        'Roadwork and Construction Activity': np.random.randint(0, 3),
        'Day': day,
        'Month': month,
        'Weekday': weekday
    }
    
    return user_input

def set_dummy_variables(user_input, df_columns, area, road, weather):
    """Set dummy variables correctly using exact column names"""
    
    # Initialize ALL dummy variables to 0
    for col in AREA_COLUMNS + ROAD_COLUMNS + WEATHER_COLUMNS:
        user_input[col] = 0
    
    # Set area dummy variable (only one should be 1)
    area_col = LOCATION_MAPPING.get(area)
    if area_col:
        user_input[area_col] = 1
    
    # Set road dummy variable (only one should be 1)  
    road_col = ROAD_MAPPING.get(road)
    if road_col:
        user_input[road_col] = 1
    
    # Set weather dummy variable (only one should be 1)
    weather_col = WEATHER_MAPPING.get(weather)
    if weather_col:
        user_input[weather_col] = 1
    
    return user_input

# --------------------------
# Main App
# --------------------------
st.title("🚦 AI Traffic Prediction & Route Optimization")
st.markdown("---")

# Load data and train models
df = load_data()

if df is not None:
    rf_model, lgb_model, X, X_test, y_test, feature_columns = train_models(df)
    
    if rf_model is not None:
       # st.success("✅ Models trained successfully!")
        
        # --------------------------
        # Sidebar Inputs
        # --------------------------
        st.sidebar.header("🔧 Prediction Controls")
        
        # Location inputs
        source_input = st.sidebar.selectbox(
            "Select Source", 
            list(LOCATION_MAPPING.keys()),
            index=2  # Default to Koramangala
        )
        
        destination_input = st.sidebar.selectbox(
            "Select Destination", 
            list(LOCATION_MAPPING.keys()),
            index=1  # Default to Electronic City
        )
        
        # Time inputs
        col1, col2 = st.sidebar.columns(2)
        with col1:
            date = st.date_input("Select Date")
        with col2:
            time = st.time_input("Select Time", value=datetime.now().time())
        
        # Road and weather inputs with exact options from your data
        road_input = st.sidebar.selectbox(
            "Select Road/Intersection",
            [
                'CMH Road', 'Hosur Road', 'Hebbal Flyover', 'Silk Board Junction',
                'ITPL Main Road', 'Ballari Road', 'Sarjapur Road', 'Marathahalli Bridge',
                'Anil Kumble Circle', 'Trinity Circle', 'South End Circle', 
                'Sony World Junction', 'Jayanagar 4th Block', 'Tumkur Road', 
                'Yeshwanthpur Circle'
            ],
            index=0
        )
        
        weather_input = st.sidebar.selectbox(
            "Weather Condition",
            ["Overcast", "Rain", "Fog", "Windy"],
            index=0
        )
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Settings"):
            model_choice = st.radio("Select Model", ["Random Forest", "LightGBM", "Ensemble"])
            show_debug = st.checkbox("Show Debug Info")
        
        # --------------------------
        # Process inputs and make prediction
        # --------------------------
        hour = time.hour
        day_of_week = date.weekday() + 1
        day = date.day
        month = date.month
        
        # Generate realistic features
        realistic_features = create_realistic_features(
            hour, day_of_week, month, day, source_input, road_input, weather_input
        )
        
        # Create input for prediction
        user_input = create_prediction_input(
            hour, day, month, day_of_week, source_input, road_input, weather_input, realistic_features
        )
        
        # Set dummy variables
        user_input = set_dummy_variables(user_input, feature_columns, source_input, road_input, weather_input)
        
        # Ensure all columns are present
        for col in feature_columns:
            if col not in user_input:
                user_input[col] = 0
        
        # Create DataFrame for prediction
        user_df = pd.DataFrame([user_input])
        user_df_ordered = user_df.reindex(columns=feature_columns, fill_value=0)
        
        # Make predictions
        try:
            rf_pred = rf_model.predict(user_df_ordered)[0]
            lgb_pred = lgb_model.predict(user_df_ordered)[0]
            
            if model_choice == "Random Forest":
                pred = rf_pred
                model_used = "Random Forest"
            elif model_choice == "LightGBM":
                pred = lgb_pred
                model_used = "LightGBM"
            else:  # Ensemble
                pred = (rf_pred + lgb_pred) / 2
                model_used = "Ensemble (RF + LGB)"
            
            # Ensure prediction is within valid range
            pred = max(0, min(100, pred))
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            pred = 50  # Default fallback
            model_used = "Fallback"
        
        # --------------------------
        # Display Results
        # --------------------------
        
        # Main prediction display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if pred >= 75:
                st.metric("🔴 Congestion Level", f"{pred:.1f}%", "High Traffic")
            elif pred >= 50:
                st.metric("🟡 Congestion Level", f"{pred:.1f}%", "Medium Traffic")
            elif pred >= 25:
                st.metric("🟢 Congestion Level", f"{pred:.1f}%", "Low Traffic")
            else:
                st.metric("🟢 Congestion Level", f"{pred:.1f}%", "Very Low Traffic")
        
        with col2:
            st.metric("🚗 Traffic Volume", f"{realistic_features['traffic_volume']:,.0f}")
        
        with col3:
            st.metric("⚡ Average Speed", f"{realistic_features['avg_speed']:.1f} km/h")
        
        with col4:
            st.metric("🤖 Model Used", model_used)
        
        # --------------------------
        # Tabs for detailed analysis
        # --------------------------
        tab1, tab2, tab3 = st.tabs(["📊 Traffic Analysis", "🗺️ Route Planning", "📈 Model Performance"])
        
        with tab1:
            st.subheader("Traffic Pattern Analysis")
            
            # Generate hourly predictions for the day
            hourly_predictions = []
            for h in range(24):
                h_features = create_realistic_features(
                    h, day_of_week, month, day, source_input, road_input, weather_input
                )
                h_input = create_prediction_input(
                    h, day, month, day_of_week, source_input, road_input, weather_input, h_features
                )
                h_input = set_dummy_variables(h_input, feature_columns, source_input, road_input, weather_input)
                
                # Fill missing columns
                for col in feature_columns:
                    if col not in h_input:
                        h_input[col] = 0
                
                h_df = pd.DataFrame([h_input])
                h_df_ordered = h_df.reindex(columns=feature_columns, fill_value=0)
                
                if model_choice == "Random Forest":
                    h_pred = rf_model.predict(h_df_ordered)[0]
                elif model_choice == "LightGBM":
                    h_pred = lgb_model.predict(h_df_ordered)[0]
                else:
                    h_pred = (rf_model.predict(h_df_ordered)[0] + lgb_model.predict(h_df_ordered)[0]) / 2
                
                hourly_predictions.append(max(0, min(100, h_pred)))
            
            # Plot hourly congestion
            hourly_df = pd.DataFrame({
                'Hour': range(24),
                'Congestion Level': hourly_predictions
            })
            
            fig = px.line(hourly_df, x='Hour', y='Congestion Level', 
                         title=f"Predicted Congestion - {source_input} to {destination_input}",
                         markers=True)
            
            # Highlight current hour
            fig.add_vline(x=hour, line_dash="dash", line_color="red", 
                         annotation_text=f"Current Time ({hour}:00)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Current Conditions")
                conditions_df = pd.DataFrame({
                    'Factor': ['Traffic Volume', 'Average Speed', 'Road Capacity', 'Incidents', 'Weather Impact'],
                    'Value': [
                        f"{realistic_features['traffic_volume']:,.0f}",
                        f"{realistic_features['avg_speed']:.1f} km/h",
                        f"{realistic_features['capacity_util']:.1f}%",
                        f"{realistic_features['incidents']} reports",
                        weather_input
                    ]
                })
                st.dataframe(conditions_df, use_container_width=True)
            
            with col2:
                st.write("### Recommendations")
                if pred >= 75:
                    st.error("🔴 **High Congestion Predicted**")
                    st.write("• Consider delaying travel by 1-2 hours")
                    st.write("• Use public transport if available")
                    st.write("• Allow extra 30-40 minutes")
                elif pred >= 50:
                    st.warning("🟡 **Moderate Congestion Expected**")
                    st.write("• Allow extra 15-20 minutes")
                    st.write("• Consider alternative routes")
                elif pred >= 25:
                    st.info("🟢 **Good Travel Conditions**")
                    st.write("• Normal travel time expected")
                    st.write("• Good time to travel")
                else:
                    st.success("🟢 **Excellent Travel Conditions**")
                    st.write("• Optimal time for travel")
                    st.write("• Minimal delays expected")
        
        with tab2:
            st.subheader("Route Planning")
            
            # Create map
            m = folium.Map(location=[12.9716, 77.5946], zoom_start=11)
            
            # Location coordinates for exact areas in your dataset
            coords = {
                'Hebbal': [13.0359, 77.5890],
                'Indiranagar': [12.9719, 77.6412], 
                'Jayanagar': [12.9249, 77.5834],
                'Koramangala': [12.9352, 77.6245],
                'M.G. Road': [12.9767, 77.6099],
                'Whitefield': [12.9698, 77.7500],
                'Yeshwanthpur': [13.0284, 77.5547]
            }
            
            start_coords = coords.get(source_input, [12.9352, 77.6245])  # Default to Koramangala
            end_coords = coords.get(destination_input, [12.9719, 77.6412])  # Default to Indiranagar
            
            # Add markers
            folium.Marker(start_coords, popup=f"Source: {source_input}", 
                         icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker(end_coords, popup=f"Destination: {destination_input}", 
                         icon=folium.Icon(color="red", icon="stop")).add_to(m)
            
            # Route line with color based on congestion
            if pred >= 75:
                route_color = "red"
            elif pred >= 50:
                route_color = "orange"
            else:
                route_color = "green"
            
            folium.PolyLine([start_coords, end_coords], 
                           color=route_color, weight=6, opacity=0.8).add_to(m)
            
            st_folium(m, width=700, height=400)
            
            # Route summary
            st.write("### Route Summary")
            
            # Calculate estimated time based on congestion
            base_time = 35  # Base travel time in minutes
            if pred >= 75:
                estimated_time = base_time * 1.6
            elif pred >= 50:
                estimated_time = base_time * 1.3
            else:
                estimated_time = base_time * 1.0
            
            summary_df = pd.DataFrame({
                'Metric': ['Route', 'Distance', 'Normal Time', 'Predicted Time', 'Extra Time', 'Congestion'],
                'Value': [
                    f"{source_input} → {destination_input}",
                    "~18 km",
                    f"{base_time} min",
                    f"{estimated_time:.0f} min",
                    f"+{estimated_time-base_time:.0f} min",
                    f"{pred:.1f}%"
                ]
            })
            st.table(summary_df)
        
        with tab3:
            st.subheader("Model Performance & Insights")
            
            if X_test is not None and y_test is not None:
                # Calculate model metrics
                rf_predictions = rf_model.predict(X_test)
                lgb_predictions = lgb_model.predict(X_test)
                
                rf_r2 = r2_score(y_test, rf_predictions)
                lgb_r2 = r2_score(y_test, lgb_predictions)
                rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
                lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_predictions))
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Random Forest")
                    st.metric("R² Score", f"{rf_r2:.4f}")
                    st.metric("RMSE", f"{rf_rmse:.2f}")
                
                with col2:
                    st.write("#### LightGBM")
                    st.metric("R² Score", f"{lgb_r2:.4f}")
                    st.metric("RMSE", f"{lgb_rmse:.2f}")
                
                # Model comparison chart
                comparison_df = pd.DataFrame({
                    'Model': ['Random Forest', 'LightGBM'],
                    'R² Score': [rf_r2, lgb_r2],
                    'RMSE': [rf_rmse, lgb_rmse]
                })
                
                fig = px.bar(comparison_df, x='Model', y='R² Score', 
                            title="Model Accuracy Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': feature_columns[:15],  # Top 15 features
                    'Importance': rf_model.feature_importances_[:15]
                }).sort_values('Importance', ascending=True)
                
                fig2 = px.bar(feature_importance, x='Importance', y='Feature', 
                             orientation='h', title="Top 15 Most Important Features")
                st.plotly_chart(fig2, use_container_width=True)
        
        # Debug information
        if show_debug:
            with st.expander("🔧 Debug Information"):
                st.write("**Input Features:**")
                debug_df = pd.DataFrame([user_input]).T
                debug_df.columns = ['Value']
                st.dataframe(debug_df)
                
                st.write("**Model Predictions:**")
                st.write(f"Random Forest: {rf_pred:.2f}")
                st.write(f"LightGBM: {lgb_pred:.2f}")
                st.write(f"Final Prediction: {pred:.2f}")
    
    else:
        st.error("❌ Failed to train models. Please check your dataset.")
else:
    st.error("❌ Please ensure your dataset is available and properly formatted.")