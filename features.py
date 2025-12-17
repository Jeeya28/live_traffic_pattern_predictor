import numpy as np
from config import AREA_COLUMNS, ROAD_COLUMNS, WEATHER_COLUMNS, LOCATION_MAPPING, ROAD_MAPPING, WEATHER_MAPPING

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
    for col in AREA_COLUMNS + ROAD_COLUMNS + WEATHER_COLUMNS:
        user_input[col] = 0
    if LOCATION_MAPPING.get(area): user_input[LOCATION_MAPPING[area]] = 1
    if ROAD_MAPPING.get(road): user_input[ROAD_MAPPING[road]] = 1
    if WEATHER_MAPPING.get(weather): user_input[WEATHER_MAPPING[weather]] = 1
    return user_input