"""
Configuration Module
Contains all mappings and constants for the traffic prediction system
"""

# Exact column mappings from dataset
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

# Location coordinates for mapping
LOCATION_COORDINATES = {
    'Hebbal': (13.0359, 77.5890),
    'Indiranagar': (12.9719, 77.6412),
    'Jayanagar': (12.9249, 77.5834),
    'Koramangala': (12.9352, 77.6245),
    'M.G. Road': (12.9767, 77.6099),
    'Whitefield': (12.9698, 77.7500),
    'Yeshwanthpur': (13.0284, 77.5547)
}

# Traffic thresholds
CONGESTION_THRESHOLDS = {
    'low': 25,
    'medium': 50,
    'high': 75,
    'critical': 90
}

# Speed categories (km/h)
SPEED_CATEGORIES = {
    'very_slow': 15,
    'slow': 30,
    'moderate': 45,
    'fast': 60
}

# Time categories
PEAK_HOURS = {
    'morning': (7, 10),
    'evening': (17, 20),
    'night': (22, 5)
}

# Default values
DEFAULT_VALUES = {
    'base_travel_time': 35,  # minutes
    'default_distance': 18,  # km
    'fuel_cost_per_km': 8.5,  # INR
    'toll_cost': 50,  # INR
}

# Color scheme for visualizations
COLOR_SCHEME = {
    'low': '#10b981',      # Green
    'medium': '#fbbf24',   # Yellow
    'high': '#f97316',     # Orange
    'critical': '#ef4444', # Red
    'primary': '#3b82f6',  # Blue
    'secondary': '#8b5cf6' # Purple
}

# API configurations (if using external APIs)
API_CONFIG = {
    'openweather_base_url': 'http://api.openweathermap.org/data/2.5/weather',
    'google_maps_base_url': 'https://maps.googleapis.com/maps/api/directions/json',
    'timeout': 10  # seconds
}