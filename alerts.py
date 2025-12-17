"""
Smart Alert and Notification System
Provides intelligent traffic alerts and recommendations
"""

import streamlit as st
from datetime import datetime, timedelta

class TrafficAlertSystem:
    """Smart alert system for traffic conditions"""
    
    def __init__(self):
        self.alert_thresholds = {
            'high_congestion': 75,
            'very_high_congestion': 90,
            'low_speed': 15,
            'high_incidents': 3,
            'moderate_congestion': 50
        }
    
    def check_alerts(self, current_data):
        """Check if any alert conditions are met"""
        alerts = []
        
        congestion = current_data.get('congestion', 0)
        avg_speed = current_data.get('avg_speed', 30)
        incidents = current_data.get('incidents', 0)
        weather = current_data.get('weather', 'Overcast')
        
        # Critical congestion alert
        if congestion >= self.alert_thresholds['very_high_congestion']:
            alerts.append({
                'level': 'critical',
                'message': f"CRITICAL: Severe congestion detected ({congestion:.0f}%)",
                'action': "Consider alternative routes or delay travel by 1-2 hours"
            })
        
        # High congestion warning
        elif congestion >= self.alert_thresholds['high_congestion']:
            alerts.append({
                'level': 'warning',
                'message': f"WARNING: High congestion ({congestion:.0f}%)",
                'action': "Expect 30-40% longer travel time. Plan accordingly."
            })
        
        # Moderate congestion info
        elif congestion >= self.alert_thresholds['moderate_congestion']:
            alerts.append({
                'level': 'info',
                'message': f"MODERATE: Normal traffic conditions ({congestion:.0f}%)",
                'action': "Travel is manageable with minor delays expected."
            })
        
        # Low speed alert
        if avg_speed < self.alert_thresholds['low_speed']:
            alerts.append({
                'level': 'warning',
                'message': f"SLOW TRAFFIC: Average speed {avg_speed:.0f} km/h",
                'action': "Heavy traffic ahead. Consider waiting or taking alternative route."
            })
        
        # Incident alert
        if incidents >= self.alert_thresholds['high_incidents']:
            alerts.append({
                'level': 'warning',
                'message': f"INCIDENTS: {incidents} incidents reported on route",
                'action': "Drive carefully. Road disruptions expected."
            })
        
        # Weather-based alerts
        if weather == "Rain":
            alerts.append({
                'level': 'info',
                'message': "WEATHER: Rain detected on route",
                'action': "Allow 40% extra travel time. Drive cautiously."
            })
        elif weather == "Fog":
            alerts.append({
                'level': 'warning',
                'message': "WEATHER: Fog conditions present",
                'action': "Visibility reduced. Use fog lights and reduce speed."
            })
        
        return alerts
    
    def display_alerts(self, alerts):
        """Display alerts in Streamlit UI with proper formatting"""
        if not alerts:
            st.success("ALL CLEAR - No traffic alerts. Conditions are normal!")
            return
        
        st.markdown("### Active Traffic Alerts")
        
        for alert in alerts:
            if alert['level'] == 'critical':
                st.error(f"**{alert['message']}**\n\n{alert['action']}")
            elif alert['level'] == 'warning':
                st.warning(f"**{alert['message']}**\n\n{alert['action']}")
            else:
                st.info(f"**{alert['message']}**\n\n{alert['action']}")
    
    def get_smart_recommendations(self, current_data, hourly_predictions, current_hour):
        """Provide AI-powered travel recommendations"""
        recommendations = []
        
        congestion = current_data.get('congestion', 0)
        avg_speed = current_data.get('avg_speed', 30)
        traffic_volume = current_data.get('traffic_volume', 30000)
        weather = current_data.get('weather', 'Overcast')
        
        # Find best travel time in next 6 hours
        next_6_hours = hourly_predictions[current_hour:min(current_hour + 6, 24)]
        if next_6_hours:
            best_hour_idx = next_6_hours.index(min(next_6_hours))
            
            if best_hour_idx > 0:
                actual_hour = current_hour + best_hour_idx
                recommendations.append(
                    f"Optimal Travel Time: In {best_hour_idx} hour(s) at {actual_hour}:00 "
                    f"(expected congestion: {next_6_hours[best_hour_idx]:.0f}%)"
                )
            else:
                recommendations.append(
                    "Best Time: Right now! Current conditions are optimal for travel."
                )
        
        # Current vs optimal comparison
        if congestion > 70:
            time_saved = int((congestion - min(hourly_predictions)) * 0.5)
            recommendations.append(
                f"Time Savings: You could save ~{time_saved} minutes by traveling at off-peak hours"
            )
        
        # Weekend vs weekday recommendation
        current_day = datetime.now().weekday()
        if current_day < 5:  # Weekday
            recommendations.append(
                "Weekend Tip: This route typically has 30-40% less congestion on weekends"
            )
        else:
            recommendations.append(
                "Weekend Travel: You're traveling during off-peak time. Enjoy lighter traffic!"
            )
        
        # Weather-based recommendations
        if weather == "Rain":
            recommendations.append(
                "Weather Advisory: Rain detected. Allow 40% extra travel time and use headlights"
            )
        elif weather == "Fog":
            recommendations.append(
                "Weather Advisory: Foggy conditions. Reduce speed by 30% and use fog lights"
            )
        
        # Speed-based recommendations
        if avg_speed < 20:
            recommendations.append(
                "Alternative: Consider public transport. Metro/Bus might be faster in heavy traffic"
            )
        
        # Volume-based recommendations
        if traffic_volume > 50000:
            recommendations.append(
                "High Volume: Very heavy traffic. Consider carpooling or working from home if possible"
            )
        
        # Time of day recommendations
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            recommendations.append(
                "Rush Hour Alert: You're traveling during peak hours. Expect maximum congestion"
            )
        elif 22 <= current_hour or current_hour <= 5:
            recommendations.append(
                "Night Travel: Low traffic conditions. Fastest travel times expected"
            )
        
        # Environmental recommendation
        env_impact = current_data.get('env_impact', 100)
        if env_impact > 120:
            recommendations.append(
                "Eco Tip: High environmental impact. Consider public transport or carpooling to reduce emissions"
            )
        
        return recommendations