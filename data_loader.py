import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path=r"C:\Users\Hp\Desktop\Prediction-Dashboard\Bangalore_Traffic_Cleaned.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found!")
        return None