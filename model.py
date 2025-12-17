from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import streamlit as st

@st.cache_resource
def train_models(df, target_column="Congestion Level"):
    if df is None or df.empty:
        return None, None, None, None, None, None, None

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    return rf_model, lgb_model, lr_model, X, X_test, y_test, X.columns