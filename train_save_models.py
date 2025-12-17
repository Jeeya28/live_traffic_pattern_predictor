import pandas as pd
import joblib
from data_loader import load_data
from model import train_models

# 1️⃣ Load your dataset
df = load_data()

# 2️⃣ Train models
rf_model, lgb_model, lr_model, X, X_test, y_test, feature_columns = train_models(df)

# 3️⃣ Save models and other necessary objects
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(lgb_model, "lgb_model.pkl")
joblib.dump(lr_model, "lr_model.pkl") 
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

print("✅ Models and feature columns saved successfully!")