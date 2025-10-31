import pandas as pd
import joblib
import numpy as np

# Load artifacts
model = joblib.load("traffic_model_key.pkl")
scaler = joblib.load("scaler_key.pkl")
encoders = joblib.load("encoders_key.pkl")
feature_columns = joblib.load("feature_columns_key.pkl")
le_target = joblib.load("le_target.pkl")

# Important features to ask user
important_features = ['time_of_day','day_of_week','borough','temperature','humidity','vehicle_count','avg_speed_kmph']

# Default ranges for other features
defaults = {
    'location': 'Industrial',
    'weather': 'Clear',
    'vehicle_type': 'Car',
    'accidents_reported': (0,5),
    'road_condition_score': (3,10),
    'holiday_flag': 0
}

# Collect user input
user_input = {}
for feature in important_features:
    if feature in encoders:
        options = list(encoders[feature].classes_)
        value = input(f"{feature} options {options}: ")
        if value not in options:
            print(f"Invalid input! Using default: {options[0]}")
            value = options[0]
        user_input[feature] = value
    else:
        user_input[feature] = float(input(f"Enter {feature} (numeric): "))

# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# Add defaults
for col, val in defaults.items():
    if col not in user_df.columns:
        if isinstance(val, tuple):
            user_df[col] = np.random.randint(val[0], val[1]+1)
        else:
            user_df[col] = val

# Encode categorical columns
for col, le in encoders.items():
    user_df[col] = le.transform(user_df[col])

# Scale numeric columns
numeric_cols = ['temperature','humidity','avg_speed_kmph','vehicle_count','accidents_reported','road_condition_score']
user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

# Ensure correct column order
user_df = user_df[feature_columns]

# Predict
pred = model.predict(user_df)
pred_label = le_target.inverse_transform(pred)

print(f"\n🚦 Predicted Congestion Level: {pred_label[0]}")
