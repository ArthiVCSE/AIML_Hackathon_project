import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("🚦 Starting Traffic Congestion Model Training...")

# Load dataset
print("\n📂 Loading dataset...")
data = pd.read_csv("city_traffic_volume_dataset.csv")
print(f"✅ Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")

# Fill missing numeric values
print("\n🧮 Handling missing numeric values...")
numeric_cols = ['temperature','humidity','avg_speed_kmph','vehicle_count','accidents_reported','road_condition_score']
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
print("✅ Numeric missing values filled with mean.")

# Fill missing categorical values
print("\n🔤 Handling missing categorical values...")
categorical_cols = ['time_of_day','day_of_week','borough','location','weather','vehicle_type','holiday_flag']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
print("✅ Categorical missing values filled with mode.")

# Encode categorical variables
print("\n🔢 Encoding categorical variables...")
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
print("✅ All categorical columns encoded successfully.")

# Encode target
print("\n🎯 Encoding target variable 'congestion_level'...")
target = 'congestion_level'
le_target = LabelEncoder()
y = le_target.fit_transform(data[target])
print(f"✅ Target encoding complete. Classes: {list(le_target.classes_)}")

# Prepare features
feature_columns = numeric_cols + categorical_cols
X = data[feature_columns].copy()

# Scale numeric features
print("\n📏 Scaling numeric features...")
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print("✅ Numeric features standardized.")

# Train-test split
print("\n📚 Splitting data into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split complete. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Train Random Forest
print("\n🌲 Training Random Forest model (200 trees)...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Save artifacts
print("\n💾 Saving trained model and encoders...")
joblib.dump(model, "traffic_model_key.pkl")
joblib.dump(scaler, "scaler_key.pkl")
joblib.dump(encoders, "encoders_key.pkl")
joblib.dump(feature_columns, "feature_columns_key.pkl")
joblib.dump(le_target, "le_target.pkl")
print("✅ All files saved successfully.")

print("\n🎉 Training pipeline completed successfully!")
