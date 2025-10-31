import pandas as pd
import numpy as np
import random

# Number of rows to generate
rows = 5000   # you can increase this up to 50,000 if needed

# Feature categories
times = ['Morning', 'Afternoon', 'Evening', 'Night']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weather = ['Clear', 'Cloudy', 'Rainy', 'Foggy', 'Stormy', 'Windy']
locations = ['Downtown', 'Uptown', 'Suburb', 'Industrial', 'Highway', 'Airport', 'Market']
boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
vehicle_types = ['Car', 'Truck', 'Bus', 'Motorbike', 'Bicycle']

# Generate random data
data = {
    'date': pd.date_range(start='2023-01-01', periods=rows, freq='H'),
    'time_of_day': np.random.choice(times, rows),
    'day_of_week': np.random.choice(days, rows),
    'borough': np.random.choice(boroughs, rows),
    'location': np.random.choice(locations, rows),
    'temperature': np.random.uniform(15, 42, rows).round(1),
    'humidity': np.random.randint(25, 95, rows),
    'weather': np.random.choice(weather, rows),
    'vehicle_type': np.random.choice(vehicle_types, rows),
    'avg_speed_kmph': np.random.uniform(10, 80, rows).round(1),
    'vehicle_count': np.random.randint(100, 5000, rows),
    'accidents_reported': np.random.randint(0, 5, rows),
    'road_condition_score': np.random.randint(1, 10, rows),
    'holiday_flag': np.random.choice([0, 1], rows, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# Derive congestion level based on traffic & speed
def get_congestion(volume, speed):
    if volume < 1000 and speed > 40:
        return 'LOW'
    elif (1000 <= volume < 3000) or (20 < speed <= 40):
        return 'MEDIUM'
    else:
        return 'HIGH'

df['congestion_level'] = df.apply(lambda x: get_congestion(x['vehicle_count'], x['avg_speed_kmph']), axis=1)

# Save CSV
df.to_csv("city_traffic_volume_dataset.csv", index=False)
print(f"✅ city_traffic_volume_dataset.csv created successfully with {rows} rows and {len(df.columns)} columns!")
print("\nColumns:", list(df.columns))
print(df.head())
