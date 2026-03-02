import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
num_samples = 1000

# Generate Synthetic Features
data = {
    'temp_celsius': np.random.uniform(20, 100, num_samples),
    'smoke_obscuration': np.random.uniform(0, 5, num_samples),
    'electrical_load_kw': np.random.uniform(100, 500, num_samples),
    'fuel_flow_rate': np.random.uniform(5, 50, num_samples),
    'engine_vibration': np.random.uniform(0.1, 1.0, num_samples)
}

df = pd.DataFrame(data)

# Define Risk Logic (0: Low, 1: Medium, 2: High)
def define_risk(row):
    # High Risk: High temp + Smoke OR High temp + High electrical load
    if (row['temp_celsius'] > 85 and row['smoke_obscuration'] > 3) or \
       (row['temp_celsius'] > 90 and row['electrical_load_kw'] > 450):
        return 2 
    # Medium Risk: Moderate temp elevation or high fuel flow instability
    elif row['temp_celsius'] > 70 or row['fuel_flow_rate'] > 40:
        return 1
    else:
        return 0

df['risk_level'] = df.apply(define_risk, axis=1)
df.to_csv('ship_fire_data.csv', index=False)
print("Dataset created and saved as 'ship_fire_data.csv'")