import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("Loading data...")
df = pd.read_csv('ai4i2020.csv')

print("Your exact columns:", list(df.columns))
print(df.head(3))

# Auto-find numeric columns (temp, speed, torque, wear)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'] if col in numeric_cols]
if not features:
    features = numeric_cols[:5]  # Use first 5 numerics if no match

print(f"Using features: {features}")
X = df[features].dropna()

print(f"Training on {X.shape[0]} rows...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.03, random_state=42)
model.fit(X_scaled)

anomalies = model.predict(X_scaled)
df['anomaly'] = anomalies
df['anomaly_score'] = model.decision_function(X_scaled)

anomaly_count = (anomalies == -1).sum()
print(f"🚨 Found {anomaly_count} anomalies!")

print("\nSample results:")
result_cols = features + ['anomaly']
print(df[result_cols].head(10))

df.to_csv('maintenance_data_with_anomalies.csv', index=False)
print("\n✅ Model saved!")
