import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
print("[INFO] Loading dataset...")
df = pd.read_csv('data/Crop_recommendation.csv')
print("[INFO] Dataset loaded. Shape:", df.shape)

# Prepare features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train the model
print("[INFO] Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("[INFO] Model trained.")

# Create directory if it doesn't exist
os.makedirs('app/models', exist_ok=True)

# Save model
print("[INFO] Saving model...")
joblib.dump(model, 'app/models/crop_model.pkl')
print("✅ Crop model saved successfully!") 