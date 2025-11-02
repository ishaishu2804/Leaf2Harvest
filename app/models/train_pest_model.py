print("Script started.")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

try:
    print("[INFO] Loading dataset...")
    df = pd.read_csv('data/pest_data.csv')
    print("[INFO] Dataset loaded. Shape:", df.shape)

    # Encode crop type
    le = LabelEncoder()
    df['crop_type'] = le.fit_transform(df['crop_type'])
    print("[INFO] Crop type encoded.")

    X = df[['temperature', 'humidity', 'rainfall', 'crop_type']]
    y = df['pest_risk']

    # Train the model
    print("[INFO] Training model...")
    model = RandomForestClassifier()
    model.fit(X, y)
    print("[INFO] Model trained.")

    # Save model and encoder
    os.makedirs('app/models', exist_ok=True)
    print("[INFO] Saving model and encoder...")
    joblib.dump(model, 'app/models/pest_model.pkl')
    joblib.dump(le, 'app/models/crop_encoder.pkl')
    print("✅ Pest model and encoder saved.")
except Exception as e:
    print("[ERROR]", e)
