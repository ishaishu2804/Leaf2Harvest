import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('pest_data.csv')

# Encode crop_type
le = LabelEncoder()
df['crop_type'] = le.fit_transform(df['crop_type'])

X = df[['temperature', 'humidity', 'rainfall', 'crop_type']]
y = df['pest_risk']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoder
joblib.dump(model, 'models/pest_model.pkl')
joblib.dump(le, 'models/crop_encoder.pkl')
