import joblib
import numpy as np
import os

# Get the absolute path to the models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'app', 'models')

model = joblib.load(os.path.join(MODELS_DIR, 'crop_model.pkl'))

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_features)
    return prediction[0]
