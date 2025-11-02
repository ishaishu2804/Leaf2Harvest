#!/usr/bin/env python3
"""
Sample data generation script for Crop Health Monitoring Dashboard
This script generates realistic sample data for testing the health monitoring features.
"""

import os
import sys
import random
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.models import Base, User, WeatherData, PestRiskData, DiseaseData

def generate_sample_health_data():
    """Generate sample health monitoring data for testing"""
    
    # Database setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATABASE_PATH = os.path.join(BASE_DIR, 'app', 'site.db')
    engine = create_engine(f'sqlite:///{DATABASE_PATH}?check_same_thread=False')
    Base.metadata.create_all(engine)
    
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    
    # Get or create a test user
    test_user = session.query(User).filter_by(username='test_farmer').first()
    if not test_user:
        test_user = User(username='test_farmer', email='test@example.com', role='farmer')
        test_user.set_password('password123')
        session.add(test_user)
        session.commit()
        print("Created test user: test_farmer")
    
    # Clear existing health data for this user
    session.query(WeatherData).filter_by(user_id=test_user.id).delete()
    session.query(PestRiskData).filter_by(user_id=test_user.id).delete()
    session.query(DiseaseData).filter_by(user_id=test_user.id).delete()
    session.commit()
    
    # Generate sample data for the last 30 days
    base_date = datetime.datetime.now() - datetime.timedelta(days=30)
    locations = ['Farm A', 'Farm B', 'Greenhouse 1', 'Field C']
    crop_types = ['wheat', 'rice', 'maize', 'cotton', 'tomato', 'potato']
    diseases = ['rust', 'blight', 'mildew', 'anthracnose', 'bacterial spot', 'fungal infection']
    
    print("Generating sample health monitoring data...")
    
    # Generate weather data
    for i in range(30):
        date = base_date + datetime.timedelta(days=i)
        
        # Generate realistic weather patterns
        temperature = random.uniform(15, 35) + random.uniform(-5, 5) * (i / 30)  # Seasonal variation
        humidity = random.uniform(40, 90)
        rainfall = random.uniform(0, 25) if random.random() > 0.7 else 0  # 30% chance of rain
        wind_speed = random.uniform(0, 15)
        pressure = random.uniform(1000, 1020)
        
        weather = WeatherData(
            user_id=test_user.id,
            location=random.choice(locations),
            temperature=round(temperature, 1),
            humidity=round(humidity, 1),
            rainfall=round(rainfall, 1),
            wind_speed=round(wind_speed, 1),
            pressure=round(pressure, 1),
            date=date
        )
        session.add(weather)
    
    # Generate pest risk data
    for i in range(25):
        date = base_date + datetime.timedelta(days=i)
        
        # Pest risk correlates with humidity and temperature
        temperature = random.uniform(18, 32)
        humidity = random.uniform(50, 95)
        rainfall = random.uniform(0, 20)
        
        # Calculate risk score based on environmental factors
        risk_score = 0.0
        if temperature > 25 and humidity > 70:
            risk_score += 0.4
        if humidity > 80:
            risk_score += 0.3
        if rainfall > 10:
            risk_score += 0.2
        risk_score += random.uniform(0, 0.3)  # Add some randomness
        risk_score = min(1.0, risk_score)
        
        risk_level = 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high'
        
        pest_risk = PestRiskData(
            user_id=test_user.id,
            crop_type=random.choice(crop_types),
            pest_risk_level=risk_level,
            risk_score=round(risk_score, 2),
            temperature=round(temperature, 1),
            humidity=round(humidity, 1),
            rainfall=round(rainfall, 1),
            date=date
        )
        session.add(pest_risk)
    
    # Generate disease data
    for i in range(20):
        date = base_date + datetime.timedelta(days=i)
        
        # Disease severity correlates with humidity
        humidity = random.uniform(60, 95)
        temperature = random.uniform(20, 30)
        rainfall = random.uniform(0, 15)
        
        # Calculate severity based on humidity
        if humidity > 85:
            severity = 'high' if random.random() > 0.3 else 'medium'
        elif humidity > 70:
            severity = 'medium' if random.random() > 0.4 else 'low'
        else:
            severity = 'low' if random.random() > 0.2 else 'medium'
        
        disease = DiseaseData(
            user_id=test_user.id,
            crop_type=random.choice(crop_types),
            disease_name=random.choice(diseases),
            severity=severity,
            humidity=round(humidity, 1),
            temperature=round(temperature, 1),
            rainfall=round(rainfall, 1),
            date=date
        )
        session.add(disease)
    
    session.commit()
    print(f"Generated sample data:")
    print(f"- 30 weather records")
    print(f"- 25 pest risk records")
    print(f"- 20 disease records")
    print(f"Data generated for user: {test_user.username}")
    
    session.close()

if __name__ == '__main__':
    generate_sample_health_data()
