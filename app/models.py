from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime
import os
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Get the absolute path to the database directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'site.db')

Base = declarative_base()

class User(Base, UserMixin):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    role = Column(String(20), nullable=False)
    subscription_type = Column(String(20), nullable=False, default='free')  # free, premium
    openai_usage_today = Column(Integer, nullable=False, default=0)
    last_usage_reset = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    subscription_expires = Column(DateTime, nullable=True)

    products = relationship('Product', back_populates='farmer')
    orders = relationship('Order', back_populates='consumer')
    diagnoses = relationship("Diagnosis", back_populates="user", cascade="all, delete-orphan")
    usage_logs = relationship("UsageLog", back_populates="user", cascade="all, delete-orphan")

    def __init__(self, username, email, role):
        self.username = username
        self.email = email
        self.role = role

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Product(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, autoincrement=True)
    farmer_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    price = Column(Float, nullable=False)
    unit = Column(String, nullable=False)
    quantity_available = Column(Float, nullable=False)
    image_url = Column(String)
    category = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    farmer = relationship('User', back_populates='products')
    order_items = relationship('OrderItem', back_populates='product')

class Order(Base):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    consumer_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    order_date = Column(DateTime, default=datetime.datetime.utcnow)
    total_price = Column(Float, nullable=False)
    status = Column(String, nullable=False, default='pending')

    consumer = relationship('User', back_populates='orders')
    items = relationship('OrderItem', back_populates='order')

class OrderItem(Base):
    __tablename__ = 'order_items'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    quantity = Column(Float, nullable=False)
    price_at_order = Column(Float, nullable=False)

    order = relationship('Order', back_populates='items')
    product = relationship('Product', back_populates='order_items')

class Diagnosis(Base):
    __tablename__ = 'diagnoses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    image_path = Column(String(255), nullable=False)
    disease_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    treatment = Column(Text, nullable=False)
    diagnosis_method = Column(String(20), nullable=False, default='fallback')  # openai, local_model, fallback
    crop_type = Column(String(50), nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="diagnoses")

class UsageLog(Base):
    __tablename__ = 'usage_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    service_type = Column(String(20), nullable=False)  # openai, local_model, fallback
    crop_type = Column(String(50), nullable=True)
    confidence_score = Column(Float, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="usage_logs")

class WeatherData(Base):
    __tablename__ = 'weather_data'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    location = Column(String(100), nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    date = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    
    user = relationship("User")

class PestRiskData(Base):
    __tablename__ = 'pest_risk_data'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    crop_type = Column(String(50), nullable=False)
    pest_risk_level = Column(String(20), nullable=False)  # low, medium, high
    risk_score = Column(Float, nullable=False)  # 0-1 scale
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    
    user = relationship("User")

class DiseaseData(Base):
    __tablename__ = 'disease_data'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    crop_type = Column(String(50), nullable=False)
    disease_name = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)  # low, medium, high
    humidity = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    
    user = relationship("User")

class ApiUsageLog(Base):
    __tablename__ = 'api_usage_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    model_name = Column(String(100), nullable=False)  # gpt-4o, local_model, etc.
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    estimated_cost = Column(Float, nullable=False, default=0.0)
    error_message = Column(Text, nullable=True)  # Store error if API call failed
    service_type = Column(String(20), nullable=False)  # openai, local_model, fallback
    confidence_score = Column(Float, nullable=True)  # For local model predictions
    
    user = relationship("User")

if __name__ == '__main__':
    engine = create_engine(f'sqlite:///{DATABASE_PATH}')
    Base.metadata.create_all(engine)
    print(f"Database and tables created at {DATABASE_PATH}") 