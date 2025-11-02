#!/usr/bin/env python3
"""
Database Initialization Script for LeafToHarvest

This script creates the database tables including the new api_usage_logs table
for tracking OpenAI API usage and costs.
"""

import os
import sys
from sqlalchemy import create_engine
from app.models import Base

def main():
    """Initialize the database with all tables."""
    
    # Get the database path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATABASE_PATH = os.path.join(BASE_DIR, 'app', 'site.db')
    
    print(f"Initializing database at: {DATABASE_PATH}")
    
    try:
        # Create database engine
        engine = create_engine(f'sqlite:///{DATABASE_PATH}?check_same_thread=False')
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        print("✅ Database initialized successfully!")
        print("📊 Created tables:")
        print("   - users")
        print("   - products") 
        print("   - orders")
        print("   - order_items")
        print("   - diagnoses")
        print("   - usage_logs")
        print("   - api_usage_logs (NEW)")
        print("   - weather_data")
        print("   - pest_risk_data")
        print("   - disease_data")
        
        print("\n🚀 You can now run the Flask application!")
        print("   python app.py")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
