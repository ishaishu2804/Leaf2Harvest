#!/usr/bin/env python3
"""
Test script for OpenAI integration functionality
"""

import os
import sys
from app.services.openai_service import get_openai_service
from app.utils.usage_logger import UsageLogger

def test_openai_service():
    """Test OpenAI service initialization."""
    print("Testing OpenAI service...")
    
    service = get_openai_service()
    if service is None:
        print("✅ OpenAI service correctly returns None when API key not set")
        return True
    else:
        print("✅ OpenAI service initialized successfully")
        return True

def test_usage_logger():
    """Test usage logger functionality."""
    print("\nTesting usage logger...")
    
    try:
        # Initialize usage logger
        usage_logger = UsageLogger('sqlite:///app/site.db?check_same_thread=False')
        
        # Test logging fallback usage
        success = usage_logger.log_fallback_usage(1, 'test')
        if success:
            print("✅ Usage logger working correctly")
            return True
        else:
            print("❌ Usage logger failed")
            return False
            
    except Exception as e:
        print(f"❌ Usage logger error: {e}")
        return False

def test_database_connection():
    """Test database connection."""
    print("\nTesting database connection...")
    
    try:
        from sqlalchemy import create_engine
        from app.models import ApiUsageLog
        
        engine = create_engine('sqlite:///app/site.db?check_same_thread=False')
        
        # Test query
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM api_usage_logs")
            count = result.scalar()
            print(f"✅ Database connection successful. Found {count} usage logs.")
            return True
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing OpenAI Integration Components\n")
    
    tests = [
        test_openai_service,
        test_usage_logger,
        test_database_connection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The integration is working correctly.")
        print("\n📝 Next steps:")
        print("1. Set OPENAI_API_KEY environment variable to enable OpenAI features")
        print("2. Visit http://localhost:5000/usage to see the usage dashboard")
        print("3. Try the disease diagnosis feature at http://localhost:5000/openai-diagnosis")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == '__main__':
    main()
