#!/usr/bin/env python3
"""
Environment Setup Script for LeafToHarvest OpenAI Integration

This script helps you set up the OPENAI_API_KEY environment variable
and test the integration.
"""

import os
import sys

def check_environment():
    """Check current environment configuration."""
    print("🔍 Checking Environment Configuration\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        print(f"✅ OPENAI_API_KEY is set: {api_key[:20]}...")
        print("✅ OpenAI features are enabled")
        return True
    else:
        print("❌ OPENAI_API_KEY is not set")
        print("⚠️  OpenAI features will use fallback analysis")
        return False

def setup_instructions():
    """Show setup instructions."""
    print("\n📋 Setup Instructions:")
    print("=" * 50)
    
    print("\n1. Get your OpenAI API Key:")
    print("   - Visit: https://platform.openai.com/api-keys")
    print("   - Create a new API key")
    print("   - Copy the key (starts with 'sk-')")
    
    print("\n2. Set the environment variable:")
    print("   Windows (PowerShell):")
    print("   $env:OPENAI_API_KEY='your-api-key-here'")
    print("   ")
    print("   Windows (Command Prompt):")
    print("   set OPENAI_API_KEY=your-api-key-here")
    print("   ")
    print("   Linux/Mac:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    
    print("\n3. Restart the Flask application:")
    print("   python app.py")
    
    print("\n4. Test the integration:")
    print("   python test_integration.py")

def test_with_api_key():
    """Test if API key is valid."""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No API key found to test")
        return False
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("✅ API key is valid and working!")
        print(f"✅ Test response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 LeafToHarvest OpenAI Integration Setup\n")
    
    # Check current environment
    has_api_key = check_environment()
    
    if has_api_key:
        print("\n🧪 Testing API key...")
        if test_with_api_key():
            print("\n🎉 Everything is configured correctly!")
            print("✅ You can now use OpenAI features in your application")
        else:
            print("\n⚠️  API key is set but not working")
            print("Please check your API key and try again")
    else:
        setup_instructions()
    
    print("\n📚 Additional Resources:")
    print("- Usage Dashboard: http://localhost:5000/usage")
    print("- Disease Diagnosis: http://localhost:5000/openai-diagnosis")
    print("- Admin Overview: http://localhost:5000/admin/usage-overview")
    print("- Documentation: OpenAI_Integration_Guide.md")

if __name__ == '__main__':
    main()
