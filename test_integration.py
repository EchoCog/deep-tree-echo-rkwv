#!/usr/bin/env python3
"""
Test script for the parentheses bootstrap system integration
Tests the API endpoints and cognitive grammar integration
"""

import requests
import json
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_bootstrap_system_direct():
    """Test the bootstrap system directly"""
    print("🧪 Testing Parentheses Bootstrap System (Direct)")
    print("=" * 50)
    
    from cognitive_grammar import get_bootstrap_system
    
    bootstrap = get_bootstrap_system()
    
    # Test Spencer-Brown calculus
    test_cases = [
        "()",           # void
        "(())",         # mark -> void
        "((()))",       # nested
        "(+ 1 2)",      # complex expression
    ]
    
    for expr in test_cases:
        try:
            result = bootstrap.bootstrap_eval(expr)
            print(f"'{expr}' → {result}")
        except Exception as e:
            print(f"'{expr}' → Error: {e}")
    
    # Test system status
    status = bootstrap.get_system_status()
    print(f"\nSystem Status: {status['status']}")
    print(f"Environment size: {status['environment_size']}")
    print(f"Built-ins: {', '.join(status['builtins'][:5])}...")

def test_api_endpoints():
    """Test the API endpoints (requires running server)"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 30)
    
    base_url = "http://localhost:8000"
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/api/parentheses-bootstrap/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status endpoint working")
            print(f"   Capabilities: {len(data.get('capabilities', []))}")
            print(f"   Examples available: {len(data.get('examples', {}))}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Server not running - skipping API tests")
        return False
    except Exception as e:
        print(f"❌ Status endpoint error: {e}")
    
    # Test evaluation endpoint
    test_expressions = [
        "()",
        "(())",
        "((()))"
    ]
    
    for expr in test_expressions:
        try:
            response = requests.post(
                f"{base_url}/api/parentheses-bootstrap/evaluate",
                json={"expression": expr},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ '{expr}' → {data['result']} ({data['evaluation_time_ms']}ms)")
            else:
                print(f"❌ '{expr}' → Error {response.status_code}")
        except Exception as e:
            print(f"❌ '{expr}' → Error: {e}")
    
    return True

def test_grammar_integration():
    """Test integration with grammar membrane"""
    print("\n🎭 Testing Grammar Membrane Integration")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    # Test with parenthetical expressions in regular text
    test_inputs = [
        "Can you evaluate (()) for me?",
        "What is the result of ((())) using Spencer-Brown calculus?",
        "I need help with lambda expressions and (car (cons a b))",
        "Please process this: () and ((()))"
    ]
    
    try:
        # Create a session first
        session_response = requests.post(f"{base_url}/api/session")
        if session_response.status_code != 200:
            print("❌ Could not create session")
            return False
        
        session_id = session_response.json()['session_id']
        print(f"✅ Created session: {session_id[:8]}...")
        
        # Test processing with parenthetical expressions
        for input_text in test_inputs:
            try:
                response = requests.post(
                    f"{base_url}/api/process",
                    json={"session_id": session_id, "input": input_text},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    grammar_output = data.get('membrane_outputs', {}).get('grammar', '')
                    if 'symbolic' in grammar_output.lower() or 'parenthetical' in grammar_output.lower():
                        print(f"✅ Detected symbolic processing in: '{input_text[:30]}...'")
                    else:
                        print(f"❔ Regular processing for: '{input_text[:30]}...'")
                else:
                    print(f"❌ Processing failed for: '{input_text[:30]}...'")
            except Exception as e:
                print(f"❌ Error processing: {e}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Server not running - skipping grammar integration tests")
        return False
    except Exception as e:
        print(f"❌ Grammar integration test error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Deep Tree Echo Parentheses Bootstrap Integration Tests")
    print("=" * 60)
    
    # Test direct system
    test_bootstrap_system_direct()
    
    # Test API endpoints (if server is running)
    api_working = test_api_endpoints()
    
    # Test grammar integration (if server is running)
    if api_working:
        test_grammar_integration()
    
    print("\n" + "=" * 60)
    print("✅ Testing completed!")
    print("\n💡 To test the full system:")
    print("   1. Run: python src/app.py")
    print("   2. Open: http://localhost:8000")
    print("   3. Try the cognitive interface with parenthetical expressions")
    print("   4. Use API endpoint: /api/parentheses-bootstrap/evaluate")

if __name__ == '__main__':
    main()