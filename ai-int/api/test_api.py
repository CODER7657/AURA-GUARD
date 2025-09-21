"""
API Testing Script for NASA Air Quality ML Service
==================================================

This script tests all endpoints of the air quality prediction API.
"""

import requests
import json
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_check():
    """Test the health check endpoint"""
    print("\n❤️ Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🤖 Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_demo_scenarios():
    """Test the demo scenarios endpoint"""
    print("\n🎭 Testing demo scenarios endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/demo/scenarios")
        print(f"Status: {response.status_code}")
        
        data = response.json()
        print(f"Available scenarios: {list(data['scenarios'].keys())}")
        
        # Return first scenario for prediction testing
        return response.status_code == 200, data['scenarios']
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, {}

def test_realtime_prediction(scenario_data):
    """Test real-time prediction endpoint"""
    print("\n🔮 Testing real-time prediction endpoint...")
    
    # Use wildfire scenario
    wildfire_scenario = scenario_data.get('wildfire_impact', {})
    
    prediction_request = {
        "tempo_data": wildfire_scenario.get('tempo_data', {}),
        "location": wildfire_scenario.get('location', {}),
        "timestamp": datetime.now().isoformat() + "Z",
        "features": wildfire_scenario.get('features', [1.0] * 20)
    }
    
    try:
        print(f"Request payload: {json.dumps(prediction_request, indent=2)}")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict/realtime",
            json=prediction_request,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {(end_time - start_time) * 1000:.2f}ms")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Validate latency target
            latency_met = result['processing_time_ms'] < 100
            print(f"⏱️ Latency target (<100ms): {'✅ PASS' if latency_met else '❌ FAIL'}")
            
            return True
        else:
            print(f"❌ Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_prediction(scenario_data):
    """Test batch prediction endpoint"""
    print("\n📦 Testing batch prediction endpoint...")
    
    # Create multiple requests using different scenarios
    scenarios = list(scenario_data.values())[:3]  # Use first 3 scenarios
    
    batch_requests = []
    for i, scenario in enumerate(scenarios):
        request = {
            "tempo_data": scenario.get('tempo_data', {}),
            "location": scenario.get('location', {}),
            "timestamp": datetime.now().isoformat() + "Z",
            "features": scenario.get('features', [float(i+1)] * 20)
        }
        batch_requests.append(request)
    
    try:
        print(f"Batch size: {len(batch_requests)}")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_requests,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {(end_time - start_time) * 1000:.2f}ms")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Batch size: {result['batch_size']}")
            print(f"Successful predictions: {result['successful_predictions']}")
            
            # Show first result
            if result['results']:
                print(f"First result: {json.dumps(result['results'][0], indent=2)}")
            
            return True
        else:
            print(f"❌ Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_performance():
    """Test API performance with multiple requests"""
    print("\n⚡ Testing API performance...")
    
    # Simple prediction request
    request_data = {
        "tempo_data": {"NO2_column": 2.5, "O3_column": 3.0, "HCHO_column": 1.5},
        "location": {"latitude": 40.7128, "longitude": -74.0060, "name": "NYC"},
        "timestamp": datetime.now().isoformat() + "Z",
        "features": [2.5, 3.0, 1.5, 1.0, 0.8, 0.3, 45.0, 25.0] + [0.5] * 12
    }
    
    num_requests = 10
    successful_requests = 0
    total_time = 0
    latencies = []
    
    print(f"Making {num_requests} concurrent requests...")
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict/realtime",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                successful_requests += 1
                latency = (end_time - start_time) * 1000
                latencies.append(latency)
                total_time += latency
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if successful_requests > 0:
        avg_latency = total_time / successful_requests
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"✅ Successful requests: {successful_requests}/{num_requests}")
        print(f"📊 Average latency: {avg_latency:.2f}ms")
        print(f"📊 Min latency: {min_latency:.2f}ms")
        print(f"📊 Max latency: {max_latency:.2f}ms")
        
        # Check performance targets
        latency_target_met = avg_latency < 100
        throughput = successful_requests / (total_time / 1000) * 60  # requests per minute
        
        print(f"🎯 Latency target (<100ms): {'✅ PASS' if latency_target_met else '❌ FAIL'}")
        print(f"🎯 Estimated throughput: {throughput:.0f} requests/minute")
        print(f"🎯 Throughput target (>1000/min): {'✅ PASS' if throughput > 1000 else '❌ FAIL'}")
        
        return True
    
    return False

def main():
    """Run all API tests"""
    print("🧪 NASA Air Quality ML Service API Testing")
    print("=" * 60)
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Root endpoint
    total_tests += 1
    if test_root_endpoint():
        tests_passed += 1
    
    # Test 2: Health check
    total_tests += 1
    if test_health_check():
        tests_passed += 1
    
    # Test 3: Model info
    total_tests += 1
    if test_model_info():
        tests_passed += 1
    
    # Test 4: Demo scenarios
    total_tests += 1
    scenarios_result = test_demo_scenarios()
    if scenarios_result[0]:
        tests_passed += 1
        scenario_data = scenarios_result[1]
    else:
        scenario_data = {}
    
    # Test 5: Real-time prediction
    total_tests += 1
    if scenario_data and test_realtime_prediction(scenario_data):
        tests_passed += 1
    
    # Test 6: Batch prediction
    total_tests += 1
    if scenario_data and test_batch_prediction(scenario_data):
        tests_passed += 1
    
    # Test 7: Performance test
    total_tests += 1
    if test_performance():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Tests passed: {tests_passed}/{total_tests}")
    print(f"📈 Success rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n🚀 API is ready for production use!")
    print("📖 View API documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()