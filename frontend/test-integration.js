// Frontend-Backend Integration Test
import axios from 'axios';

const API_BASE_URL = 'http://localhost:3000/api/v1';

async function testBackendIntegration() {
  console.log('🚀 Testing Frontend-Backend Integration...\n');

  try {
    // Test 1: Real-time prediction
    console.log('1️⃣ Testing Real-time Prediction Endpoint...');
    const predictionResponse = await axios.post(`${API_BASE_URL}/predictions/realtime`, {
      latitude: 34.0522,
      longitude: -118.2437,
      forecast_hours: 1
    });
    
    console.log('✅ Real-time prediction successful');
    console.log('   AQI:', predictionResponse.data.data.prediction.aqi);
    console.log('   Category:', predictionResponse.data.data.prediction.category);
    console.log('   Confidence:', predictionResponse.data.data.prediction.confidence);
    console.log('   Fallback Mode:', predictionResponse.data.data.satellite_data.fallback_mode);
    console.log();

    // Test 2: Model accuracy
    console.log('2️⃣ Testing Model Accuracy Endpoint...');
    const accuracyResponse = await axios.get(`${API_BASE_URL}/predictions/accuracy`);
    
    console.log('✅ Model accuracy fetch successful');
    console.log('   R² Score:', accuracyResponse.data.data.model_performance.r2_score);
    console.log('   NASA Compliance:', accuracyResponse.data.data.nasa_compliance.compliance_percentage + '%');
    console.log('   Inference Time:', accuracyResponse.data.data.model_performance.inference_time_ms + 'ms');
    console.log();

    // Test 3: Health check
    console.log('3️⃣ Testing Health Check Endpoint...');
    const healthResponse = await axios.get('http://localhost:3000/health/detailed');
    
    console.log('✅ Health check successful');
    console.log('   Status:', healthResponse.data.status);
    console.log('   PostgreSQL:', healthResponse.data.services.postgresql.status);
    console.log('   Uptime:', Math.floor(healthResponse.data.uptime / 60) + 'm ' + Math.floor(healthResponse.data.uptime % 60) + 's');
    console.log();

    // Test 4: Extended forecast
    console.log('4️⃣ Testing Extended Forecast Endpoint...');
    const forecastResponse = await axios.post(`${API_BASE_URL}/predictions/forecast`, {
      latitude: 40.7128,  // New York
      longitude: -74.0060,
      duration: 24
    });
    
    console.log('✅ Extended forecast successful');
    console.log('   Location: New York (40.7128, -74.0060)');
    console.log('   Duration: 24 hours');
    console.log('   Predictions:', forecastResponse.data.data.predictions?.length || 'N/A');
    console.log();

    console.log('🎉 ALL INTEGRATION TESTS PASSED!');
    console.log('📊 Frontend-Backend Connection: READY FOR PRODUCTION');

  } catch (error) {
    console.error('❌ Integration test failed:', error.message);
    if (error.response) {
      console.error('   Status:', error.response.status);
      console.error('   Data:', JSON.stringify(error.response.data, null, 2));
    }
  }
}

testBackendIntegration();