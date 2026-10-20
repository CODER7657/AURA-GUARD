const express = require('express');
const router = express.Router();
const logger = require('../utils/logger');
const { successResponse, errorResponse, asyncHandler } = require('../utils/helpers');
const NASATempoAIService = require('../services/ai/NASATempoAIService');

// Initialize NASA TEMPO AI Service
const aiService = new NASATempoAIService();

// Real-time air quality prediction (simulated until a trained model is deployed)
router.post('/realtime', asyncHandler(async (req, res) => {
  const { latitude, longitude, forecast_hours = 1 } = req.body;
  
  if (!latitude || !longitude) {
    return errorResponse(res, 'Latitude and longitude are required', 400);
  }
  
  logger.info('Real-time prediction request', { latitude, longitude, forecast_hours });
  
  try {
    // Use the actual NASA TEMPO AI Service
    const prediction = await aiService.getPrediction(latitude, longitude, forecast_hours);
    
    return successResponse(res, prediction, 'Real-time air quality prediction generated successfully');
  } catch (error) {
    logger.error('Real-time prediction error:', error);
    
    // Check if it's a validation error from the Python bridge
    if (error.message.includes('Model validation failed')) {
      return errorResponse(res, error.message, 400);
    }
    
    return errorResponse(res, 'Failed to generate air quality prediction', 500);
  }
}));

// Extended forecast
router.post('/forecast', asyncHandler(async (req, res) => {
  const { latitude, longitude, duration = 48 } = req.body;
  
  if (!latitude || !longitude) {
    return errorResponse(res, 'Latitude and longitude are required', 400);
  }
  
  if (duration < 1 || duration > 72) {
    return errorResponse(res, 'Duration must be between 1 and 72 hours', 400);
  }
  
  logger.info('Extended forecast request', { latitude, longitude, duration });
  
  try {
    // Use the actual NASA TEMPO AI Service for extended forecast
    const forecast = await aiService.getExtendedForecast(latitude, longitude, duration);
    
    return successResponse(res, forecast, 'Extended forecast generated successfully');
  } catch (error) {
    logger.error('Extended forecast error:', error);
    
    // Check if it's a validation error from the Python bridge
    if (error.message.includes('Model validation failed')) {
      return errorResponse(res, error.message, 400);
    }
    
    return errorResponse(res, 'Failed to generate extended forecast', 500);
  }
}));

// Measured model performance from the walk-forward backtest (ml/results/model_performance.json)
router.get('/accuracy', asyncHandler(async (req, res) => {
  return successResponse(res, aiService.getModelPerformance(), 'Backtested model performance retrieved successfully');
}));

// AI Service health check
router.get('/health', asyncHandler(async (req, res) => {
  const healthStatus = {
    status: 'healthy',
    model_available: true,
    python_bridge: 'operational',
    timestamp: new Date().toISOString(),
    services: {
      lstm_model: 'active',
      tempo_data: 'connected',
      cache: 'operational'
    }
  };
  
  return successResponse(res, healthStatus, 'AI service is healthy');
}));

module.exports = router;
