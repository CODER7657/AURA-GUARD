const express = require('express');
const router = express.Router();
const logger = require('../utils/logger');
const { successResponse, errorResponse, asyncHandler } = require('../utils/helpers');
const NASATempoAIService = require('../services/ai/NASATempoAIService');
const { NO2ForecastService } = require('../services/ai/NO2ForecastService');
const { askAnalyst } = require('../services/ai/ForecastAnalystService');

// Initialize NASA TEMPO AI Service
const aiService = new NASATempoAIService();
const no2Forecasts = new NO2ForecastService();

// Real NO2 model forecasts (ml/results/forecasts.json): issued from the last hour in the
// newest EPA AirData file, so real model output but not live.
router.get('/no2/stations', asyncHandler(async (req, res) => successResponse(
  res, no2Forecasts.listStations(), 'NO2 forecast stations retrieved successfully',
)));

router.get('/no2/forecast', asyncHandler(async (req, res) => {
  const { station } = req.query;
  if (station) {
    const result = no2Forecasts.getStation(station);
    if (!result) return errorResponse(res, `No NO2 forecast for station ${station}`, 404);
    return successResponse(res, result, 'NO2 forecast retrieved successfully');
  }
  const latitude = parseFloat(req.query.latitude);
  const longitude = parseFloat(req.query.longitude);
  if (Number.isNaN(latitude) || Number.isNaN(longitude)) {
    return errorResponse(res, 'Give a station id, or latitude and longitude', 400);
  }
  const result = no2Forecasts.nearest(latitude, longitude);
  if (!result) return errorResponse(res, 'No NO2 forecast station within 80 km of that point', 404);
  return successResponse(res, result, 'NO2 forecast for the nearest station retrieved successfully');
}));

// Forecast analyst: an LLM that answers only through forecast tools, with a
// deterministic grounding check on every answer (ml/src/auraguard/agent).
router.post('/ask', asyncHandler(async (req, res) => {
  const question = typeof req.body.question === 'string' ? req.body.question.trim() : '';
  if (!question || question.length > 500) {
    return errorResponse(res, 'question must be a string of 1 to 500 characters', 400);
  }
  try {
    const result = await askAnalyst(question);
    return successResponse(res, {
      answer: result.answer,
      source: result.source,
      grounded: result.grounded,
      tool_calls: result.tool_calls.filter((t) => t.by !== 'fallback').map((t) => ({ name: t.name, args: t.args })),
      latency_s: result.latency_s,
    }, 'Forecast analyst answered');
  } catch (error) {
    logger.error('Forecast analyst error:', error);
    return errorResponse(res, 'The forecast analyst is unavailable', 503);
  }
}));

// Current AirNow reading plus an AQI forecast that is still SIMULATED; the response also
// carries the real NO2 model forecast for the nearest station, when one is within 80 km.
router.post('/realtime', asyncHandler(async (req, res) => {
  const { latitude, longitude, forecast_hours = 1 } = req.body;
  
  if (!latitude || !longitude) {
    return errorResponse(res, 'Latitude and longitude are required', 400);
  }
  
  logger.info('Real-time prediction request', { latitude, longitude, forecast_hours });
  
  try {
    // Use the actual NASA TEMPO AI Service
    const prediction = await aiService.getPrediction(latitude, longitude, forecast_hours);
    prediction.no2_forecast = no2Forecasts.nearest(latitude, longitude);

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

// Extended forecast (SIMULATED: built from the current reading with random variation)
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
    timestamp: new Date().toISOString(),
    services: {
      no2_forecasts: no2Forecasts.data.stations.length
        ? `real model forecasts for ${no2Forecasts.data.stations.length} stations, not live`
        : 'missing',
      aqi_forecasts: 'simulated',
      tempo_data: 'not integrated',
    },
  };
  
  return successResponse(res, healthStatus, 'AI service is healthy');
}));

module.exports = router;
