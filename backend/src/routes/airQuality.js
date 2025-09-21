const express = require('express');
const router = express.Router();
const logger = require('../utils/logger');
const { successResponse, errorResponse, asyncHandler } = require('../utils/helpers');
const { validate, schemas } = require('../utils/validation');
const { optionalAuth, authenticateToken } = require('../middleware/auth');
const AirNowService = require('../services/AirNowService');
const TempoService = require('../services/TempoService');

// Get current air quality data (with optional authentication for enhanced features)
router.get('/current', optionalAuth, asyncHandler(async (req, res) => {
  const { lat, lon, distance = 25, sources = 'airnow,tempo' } = req.query;
  
  // Validate coordinates
  if (!lat || !lon) {
    return errorResponse(res, 'Latitude and longitude are required', 400);
  }
  
  const latitude = parseFloat(lat);
  const longitude = parseFloat(lon);
  
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
    return errorResponse(res, 'Invalid coordinates', 400);
  }

  logger.api('Current air quality data request', { latitude, longitude, distance, sources });
  
  try {
    const results = {};
    const sourceList = sources.split(',');
    
    // Fetch from AirNow if requested
    if (sourceList.includes('airnow')) {
      try {
        results.airnow = await AirNowService.getCurrentObservations(latitude, longitude, distance);
      } catch (error) {
        logger.warn('AirNow service error:', error.message);
        results.airnow = { error: error.message };
      }
    }
    
    // Fetch from TEMPO if requested
    if (sourceList.includes('tempo')) {
      try {
        results.tempo = await TempoService.getLatestData(latitude, longitude, distance);
      } catch (error) {
        logger.warn('TEMPO service error:', error.message);
        results.tempo = { error: error.message };
      }
    }
    
    return successResponse(res, results, 'Current air quality data retrieved successfully');
  } catch (error) {
    logger.error('Current air quality error:', error);
    return errorResponse(res, 'Failed to fetch current air quality data', 500);
  }
}));

// Get historical air quality data (authenticated users get extended history)
router.get('/historical', optionalAuth, asyncHandler(async (req, res) => {
  const { lat, lon, startDate, endDate, distance = 25, sources = 'airnow' } = req.query;
  
  // Validate required parameters
  if (!lat || !lon || !startDate || !endDate) {
    return errorResponse(res, 'Latitude, longitude, startDate, and endDate are required', 400);
  }
  
  const latitude = parseFloat(lat);
  const longitude = parseFloat(lon);
  
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
    return errorResponse(res, 'Invalid coordinates', 400);
  }

  logger.api('Historical air quality data request', { latitude, longitude, startDate, endDate, sources });
  
  try {
    const results = {};
    const sourceList = sources.split(',');
    
    // Fetch from AirNow if requested
    if (sourceList.includes('airnow')) {
      try {
        results.airnow = await AirNowService.getHistoricalData(latitude, longitude, startDate, endDate, distance);
      } catch (error) {
        logger.warn('AirNow historical service error:', error.message);
        results.airnow = { error: error.message };
      }
    }
    
    // Fetch from TEMPO if requested
    if (sourceList.includes('tempo')) {
      try {
        const startTime = new Date(startDate).toISOString();
        const endTime = new Date(endDate).toISOString();
        results.tempo = await TempoService.getTimeSeriesData(latitude, longitude, startTime, endTime);
      } catch (error) {
        logger.warn('TEMPO historical service error:', error.message);
        results.tempo = { error: error.message };
      }
    }
    
    return successResponse(res, results, 'Historical air quality data retrieved successfully');
  } catch (error) {
    logger.error('Historical air quality error:', error);
    return errorResponse(res, 'Failed to fetch historical air quality data', 500);
  }
}));

// Get air quality data by location
router.get('/location/:coordinates', optionalAuth, (req, res) => {
  const { coordinates } = req.params;
  logger.api('Location-based air quality request', { coordinates, user: req.user?.id });
  
  res.status(200).json({
    success: true,
    message: 'Location-based air quality data endpoint - Coming soon',
    data: {
      coordinates,
      user_authenticated: !!req.user,
      notice: 'Will provide air quality data for specific coordinates',
      planned_features: [
        'GPS coordinate support',
        'Nearest station data',
        'Interpolated values',
        'Area coverage information'
      ]
    }
  });
});

// Get user's saved locations and air quality data (protected route)
router.get('/my-locations', authenticateToken, asyncHandler(async (req, res) => {
  try {
    const user = req.user;
    const userLocations = user.preferences?.locations || [];
    
    logger.api('User locations request', { userId: user.id, locationCount: userLocations.length });
    
    if (userLocations.length === 0) {
      return successResponse(res, { locations: [] }, 'No saved locations found');
    }
    
    // Get air quality data for each saved location
    const locationData = await Promise.all(
      userLocations.map(async (location) => {
        try {
          const currentData = await AirNowService.getCurrentObservations(
            location.lat, 
            location.lon, 
            25
          );
          
          return {
            ...location,
            currentAirQuality: currentData,
            lastUpdated: new Date().toISOString()
          };
        } catch (error) {
          logger.warn(`Failed to get air quality for location ${location.name}:`, error.message);
          return {
            ...location,
            currentAirQuality: { error: error.message },
            lastUpdated: new Date().toISOString()
          };
        }
      })
    );
    
    return successResponse(res, { locations: locationData }, 'User locations with air quality data retrieved');
  } catch (error) {
    logger.error('User locations error:', error);
    return errorResponse(res, 'Failed to fetch user locations', 500);
  }
}));

// Get air quality stations/monitoring points
router.get('/stations', asyncHandler(async (req, res) => {
  const { lat, lon, distance = 50, sources = 'airnow' } = req.query;
  
  logger.api('Air quality stations request', { lat, lon, distance, sources });
  
  try {
    const results = {};
    const sourceList = sources.split(',');
    
    // If coordinates provided, get nearby stations
    if (lat && lon) {
      const latitude = parseFloat(lat);
      const longitude = parseFloat(lon);
      
      if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
        return errorResponse(res, 'Invalid coordinates', 400);
      }
      
      // Fetch from AirNow if requested
      if (sourceList.includes('airnow')) {
        try {
          results.airnow = await AirNowService.getMonitoringSites(latitude, longitude, distance);
        } catch (error) {
          logger.warn('AirNow stations service error:', error.message);
          results.airnow = { error: error.message };
        }
      }
      
      // TEMPO provides pixel data, not traditional stations
      if (sourceList.includes('tempo')) {
        results.tempo = {
          info: 'TEMPO provides satellite pixel data rather than traditional monitoring stations',
          coverage: 'Continental United States and surrounding areas',
          resolution: '2.1 x 4.4 km at nadir',
          parameters: TempoService.getAvailableParameters(),
        };
      }
    } else {
      // Return general information about monitoring networks
      results.info = {
        airnow: {
          description: 'EPA AirNow network provides ground-based monitoring stations',
          coverage: 'United States, Canada, and Mexico',
          update_frequency: 'Hourly',
        },
        tempo: {
          description: 'TEMPO satellite provides atmospheric composition data',
          coverage: 'North America',
          update_frequency: 'Hourly during daylight',
          parameters: TempoService.getAvailableParameters(),
        },
      };
    }
    
    return successResponse(res, results, 'Air quality monitoring stations retrieved successfully');
  } catch (error) {
    logger.error('Stations request error:', error);
    return errorResponse(res, 'Failed to fetch monitoring stations', 500);
  }
}));

// Get specific station data
router.get('/stations/:stationId', asyncHandler(async (req, res) => {
  const { stationId } = req.params;
  const { sources = 'airnow' } = req.query;
  
  logger.api('Specific station data request', { stationId, sources });
  
  try {
    const results = {};
    const sourceList = sources.split(',');
    
    // Note: This would require station-specific endpoints from the services
    // For now, return placeholder response
    results.info = {
      stationId,
      message: 'Station-specific data retrieval requires additional API endpoints',
      suggestion: 'Use coordinates-based queries for location-specific data',
    };
    
    return successResponse(res, results, 'Station data endpoint accessed');
  } catch (error) {
    logger.error('Station data error:', error);
    return errorResponse(res, 'Failed to fetch station data', 500);
  }
}));

// Get service status and availability
router.get('/services/status', asyncHandler(async (req, res) => {
  logger.api('Service status request');
  
  try {
    const status = {
      airnow: {
        name: 'EPA AirNow',
        status: 'checking',
        lastCheck: new Date().toISOString(),
      },
      tempo: {
        name: 'NASA TEMPO Satellite',
        status: 'checking',
        lastCheck: new Date().toISOString(),
      },
    };
    
    // Test AirNow connection
    try {
      const airnowTest = await AirNowService.testConnection();
      status.airnow.status = airnowTest ? 'operational' : 'unavailable';
      status.airnow.message = airnowTest ? 'Service responding normally' : 'Service connection failed';
    } catch (error) {
      status.airnow.status = 'error';
      status.airnow.message = error.message;
    }
    
    // Test TEMPO connection
    try {
      const tempoTest = await TempoService.testConnection();
      status.tempo.status = tempoTest ? 'operational' : 'unavailable';
      status.tempo.message = tempoTest ? 'Service responding normally' : 'Service connection failed';
    } catch (error) {
      status.tempo.status = 'error';
      status.tempo.message = error.message;
    }
    
    return successResponse(res, status, 'Service status retrieved successfully');
  } catch (error) {
    logger.error('Service status error:', error);
    return errorResponse(res, 'Failed to check service status', 500);
  }
}));

module.exports = router;