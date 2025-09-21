const logger = require('../utils/logger');

const notFound = (req, res, next) => {
  logger.warn('404 Not Found:', {
    url: req.originalUrl,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });

  res.status(404).json({
    success: false,
    error: {
      message: `Route ${req.originalUrl} not found`,
      statusCode: 404
    },
    timestamp: new Date().toISOString(),
    path: req.originalUrl,
    availableEndpoints: {
      auth: '/api/v1/auth',
      airQuality: '/api/v1/air-quality',
      predictions: '/api/v1/predictions',
      notifications: '/api/v1/notifications',
      health: '/health',
      docs: '/api/docs'
    }
  });
};

module.exports = notFound;