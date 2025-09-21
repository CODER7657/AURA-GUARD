const express = require('express');
const router = express.Router();
const { getSequelize } = require('../config/database');
const { getRedisClient } = require('../config/redis');
const logger = require('../utils/logger');
const config = require('../config/config');

// Simple health check
router.get('/', (req, res) => {
  res.status(200).json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: config.appVersion,
    environment: config.nodeEnv
  });
});

// Detailed health check
router.get('/detailed', async (req, res) => {
  const health = {
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: config.appVersion,
    environment: config.nodeEnv,
    services: {}
  };

  // Check PostgreSQL
  try {
    const sequelize = getSequelize();
    if (sequelize) {
      await sequelize.authenticate();
      health.services.postgresql = {
        status: 'connected',
        host: sequelize.config.host,
        database: sequelize.config.database
      };
    } else {
      health.services.postgresql = {
        status: 'disconnected'
      };
      health.status = 'DEGRADED';
    }
  } catch (error) {
    health.services.postgresql = {
      status: 'error',
      error: error.message
    };
    health.status = 'UNHEALTHY';
  }

  // Check Redis
  try {
    const redisClient = getRedisClient();
    if (redisClient && redisClient.isReady) {
      await redisClient.ping();
      health.services.redis = {
        status: 'connected'
      };
    } else {
      health.services.redis = {
        status: 'disconnected'
      };
      health.status = health.status === 'OK' ? 'DEGRADED' : health.status;
    }
  } catch (error) {
    health.services.redis = {
      status: 'error',
      error: error.message
    };
    health.status = 'DEGRADED';
  }

  // Memory usage
  const memUsage = process.memoryUsage();
  health.memory = {
    rss: `${Math.round(memUsage.rss / 1024 / 1024)}MB`,
    heapTotal: `${Math.round(memUsage.heapTotal / 1024 / 1024)}MB`,
    heapUsed: `${Math.round(memUsage.heapUsed / 1024 / 1024)}MB`,
    external: `${Math.round(memUsage.external / 1024 / 1024)}MB`
  };

  // CPU usage (approximation)
  const cpuUsage = process.cpuUsage();
  health.cpu = {
    user: cpuUsage.user,
    system: cpuUsage.system
  };

  logger.info('Health check performed', { status: health.status });

  const statusCode = health.status === 'OK' ? 200 : 
                    health.status === 'DEGRADED' ? 200 : 503;
  
  res.status(statusCode).json(health);
});

module.exports = router;