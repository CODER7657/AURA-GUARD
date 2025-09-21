const { Redis } = require('@upstash/redis');
const logger = require('../utils/logger');

let redisClient = null;

// Initialize Upstash Redis client
try {
  if (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN) {
    redisClient = new Redis({
      url: process.env.UPSTASH_REDIS_REST_URL,
      token: process.env.UPSTASH_REDIS_REST_TOKEN,
    });
    
    logger.info('🔄 Redis: Upstash Redis client initialized');
  } else {
    logger.warn('⚠️ Redis: Environment variables not found, Redis disabled');
  }
} catch (error) {
  logger.error('❌ Redis initialization failed:', error.message);
  redisClient = null;
}

// Cache utility functions
const cache = {
  async get(key) {
    if (!redisClient) return null;
    try {
      const value = await redisClient.get(key);
      return value;
    } catch (error) {
      logger.error('Redis GET error:', error.message);
      return null;
    }
  },

  async set(key, value, ttl = 3600) {
    if (!redisClient) return false;
    try {
      await redisClient.set(key, value, { ex: ttl });
      return true;
    } catch (error) {
      logger.error('Redis SET error:', error.message);
      return false;
    }
  },

  async del(key) {
    if (!redisClient) return false;
    try {
      await redisClient.del(key);
      return true;
    } catch (error) {
      logger.error('Redis DEL error:', error.message);
      return false;
    }
  },

  async exists(key) {
    if (!redisClient) return false;
    try {
      const result = await redisClient.exists(key);
      return result === 1;
    } catch (error) {
      logger.error('Redis EXISTS error:', error.message);
      return false;
    }
  }
};

// Test Redis connection
const connectRedis = async () => {
  try {
    if (redisClient) {
      const result = await redisClient.ping();
      logger.info('✅ Upstash Redis connected successfully (REST API)');
      return true;
    }
    return false;
  } catch (error) {
    logger.error('❌ Redis connection test failed:', error.message);
    return false;
  }
};

// Get Redis client instance
const getRedisClient = () => {
  return redisClient;
};

// Legacy function aliases for backward compatibility
const setCache = async (key, value, ttl = 3600) => {
  return await cache.set(key, value, ttl);
};

const getCache = async (key) => {
  return await cache.get(key);
};

const deleteCache = async (key) => {
  return await cache.del(key);
};

const flushCache = async () => {
  if (!redisClient) return false;
  try {
    await redisClient.flushall();
    return true;
  } catch (error) {
    logger.error('Redis flush error:', error.message);
    return false;
  }
};

// Graceful shutdown - Upstash doesn't need cleanup
process.on('SIGTERM', () => {
  logger.info('🔄 Redis: Graceful shutdown (Upstash REST client - no cleanup needed)');
});

process.on('SIGINT', () => {
  logger.info('🔄 Redis: Graceful shutdown (Upstash REST client - no cleanup needed)');
});

module.exports = {
  connectRedis,
  getRedisClient,
  setCache,
  getCache,
  deleteCache,
  flushCache,
  cache,
  client: redisClient,
  isConnected: () => redisClient !== null
};