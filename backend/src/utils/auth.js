const jwt = require('jsonwebtoken');
const config = require('../config/config');
const logger = require('../utils/logger');

// Generate JWT token
const generateToken = (payload) => {
  return jwt.sign(payload, config.jwtSecret, {
    expiresIn: config.jwtExpire,
  });
};

// Generate refresh token
const generateRefreshToken = (payload) => {
  return jwt.sign(payload, config.jwtRefreshSecret, {
    expiresIn: config.jwtRefreshExpire,
  });
};

// Verify JWT token
const verifyToken = (token) => {
  try {
    return jwt.verify(token, config.jwtSecret);
  } catch (error) {
    throw new Error('Invalid token');
  }
};

// Verify refresh token
const verifyRefreshToken = (token) => {
  try {
    return jwt.verify(token, config.jwtRefreshSecret);
  } catch (error) {
    throw new Error('Invalid refresh token');
  }
};

// Extract token from request
const extractToken = (req) => {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    return authHeader.substring(7);
  }
  return null;
};

// Middleware to protect routes
const authenticate = (req, res, next) => {
  try {
    const token = extractToken(req);
    
    if (!token) {
      return res.status(401).json({
        success: false,
        error: {
          message: 'Access denied. No token provided.',
          statusCode: 401
        }
      });
    }

    const decoded = verifyToken(token);
    req.user = decoded;
    
    logger.auth('Token authenticated', { userId: decoded.id });
    next();
    
  } catch (error) {
    logger.auth('Authentication failed', { error: error.message });
    return res.status(401).json({
      success: false,
      error: {
        message: 'Invalid token',
        statusCode: 401
      }
    });
  }
};

// Middleware to check user roles
const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        error: {
          message: 'Access denied. User not authenticated.',
          statusCode: 401
        }
      });
    }

    if (!roles.includes(req.user.role)) {
      logger.auth('Authorization failed', { 
        userId: req.user.id, 
        userRole: req.user.role, 
        requiredRoles: roles 
      });
      
      return res.status(403).json({
        success: false,
        error: {
          message: 'Access denied. Insufficient permissions.',
          statusCode: 403
        }
      });
    }

    next();
  };
};

// Middleware for optional authentication
const optionalAuth = (req, res, next) => {
  try {
    const token = extractToken(req);
    
    if (token) {
      const decoded = verifyToken(token);
      req.user = decoded;
      logger.auth('Optional auth - Token found and valid', { userId: decoded.id });
    } else {
      logger.auth('Optional auth - No token provided');
    }
    
    next();
    
  } catch (error) {
    logger.auth('Optional auth - Invalid token', { error: error.message });
    // Continue without authentication for optional auth
    next();
  }
};

module.exports = {
  generateToken,
  generateRefreshToken,
  verifyToken,
  verifyRefreshToken,
  extractToken,
  authenticate,
  authorize,
  optionalAuth
};