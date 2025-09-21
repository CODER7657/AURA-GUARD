const express = require('express');
const Joi = require('joi');
const AuthService = require('../services/AuthService');
const { authenticateToken } = require('../middleware/auth');
const logger = require('../utils/logger');

const router = express.Router();

// Validation schemas
const registerSchema = Joi.object({
  name: Joi.string().min(2).max(100).required(),
  email: Joi.string().email().required(),
  password: Joi.string().min(8).max(128).pattern(new RegExp('^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]')).required()
    .messages({
      'string.pattern.base': 'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'
    })
});

const loginSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().required()
});

const refreshTokenSchema = Joi.object({
  refreshToken: Joi.string().required()
});

const changePasswordSchema = Joi.object({
  currentPassword: Joi.string().required(),
  newPassword: Joi.string().min(8).max(128).pattern(new RegExp('^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]')).required()
    .messages({
      'string.pattern.base': 'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'
    })
});

const updateProfileSchema = Joi.object({
  name: Joi.string().min(2).max(100),
  preferences: Joi.object({
    notifications: Joi.boolean(),
    alert_threshold: Joi.string().valid('good', 'moderate', 'unhealthy_sensitive', 'unhealthy', 'very_unhealthy'),
    preferred_units: Joi.string().valid('metric', 'imperial'),
    locations: Joi.array().items(Joi.object({
      name: Joi.string().required(),
      lat: Joi.number().min(-90).max(90).required(),
      lon: Joi.number().min(-180).max(180).required()
    }))
  })
});

/**
 * POST /auth/register
 * Register a new user
 */
router.post('/register', async (req, res) => {
  try {
    // Validate request body
    const { error, value } = registerSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const { name, email, password } = value;

    // Register user
    const result = await AuthService.register({ name, email, password });

    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      data: result
    });
  } catch (error) {
    logger.error('Registration route error:', error);
    
    if (error.message === 'User already exists with this email') {
      return res.status(409).json({
        success: false,
        message: error.message,
        error: 'USER_EXISTS'
      });
    }

    res.status(500).json({
      success: false,
      message: 'Registration failed',
      error: 'REGISTRATION_ERROR'
    });
  }
});

/**
 * POST /auth/login
 * Login user
 */
router.post('/login', async (req, res) => {
  try {
    // Validate request body
    const { error, value } = loginSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const { email, password } = value;

    // Login user
    const result = await AuthService.login(email, password);

    res.json({
      success: true,
      message: 'Login successful',
      data: result
    });
  } catch (error) {
    logger.error('Login route error:', error);
    
    if (error.message === 'Invalid email or password' || error.message === 'Account is deactivated') {
      return res.status(401).json({
        success: false,
        message: error.message,
        error: 'INVALID_CREDENTIALS'
      });
    }

    res.status(500).json({
      success: false,
      message: 'Login failed',
      error: 'LOGIN_ERROR'
    });
  }
});

/**
 * POST /auth/refresh
 * Refresh access token
 */
router.post('/refresh', async (req, res) => {
  try {
    // Validate request body
    const { error, value } = refreshTokenSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const { refreshToken } = value;

    // Refresh token
    const result = await AuthService.refreshToken(refreshToken);

    res.json({
      success: true,
      message: 'Token refreshed successfully',
      data: result
    });
  } catch (error) {
    logger.error('Token refresh route error:', error);
    
    res.status(401).json({
      success: false,
      message: 'Invalid refresh token',
      error: 'INVALID_REFRESH_TOKEN'
    });
  }
});

/**
 * GET /auth/profile
 * Get user profile (protected)
 */
router.get('/profile', authenticateToken, async (req, res) => {
  try {
    const user = await AuthService.getProfile(req.user.id);

    res.json({
      success: true,
      message: 'Profile retrieved successfully',
      data: { user }
    });
  } catch (error) {
    logger.error('Get profile route error:', error);
    
    res.status(500).json({
      success: false,
      message: 'Failed to get profile',
      error: 'PROFILE_ERROR'
    });
  }
});

/**
 * PUT /auth/profile
 * Update user profile (protected)
 */
router.put('/profile', authenticateToken, async (req, res) => {
  try {
    // Validate request body
    const { error, value } = updateProfileSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    // Update profile
    const user = await AuthService.updateProfile(req.user.id, value);

    res.json({
      success: true,
      message: 'Profile updated successfully',
      data: { user }
    });
  } catch (error) {
    logger.error('Update profile route error:', error);
    
    res.status(500).json({
      success: false,
      message: 'Failed to update profile',
      error: 'UPDATE_PROFILE_ERROR'
    });
  }
});

/**
 * POST /auth/change-password
 * Change user password (protected)
 */
router.post('/change-password', authenticateToken, async (req, res) => {
  try {
    // Validate request body
    const { error, value } = changePasswordSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const { currentPassword, newPassword } = value;

    // Change password
    const result = await AuthService.changePassword(req.user.id, currentPassword, newPassword);

    res.json({
      success: true,
      message: result.message
    });
  } catch (error) {
    logger.error('Change password route error:', error);
    
    if (error.message === 'Current password is incorrect') {
      return res.status(400).json({
        success: false,
        message: error.message,
        error: 'INVALID_CURRENT_PASSWORD'
      });
    }

    res.status(500).json({
      success: false,
      message: 'Failed to change password',
      error: 'PASSWORD_CHANGE_ERROR'
    });
  }
});

/**
 * POST /auth/logout
 * Logout user (protected)
 * Note: In a stateless JWT system, logout is handled client-side by removing the token
 */
router.post('/logout', authenticateToken, async (req, res) => {
  try {
    logger.info(`User logged out: ${req.user.email}`);
    
    res.json({
      success: true,
      message: 'Logout successful'
    });
  } catch (error) {
    logger.error('Logout route error:', error);
    
    res.status(500).json({
      success: false,
      message: 'Logout failed',
      error: 'LOGOUT_ERROR'
    });
  }
});

/**
 * GET /auth/verify
 * Verify token validity (protected)
 */
router.get('/verify', authenticateToken, async (req, res) => {
  try {
    res.json({
      success: true,
      message: 'Token is valid',
      data: {
        user: {
          id: req.user.id,
          name: req.user.name,
          email: req.user.email,
          role: req.user.role
        }
      }
    });
  } catch (error) {
    logger.error('Token verify route error:', error);
    
    res.status(500).json({
      success: false,
      message: 'Token verification failed',
      error: 'VERIFY_ERROR'
    });
  }
});

module.exports = router;