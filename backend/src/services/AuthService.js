const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { User } = require('../models');
const logger = require('../utils/logger');

/**
 * Authentication Service
 * Handles user authentication, registration, and token management
 */
class AuthService {
  /**
   * Generate JWT access token
   */
  generateAccessToken(user) {
    const payload = {
      userId: user.id,
      email: user.email,
      role: user.role
    };

    return jwt.sign(payload, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRES_IN || '24h',
      issuer: 'nasa-air-quality-api',
      audience: 'nasa-air-quality-client'
    });
  }

  /**
   * Generate JWT refresh token
   */
  generateRefreshToken(user) {
    const payload = {
      userId: user.id,
      tokenType: 'refresh'
    };

    return jwt.sign(payload, process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '7d',
      issuer: 'nasa-air-quality-api',
      audience: 'nasa-air-quality-client'
    });
  }

  /**
   * Register a new user
   */
  async register(userData) {
    try {
      const { name, email, password } = userData;

      // Check if user already exists
      const existingUser = await User.findOne({ where: { email } });
      if (existingUser) {
        throw new Error('User already exists with this email');
      }

      // Hash password
      const hashedPassword = await bcrypt.hash(password, 12);

      // Create user
      const user = await User.create({
        name,
        email,
        password: hashedPassword,
        role: 'user',
        is_active: true,
        is_email_verified: false,
        preferences: {
          notifications: true,
          alert_threshold: 'moderate',
          preferred_units: 'metric'
        }
      });

      // Remove password from response
      const userResponse = user.toJSON();
      delete userResponse.password;

      // Generate tokens
      const accessToken = this.generateAccessToken(user);
      const refreshToken = this.generateRefreshToken(user);

      logger.info(`New user registered: ${email}`);

      return {
        user: userResponse,
        accessToken,
        refreshToken
      };
    } catch (error) {
      logger.error('Registration error:', error);
      throw error;
    }
  }

  /**
   * Login user
   */
  async login(email, password) {
    try {
      // Find user
      const user = await User.findOne({ where: { email } });
      if (!user) {
        throw new Error('Invalid email or password');
      }

      // Check if account is active
      if (!user.is_active) {
        throw new Error('Account is deactivated');
      }

      // Verify password
      const isValidPassword = await bcrypt.compare(password, user.password);
      if (!isValidPassword) {
        throw new Error('Invalid email or password');
      }

      // Update last login
      await user.update({ last_login_at: new Date() });

      // Remove password from response
      const userResponse = user.toJSON();
      delete userResponse.password;

      // Generate tokens
      const accessToken = this.generateAccessToken(user);
      const refreshToken = this.generateRefreshToken(user);

      logger.info(`User logged in: ${email}`);

      return {
        user: userResponse,
        accessToken,
        refreshToken
      };
    } catch (error) {
      logger.error('Login error:', error);
      throw error;
    }
  }

  /**
   * Refresh access token
   */
  async refreshToken(refreshToken) {
    try {
      // Verify refresh token
      const decoded = jwt.verify(
        refreshToken, 
        process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET
      );

      if (decoded.tokenType !== 'refresh') {
        throw new Error('Invalid refresh token');
      }

      // Find user
      const user = await User.findByPk(decoded.userId);
      if (!user || !user.is_active) {
        throw new Error('User not found or inactive');
      }

      // Generate new access token
      const accessToken = this.generateAccessToken(user);

      logger.info(`Token refreshed for user: ${user.email}`);

      return { accessToken };
    } catch (error) {
      logger.error('Token refresh error:', error);
      throw error;
    }
  }

  /**
   * Change user password
   */
  async changePassword(userId, currentPassword, newPassword) {
    try {
      const user = await User.findByPk(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Verify current password
      const isValidPassword = await bcrypt.compare(currentPassword, user.password);
      if (!isValidPassword) {
        throw new Error('Current password is incorrect');
      }

      // Hash new password
      const hashedPassword = await bcrypt.hash(newPassword, 12);

      // Update password
      await user.update({ password: hashedPassword });

      logger.info(`Password changed for user: ${user.email}`);

      return { message: 'Password changed successfully' };
    } catch (error) {
      logger.error('Password change error:', error);
      throw error;
    }
  }

  /**
   * Verify JWT token
   */
  async verifyToken(token) {
    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      const user = await User.findByPk(decoded.userId, {
        attributes: { exclude: ['password'] }
      });

      if (!user || !user.is_active) {
        throw new Error('User not found or inactive');
      }

      return { user, decoded };
    } catch (error) {
      logger.error('Token verification error:', error);
      throw error;
    }
  }

  /**
   * Get user profile
   */
  async getProfile(userId) {
    try {
      const user = await User.findByPk(userId, {
        attributes: { exclude: ['password'] }
      });

      if (!user) {
        throw new Error('User not found');
      }

      return user;
    } catch (error) {
      logger.error('Get profile error:', error);
      throw error;
    }
  }

  /**
   * Update user profile
   */
  async updateProfile(userId, updateData) {
    try {
      const user = await User.findByPk(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Only allow certain fields to be updated
      const allowedFields = ['name', 'preferences'];
      const filteredData = {};
      
      allowedFields.forEach(field => {
        if (updateData[field] !== undefined) {
          filteredData[field] = updateData[field];
        }
      });

      await user.update(filteredData);

      // Return user without password
      const updatedUser = await User.findByPk(userId, {
        attributes: { exclude: ['password'] }
      });

      logger.info(`Profile updated for user: ${user.email}`);

      return updatedUser;
    } catch (error) {
      logger.error('Update profile error:', error);
      throw error;
    }
  }
}

module.exports = new AuthService();