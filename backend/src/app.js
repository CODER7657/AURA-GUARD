const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const passport = require('passport');
require('dotenv').config();

const config = require('./config/config');
const { connectDB } = require('./config/database');
const { connectRedis } = require('./config/redis');
const logger = require('./utils/logger');
const errorHandler = require('./middleware/errorHandler');
const notFound = require('./middleware/notFound');

// Import routes
const authRoutes = require('./routes/auth');
const airQualityRoutes = require('./routes/airQuality');
const predictionRoutes = require('./routes/predictions');
const notificationRoutes = require('./routes/notifications');
const healthRoutes = require('./routes/health');

const app = express();

// Database connections
connectDB();
connectRedis();

// Security middleware
app.use(helmet());
app.use(cors({
  origin: config.corsOrigin,
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: config.rateLimitWindowMs,
  max: config.rateLimitMaxRequests,
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// General middleware
app.use(compression());
app.use(morgan('combined', { stream: { write: message => logger.info(message.trim()) } }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Passport initialization
app.use(passport.initialize());
require('./config/passport')(passport);

// Health check endpoint
app.use('/health', healthRoutes);

// API routes
app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/air-quality', airQualityRoutes);
app.use('/api/v1/predictions', predictionRoutes);
app.use('/api/v1/notifications', notificationRoutes);

// API documentation
app.get('/api/docs', (req, res) => {
  res.json({
    message: 'NASA Air Quality Forecasting API',
    version: '1.0.0',
    endpoints: {
      auth: '/api/v1/auth',
      airQuality: '/api/v1/air-quality',
      predictions: '/api/v1/predictions',
      notifications: '/api/v1/notifications',
      health: '/health'
    },
    documentation: 'https://api-docs.nasa-air-quality.com'
  });
});

// 404 handler
app.use(notFound);

// Error handling middleware
app.use(errorHandler);

const PORT = config.port || 3000;

const server = app.listen(PORT, () => {
  logger.info(`🚀 NASA Air Quality Backend Server running on port ${PORT}`);
  logger.info(`📊 Environment: ${config.nodeEnv}`);
  logger.info(`💾 Database: ${config.mongodbUri ? 'Connected' : 'Not configured'}`);
  logger.info(`🔄 Redis: ${config.redisUrl ? 'Connected' : 'Not configured'}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received. Shutting down gracefully...');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

process.on('unhandledRejection', (err) => {
  logger.error('Unhandled Promise Rejection:', err);
  server.close(() => {
    process.exit(1);
  });
});

module.exports = app;