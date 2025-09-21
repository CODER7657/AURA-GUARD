const winston = require('winston');
const path = require('path');
const config = require('../config/config');

// Create logs directory if it doesn't exist
const fs = require('fs');
const logDir = path.dirname(config.logFile);
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}

// Define log format
const logFormat = winston.format.combine(
  winston.format.timestamp({
    format: 'YYYY-MM-DD HH:mm:ss'
  }),
  winston.format.errors({ stack: true }),
  winston.format.json()
);

// Define console format for development
const consoleFormat = winston.format.combine(
  winston.format.colorize(),
  winston.format.timestamp({
    format: 'HH:mm:ss'
  }),
  winston.format.printf(({ timestamp, level, message, stack }) => {
    if (stack) {
      return `${timestamp} ${level}: ${message}\n${stack}`;
    }
    return `${timestamp} ${level}: ${message}`;
  })
);

// Create logger instance
const logger = winston.createLogger({
  level: config.logLevel,
  format: logFormat,
  defaultMeta: { 
    service: config.appName,
    version: config.appVersion 
  },
  transports: [
    // File transport for all logs
    new winston.transports.File({
      filename: config.logFile,
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
    
    // Separate file for errors
    new winston.transports.File({
      filename: path.join(logDir, 'error.log'),
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    })
  ]
});

// Add console transport for development
if (config.nodeEnv !== 'production') {
  logger.add(new winston.transports.Console({
    format: consoleFormat
  }));
}

// Create request logger for morgan
logger.stream = {
  write: (message) => {
    logger.info(message.trim());
  }
};

// Add custom methods for better categorization
logger.api = (message, meta = {}) => {
  logger.info(message, { category: 'API', ...meta });
};

logger.db = (message, meta = {}) => {
  logger.info(message, { category: 'DATABASE', ...meta });
};

logger.cache = (message, meta = {}) => {
  logger.info(message, { category: 'CACHE', ...meta });
};

logger.auth = (message, meta = {}) => {
  logger.info(message, { category: 'AUTH', ...meta });
};

logger.external = (message, meta = {}) => {
  logger.info(message, { category: 'EXTERNAL_API', ...meta });
};

logger.performance = (message, meta = {}) => {
  logger.info(message, { category: 'PERFORMANCE', ...meta });
};

module.exports = logger;