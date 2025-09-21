module.exports = {
  // Server configuration
  port: process.env.PORT || 3000,
  nodeEnv: process.env.NODE_ENV || 'development',
  
  // Database configuration
  postgresqlUri: process.env.POSTGRESQL_URI || 'postgresql://localhost:5432/nasa-air-quality',
  postgresqlTestUri: process.env.POSTGRESQL_TEST_URI || 'postgresql://localhost:5432/nasa-air-quality-test',
  postgresqlHost: process.env.POSTGRESQL_HOST || 'localhost',
  postgresqlPort: process.env.POSTGRESQL_PORT || 5432,
  postgresqlDatabase: process.env.POSTGRESQL_DATABASE || 'nasa-air-quality',
  postgresqlUsername: process.env.POSTGRESQL_USERNAME || 'postgres',
  postgresqlPassword: process.env.POSTGRESQL_PASSWORD || '',
  
  // Redis configuration
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
  redisPassword: process.env.REDIS_PASSWORD || '',
  
  // JWT configuration
  jwtSecret: process.env.JWT_SECRET || 'fallback-secret-change-in-production',
  jwtExpire: process.env.JWT_EXPIRE || '7d',
  jwtRefreshSecret: process.env.JWT_REFRESH_SECRET || 'fallback-refresh-secret',
  jwtRefreshExpire: process.env.JWT_REFRESH_EXPIRE || '30d',
  
  // NASA TEMPO API configuration
  tempoApiBaseUrl: process.env.TEMPO_API_BASE_URL || 'https://tempo.si.edu/api',
  tempoApiKey: process.env.TEMPO_API_KEY || '',
  tempoDataRefreshInterval: process.env.TEMPO_DATA_REFRESH_INTERVAL || 300000,
  
  // EPA AirNow API configuration
  airnowApiBaseUrl: process.env.AIRNOW_API_BASE_URL || 'https://www.airnowapi.org',
  airnowApiKey: process.env.AIRNOW_API_KEY || '',
  
  // Email configuration
  smtpHost: process.env.SMTP_HOST || 'smtp.gmail.com',
  smtpPort: process.env.SMTP_PORT || 587,
  smtpUser: process.env.SMTP_USER || '',
  smtpPass: process.env.SMTP_PASS || '',
  
  // Rate limiting
  rateLimitWindowMs: process.env.RATE_LIMIT_WINDOW_MS || 900000, // 15 minutes
  rateLimitMaxRequests: process.env.RATE_LIMIT_MAX_REQUESTS || 100,
  
  // Logging
  logLevel: process.env.LOG_LEVEL || 'info',
  logFile: process.env.LOG_FILE || 'logs/app.log',
  
  // CORS
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:3001',
  
  // Application settings
  appName: process.env.APP_NAME || 'NASA Air Quality Forecasting',
  appVersion: process.env.APP_VERSION || '1.0.0',
  
  // Cache settings
  cacheDefaultTTL: 300, // 5 minutes
  cachePredictionTTL: 1800, // 30 minutes
  cacheHistoricalTTL: 3600, // 1 hour
  
  // Pagination defaults
  defaultPageSize: 20,
  maxPageSize: 100,
  
  // File upload limits
  maxFileSize: 10 * 1024 * 1024, // 10MB
  
  // API response time targets
  apiResponseTimeTarget: 200, // milliseconds
  dbQueryTimeTarget: 50, // milliseconds
};