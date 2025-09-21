const { Sequelize } = require('sequelize');
const logger = require('../utils/logger');

let sequelize;

// Check if we're in development and use SQLite for quick setup
if (process.env.NODE_ENV === 'development' && !process.env.POSTGRESQL_URI) {
  // Use SQLite for development if PostgreSQL is not configured
  sequelize = new Sequelize({
    dialect: 'sqlite',
    storage: 'database/development.sqlite',
    logging: (msg) => logger.debug(msg),
    define: {
      timestamps: true,
      underscored: true,
    },
  });
  
  logger.info('📊 Using SQLite database for development');
} else {
  // Use PostgreSQL for production or when specifically configured
  const dbUrl = process.env.POSTGRESQL_URI || 
    `postgresql://${process.env.POSTGRESQL_USERNAME}:${process.env.POSTGRESQL_PASSWORD}@${process.env.POSTGRESQL_HOST}:${process.env.POSTGRESQL_PORT}/${process.env.POSTGRESQL_DATABASE}`;

  sequelize = new Sequelize(dbUrl, {
    dialect: 'postgres',
    logging: process.env.NODE_ENV === 'development' ? 
      (msg) => logger.debug(msg) : false,
    pool: {
      max: 5,
      min: 0,
      acquire: 30000,
      idle: 10000,
    },
    define: {
      timestamps: true,
      underscored: true,
    },
    dialectOptions: {
      ssl: process.env.NODE_ENV === 'production' ? {
        require: true,
        rejectUnauthorized: false,
      } : false,
    },
  });
}

// Test the connection
async function testConnection() {
  try {
    await sequelize.authenticate();
    logger.info('✅ Database connection established successfully');
    return true;
  } catch (error) {
    logger.error('❌ Unable to connect to database:', error.message);
    return false;
  }
}

// Graceful shutdown
async function closeConnection() {
  try {
    await sequelize.close();
    logger.info('Database connection closed');
  } catch (error) {
    logger.error('Error closing database connection:', error);
  }
}

module.exports = {
  sequelize,
  testConnection,
  closeConnection,
};