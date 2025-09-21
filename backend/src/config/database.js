const { Sequelize } = require('sequelize');
const config = require('./config');
const logger = require('../utils/logger');

let sequelize = null;

const connectDB = async () => {
  try {
    // Force PostgreSQL connection in development if POSTGRESQL_URI is set
    if (config.postgresqlUri && config.postgresqlUri.includes('supabase.co')) {
      logger.info('🔄 Attempting Supabase PostgreSQL connection...');
      
      // Use PostgreSQL configuration with proper SSL for Supabase
      sequelize = new Sequelize(config.postgresqlUri, {
        dialect: 'postgres',
        logging: config.nodeEnv === 'development' ? (msg) => logger.db(msg) : false,
        pool: {
          max: 10,
          min: 0,
          acquire: 30000,
          idle: 10000,
        },
        define: {
          timestamps: true,
          underscored: true,
          freezeTableName: true,
        },
        dialectOptions: {
          ssl: {
            require: false,
            rejectUnauthorized: false, // For Supabase - ignore self-signed cert issues
          },
        },
      });

      // Test the connection
      await sequelize.authenticate();
      logger.info(`✅ Supabase PostgreSQL connected successfully`);

      // Sync database in development (create tables)
      if (config.nodeEnv === 'development') {
        await sequelize.sync({ alter: true });
        logger.info('✅ Database synchronized');
      }
      return;
    }
    
    // Check if PostgreSQL is configured, otherwise use SQLite for development
    const hasPostgres = config.postgresqlUri || (config.postgresqlHost && config.postgresqlDatabase);
    
    if (!hasPostgres && config.nodeEnv === 'development') {
      // Use SQLite for development when PostgreSQL is not available
      sequelize = new Sequelize({
        dialect: 'sqlite',
        storage: 'database/development.sqlite',
        logging: (msg) => logger.db(msg),
        define: {
          timestamps: true,
          underscored: true,
          freezeTableName: true,
        },
      });
      
      await sequelize.authenticate();
      logger.info('✅ SQLite database connected (development fallback)');
      
      // Sync database (create tables)
      await sequelize.sync({ alter: true });
      logger.info('✅ Database synchronized');
      
      return;
    }
    
    // Use PostgreSQL configuration
    const dbUri = config.nodeEnv === 'test' ? config.postgresqlTestUri : config.postgresqlUri;
    
    // Create Sequelize instance
    sequelize = new Sequelize(dbUri, {
      dialect: 'postgres',
      host: config.postgresqlHost,
      port: config.postgresqlPort,
      username: config.postgresqlUsername,
      password: config.postgresqlPassword,
      database: config.postgresqlDatabase,
      logging: config.nodeEnv === 'development' ? (msg) => logger.db(msg) : false,
      pool: {
        max: 10,
        min: 0,
        acquire: 30000,
        idle: 10000,
      },
      define: {
        timestamps: true,
        underscored: true,
        freezeTableName: true,
      },
      dialectOptions: {
        // For SSL connection if needed
        ...(config.nodeEnv === 'production' && {
          ssl: {
            require: true,
            rejectUnauthorized: false,
          },
        }),
      },
    });

    // Test the connection
    await sequelize.authenticate();
    logger.info(`✅ PostgreSQL connected: ${config.postgresqlHost}:${config.postgresqlPort}/${config.postgresqlDatabase}`);

    // Sync database in development (create tables)
    if (config.nodeEnv === 'development') {
      await sequelize.sync({ alter: true });
      logger.info('✅ Database synchronized');
    }

  } catch (error) {
    logger.error('❌ Database connection failed:', error.message);
    logger.error('📋 Connection details:', {
      host: config.postgresqlHost,
      port: config.postgresqlPort,
      database: config.postgresqlDatabase,
      username: config.postgresqlUsername,
      uri: config.postgresqlUri ? config.postgresqlUri.replace(/:[^:]*@/, ':***@') : 'not set'
    });
    
    if (config.nodeEnv === 'development') {
      logger.warn('⚠️ Attempting SQLite fallback for development...');
      
      try {
        // Fallback to SQLite for development
        sequelize = new Sequelize({
          dialect: 'sqlite',
          storage: 'database/development.sqlite',
          logging: (msg) => logger.db(msg),
          define: {
            timestamps: true,
            underscored: true,
            freezeTableName: true,
          },
        });
        
        await sequelize.authenticate();
        await sequelize.sync({ alter: true });
        logger.info('✅ SQLite fallback database connected and synchronized');
      } catch (sqliteError) {
        logger.error('❌ SQLite fallback also failed:', sqliteError.message);
        logger.warn('⚠️ Application will continue without database connection');
      }
    } else {
      logger.warn('⚠️ Application will continue without database connection for development');
      
      // Exit in production
      if (config.nodeEnv === 'production') {
        process.exit(1);
      }
    }
  }
};

// Get Sequelize instance
const getSequelize = () => {
  return sequelize;
};

// Close database connection
const closeDB = async () => {
  if (sequelize) {
    try {
      await sequelize.close();
      logger.info('PostgreSQL connection closed');
    } catch (error) {
      logger.error('Error closing PostgreSQL connection:', error);
    }
  }
};

// Test database connection
const testConnection = async () => {
  try {
    if (sequelize) {
      await sequelize.authenticate();
      return true;
    }
    return false;
  } catch (error) {
    logger.error('Database connection test failed:', error);
    return false;
  }
};

// Graceful shutdown
process.on('SIGINT', async () => {
  try {
    await closeDB();
    logger.info('PostgreSQL connection closed through app termination');
    process.exit(0);
  } catch (error) {
    logger.error('Error during PostgreSQL shutdown:', error);
    process.exit(1);
  }
});

process.on('SIGTERM', async () => {
  try {
    await closeDB();
    logger.info('PostgreSQL connection closed through SIGTERM');
    process.exit(0);
  } catch (error) {
    logger.error('Error during PostgreSQL shutdown:', error);
    process.exit(1);
  }
});

module.exports = {
  connectDB,
  getSequelize,
  closeDB,
  testConnection,
};