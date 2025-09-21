require('dotenv').config();

module.exports = {
  development: {
    username: process.env.POSTGRESQL_USERNAME || 'postgres',
    password: process.env.POSTGRESQL_PASSWORD || '',
    database: process.env.POSTGRESQL_DATABASE || 'nasa-air-quality',
    host: process.env.POSTGRESQL_HOST || 'localhost',
    port: process.env.POSTGRESQL_PORT || 5432,
    dialect: 'postgres',
    logging: console.log,
    define: {
      timestamps: true,
      underscored: true,
      freezeTableName: true,
    },
  },
  test: {
    username: process.env.POSTGRESQL_USERNAME || 'postgres',
    password: process.env.POSTGRESQL_PASSWORD || '',
    database: process.env.POSTGRESQL_DATABASE || 'nasa-air-quality-test',
    host: process.env.POSTGRESQL_HOST || 'localhost',
    port: process.env.POSTGRESQL_PORT || 5432,
    dialect: 'postgres',
    logging: false,
    define: {
      timestamps: true,
      underscored: true,
      freezeTableName: true,
    },
  },
  production: {
    use_env_variable: 'POSTGRESQL_URI',
    dialect: 'postgres',
    dialectOptions: {
      ssl: {
        require: true,
        rejectUnauthorized: false,
      },
    },
    define: {
      timestamps: true,
      underscored: true,
      freezeTableName: true,
    },
    logging: false,
  },
};