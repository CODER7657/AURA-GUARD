require('dotenv').config();
const { connectDB, getSequelize } = require('./src/config/database');

async function initializeDatabase() {
  try {
    console.log('🚀 Starting database initialization...');
    
    // Connect to database
    await connectDB();
    const sequelize = getSequelize();
    
    if (!sequelize) {
      throw new Error('Database connection not established');
    }
    
    console.log('✅ Database connected, loading models...');
    
    // Load all models
    require('./src/models');
    
    console.log('📋 Models loaded, creating tables...');
    
    // Force sync to create tables (be careful with this in production!)
    await sequelize.sync({ force: true });
    
    console.log('✅ Tables created successfully!');
    
    console.log('🌱 Running seeder...');
    
    // Load and run the seeder
    const seeder = require('./database/seeders/20250921070520-demo-data.js');
    
    // Execute the up function
    await seeder.up(sequelize.getQueryInterface(), sequelize.constructor);
    
    console.log('✅ Database initialized and seeded successfully!');
    
    // Close connection
    await sequelize.close();
    process.exit(0);
    
  } catch (error) {
    console.error('❌ Database initialization failed:', error.message);
    console.error(error);
    process.exit(1);
  }
}

initializeDatabase();