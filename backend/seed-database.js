require('dotenv').config();
const { connectDB, getSequelize } = require('./src/config/database');

async function seedDatabase() {
  try {
    console.log('🌱 Starting database seeding...');
    
    // Connect to database
    await connectDB();
    const sequelize = getSequelize();
    
    if (!sequelize) {
      throw new Error('Database connection not established');
    }
    
    console.log('✅ Database connected, running seeder...');
    
    // Load and run the seeder
    const seeder = require('./database/seeders/20250921070520-demo-data.js');
    
    // Execute the up function
    await seeder.up(sequelize.getQueryInterface(), sequelize.constructor);
    
    console.log('✅ Database seeded successfully!');
    
    // Close connection
    await sequelize.close();
    process.exit(0);
    
  } catch (error) {
    console.error('❌ Database seeding failed:', error.message);
    console.error(error);
    process.exit(1);
  }
}

seedDatabase();