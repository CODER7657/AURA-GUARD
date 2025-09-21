const { getSequelize } = require('../config/database');
const User = require('./User');
const AirQualityData = require('./AirQualityData');
const Notification = require('./Notification');
const Prediction = require('./Prediction');

let models = {};

const initializeModels = () => {
  const sequelize = getSequelize();
  
  if (!sequelize) {
    console.warn('⚠️ Sequelize instance not available, models not initialized');
    return models;
  }

  // Initialize models
  models.User = User(sequelize);
  models.AirQualityData = AirQualityData(sequelize);
  models.Notification = Notification(sequelize);
  models.Prediction = Prediction(sequelize);

  // Define associations
  
  // User associations
  models.User.hasMany(models.Notification, { 
    foreignKey: 'userId', 
    as: 'notifications' 
  });
  
  // Notification associations
  models.Notification.belongsTo(models.User, { 
    foreignKey: 'userId', 
    as: 'user' 
  });
  
  // Air Quality Data associations (if needed for user bookmarks, etc.)
  // models.User.hasMany(models.AirQualityData, { foreignKey: 'userId', as: 'bookmarkedData' });
  
  // Store sequelize instance
  models.sequelize = sequelize;
  models.Sequelize = require('sequelize');

  console.log('✅ Models initialized successfully');
  return models;
};

// Initialize models when this module is loaded
if (getSequelize()) {
  initializeModels();
}

module.exports = {
  ...models,
  initializeModels,
};