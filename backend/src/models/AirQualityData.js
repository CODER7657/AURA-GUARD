const { DataTypes } = require('sequelize');

const AirQualityData = (sequelize) => {
  const AirQualityDataModel = sequelize.define('AirQualityData', {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    stationId: {
      type: DataTypes.STRING(100),
      allowNull: false,
      validate: {
        notEmpty: true,
      },
    },
    stationName: {
      type: DataTypes.STRING(255),
      allowNull: true,
    },
    latitude: {
      type: DataTypes.DECIMAL(10, 8),
      allowNull: false,
      validate: {
        min: -90,
        max: 90,
      },
    },
    longitude: {
      type: DataTypes.DECIMAL(11, 8),
      allowNull: false,
      validate: {
        min: -180,
        max: 180,
      },
    },
    measurementTime: {
      type: DataTypes.DATE,
      allowNull: false,
    },
    dataSource: {
      type: DataTypes.ENUM('TEMPO', 'EPA_AIRNOW', 'MANUAL', 'OTHER'),
      allowNull: false,
    },
    // Air Quality Parameters
    pm25: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    pm10: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    ozone: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    no2: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    so2: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    co: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    // Air Quality Index
    aqi: {
      type: DataTypes.INTEGER,
      allowNull: true,
      validate: {
        min: 0,
        max: 500,
      },
    },
    aqiCategory: {
      type: DataTypes.ENUM(
        'good',
        'moderate', 
        'unhealthy_sensitive',
        'unhealthy',
        'very_unhealthy',
        'hazardous'
      ),
      allowNull: true,
    },
    // Weather data (if available)
    temperature: {
      type: DataTypes.DECIMAL(5, 2),
      allowNull: true,
    },
    humidity: {
      type: DataTypes.DECIMAL(5, 2),
      allowNull: true,
      validate: {
        min: 0,
        max: 100,
      },
    },
    windSpeed: {
      type: DataTypes.DECIMAL(5, 2),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    windDirection: {
      type: DataTypes.DECIMAL(5, 2),
      allowNull: true,
      validate: {
        min: 0,
        max: 360,
      },
    },
    pressure: {
      type: DataTypes.DECIMAL(8, 2),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    // Data quality indicators
    dataQuality: {
      type: DataTypes.ENUM('excellent', 'good', 'fair', 'poor'),
      defaultValue: 'good',
      allowNull: false,
    },
    validationStatus: {
      type: DataTypes.ENUM('pending', 'validated', 'rejected'),
      defaultValue: 'pending',
      allowNull: false,
    },
    // Metadata
    rawData: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    metadata: {
      type: DataTypes.JSONB,
      defaultValue: {},
      allowNull: true,
    },
  }, {
    tableName: 'air_quality_data',
    timestamps: true,
    indexes: [
      {
        fields: ['station_id'],
      },
      {
        fields: ['measurement_time'],
      },
      {
        fields: ['data_source'],
      },
      {
        fields: ['latitude', 'longitude'],
      },
      {
        fields: ['aqi'],
      },
      {
        fields: ['aqi_category'],
      },
      {
        fields: ['validation_status'],
      },
      {
        fields: ['measurement_time', 'station_id'],
        unique: true,
      },
    ],
  });

  // Class methods
  AirQualityDataModel.findByLocation = function(lat, lon, radius = 10) {
    // Find data within radius (km) of given coordinates
    // This is a simplified version - in production, use PostGIS for proper spatial queries
    const latDiff = radius / 111; // Approximate degrees per km
    const lonDiff = radius / (111 * Math.cos(lat * Math.PI / 180));
    
    return this.findAll({
      where: {
        latitude: {
          [sequelize.Sequelize.Op.between]: [lat - latDiff, lat + latDiff],
        },
        longitude: {
          [sequelize.Sequelize.Op.between]: [lon - lonDiff, lon + lonDiff],
        },
      },
      order: [['measurementTime', 'DESC']],
    });
  };

  AirQualityDataModel.findByTimeRange = function(startDate, endDate, stationId = null) {
    const whereClause = {
      measurementTime: {
        [sequelize.Sequelize.Op.between]: [startDate, endDate],
      },
    };
    
    if (stationId) {
      whereClause.stationId = stationId;
    }
    
    return this.findAll({
      where: whereClause,
      order: [['measurementTime', 'DESC']],
    });
  };

  AirQualityDataModel.getLatestByStation = function(stationId) {
    return this.findOne({
      where: { stationId },
      order: [['measurementTime', 'DESC']],
    });
  };

  return AirQualityDataModel;
};

module.exports = AirQualityData;