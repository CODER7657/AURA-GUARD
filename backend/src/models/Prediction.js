const { DataTypes } = require('sequelize');

const Prediction = (sequelize) => {
  const PredictionModel = sequelize.define('Prediction', {
    id: {
      type: DataTypes.UUID,
      defaultValue: DataTypes.UUIDV4,
      primaryKey: true,
    },
    predictionId: {
      type: DataTypes.STRING(100),
      allowNull: false,
      unique: true,
    },
    modelName: {
      type: DataTypes.STRING(100),
      allowNull: false,
      validate: {
        notEmpty: true,
      },
    },
    modelVersion: {
      type: DataTypes.STRING(50),
      allowNull: false,
      validate: {
        notEmpty: true,
      },
    },
    predictionType: {
      type: DataTypes.ENUM('realtime', 'hourly_forecast', 'daily_forecast', 'weekly_forecast'),
      allowNull: false,
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
    predictionTime: {
      type: DataTypes.DATE,
      allowNull: false,
    },
    forecastTime: {
      type: DataTypes.DATE,
      allowNull: false,
    },
    // Predicted air quality parameters
    predictedPm25: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    predictedPm10: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    predictedOzone: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    predictedNo2: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    predictedSo2: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    predictedCo: {
      type: DataTypes.DECIMAL(8, 3),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    predictedAqi: {
      type: DataTypes.INTEGER,
      allowNull: true,
      validate: {
        min: 0,
        max: 500,
      },
    },
    predictedCategory: {
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
    // Confidence metrics
    confidenceScore: {
      type: DataTypes.DECIMAL(3, 2),
      allowNull: true,
      validate: {
        min: 0,
        max: 1,
      },
    },
    confidenceIntervalLower: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    confidenceIntervalUpper: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    // Input features used for prediction
    inputFeatures: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    // Weather data used
    weatherData: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    // Historical data context
    historicalContext: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    // Model performance metrics
    modelAccuracy: {
      type: DataTypes.DECIMAL(5, 4),
      allowNull: true,
      validate: {
        min: 0,
        max: 1,
      },
    },
    modelRmse: {
      type: DataTypes.DECIMAL(8, 4),
      allowNull: true,
      validate: {
        min: 0,
      },
    },
    // Status and validation
    status: {
      type: DataTypes.ENUM('pending', 'completed', 'failed', 'validated'),
      defaultValue: 'pending',
      allowNull: false,
    },
    validationStatus: {
      type: DataTypes.ENUM('pending', 'validated', 'rejected'),
      defaultValue: 'pending',
      allowNull: false,
    },
    // Actual values for comparison (populated later)
    actualValues: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    // Error metrics (calculated after validation)
    predictionError: {
      type: DataTypes.JSONB,
      allowNull: true,
    },
    // Metadata
    metadata: {
      type: DataTypes.JSONB,
      defaultValue: {},
      allowNull: true,
    },
  }, {
    tableName: 'predictions',
    timestamps: true,
    indexes: [
      { unique: true, fields: ['prediction_id'] },
      { fields: ['model_name'] },
      { fields: ['prediction_type'] },
      { fields: ['latitude', 'longitude'] },
      { fields: ['prediction_time'] },
      { fields: ['forecast_time'] },
      { fields: ['status'] },
      { fields: ['validation_status'] },
      { fields: ['model_name', 'prediction_type'] },
      { fields: ['forecast_time', 'status'] },
    ],
  });

  // Instance methods
  PredictionModel.prototype.calculateError = function(actualValues) {
    if (!actualValues) return null;
    
    const errors = {};
    const predicted = this.getPredictedValues();
    
    Object.keys(predicted).forEach(param => {
      if (actualValues[param] !== undefined && predicted[param] !== undefined) {
        const actual = parseFloat(actualValues[param]);
        const pred = parseFloat(predicted[param]);
        
        errors[param] = {
          absolute_error: Math.abs(actual - pred),
          relative_error: actual !== 0 ? Math.abs((actual - pred) / actual) : null,
          squared_error: Math.pow(actual - pred, 2),
        };
      }
    });
    
    this.actualValues = actualValues;
    this.predictionError = errors;
    this.validationStatus = 'validated';
    
    return this.save();
  };

  PredictionModel.prototype.getPredictedValues = function() {
    return {
      pm25: this.predictedPm25,
      pm10: this.predictedPm10,
      ozone: this.predictedOzone,
      no2: this.predictedNo2,
      so2: this.predictedSo2,
      co: this.predictedCo,
      aqi: this.predictedAqi,
    };
  };

  PredictionModel.prototype.getConfidenceLevel = function() {
    if (!this.confidenceScore) return 'unknown';
    
    const score = parseFloat(this.confidenceScore);
    if (score >= 0.9) return 'very_high';
    if (score >= 0.8) return 'high';
    if (score >= 0.7) return 'medium';
    if (score >= 0.6) return 'low';
    return 'very_low';
  };

  // Class methods
  PredictionModel.findByLocation = function(lat, lon, radius = 10, limit = 50) {
    // Simple bounding box query - in production, use proper spatial queries
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
        status: 'completed',
      },
      order: [['forecastTime', 'ASC']],
      limit,
    });
  };

  PredictionModel.findByTimeRange = function(startTime, endTime, predictionType = null) {
    const whereClause = {
      forecastTime: {
        [sequelize.Sequelize.Op.between]: [startTime, endTime],
      },
      status: 'completed',
    };
    
    if (predictionType) {
      whereClause.predictionType = predictionType;
    }
    
    return this.findAll({
      where: whereClause,
      order: [['forecastTime', 'ASC']],
    });
  };

  PredictionModel.findByModel = function(modelName, modelVersion = null) {
    const whereClause = { modelName };
    
    if (modelVersion) {
      whereClause.modelVersion = modelVersion;
    }
    
    return this.findAll({
      where: whereClause,
      order: [['predictionTime', 'DESC']],
    });
  };

  PredictionModel.getModelPerformance = function(modelName, timeRange = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - timeRange);
    
    return this.findAll({
      where: {
        modelName,
        validationStatus: 'validated',
        predictionTime: {
          [sequelize.Sequelize.Op.gte]: startDate,
        },
      },
      attributes: [
        'modelAccuracy',
        'modelRmse',
        'confidenceScore',
        'predictionError',
      ],
    });
  };

  return PredictionModel;
};

module.exports = Prediction;