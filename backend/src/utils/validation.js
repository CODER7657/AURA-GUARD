const Joi = require('joi');

// Common validation schemas
const schemas = {
  // User registration
  userRegistration: Joi.object({
    name: Joi.string().min(2).max(50).required(),
    email: Joi.string().email().required(),
    password: Joi.string().min(6).max(128).required(),
    confirmPassword: Joi.string().valid(Joi.ref('password')).required(),
  }),

  // User login
  userLogin: Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().required(),
  }),

  // Coordinates
  coordinates: Joi.object({
    latitude: Joi.number().min(-90).max(90).required(),
    longitude: Joi.number().min(-180).max(180).required(),
  }),

  // Air quality prediction request
  predictionRequest: Joi.object({
    location: Joi.alternatives().try(
      Joi.object({
        latitude: Joi.number().min(-90).max(90).required(),
        longitude: Joi.number().min(-180).max(180).required(),
      }),
      Joi.string().min(2).max(100) // Address string
    ).required(),
    parameters: Joi.array().items(
      Joi.string().valid('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO')
    ).min(1).required(),
    forecastHours: Joi.number().min(1).max(72).default(24),
  }),

  // Notification subscription
  notificationSubscription: Joi.object({
    email: Joi.string().email().required(),
    preferences: Joi.object({
      thresholdLevels: Joi.array().items(
        Joi.string().valid('moderate', 'unhealthy_sensitive', 'unhealthy', 'very_unhealthy', 'hazardous')
      ).min(1).required(),
      parameters: Joi.array().items(
        Joi.string().valid('PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO')
      ).min(1).required(),
      frequency: Joi.string().valid('immediate', 'daily', 'weekly').default('immediate'),
      locations: Joi.array().items(
        Joi.alternatives().try(
          Joi.object({
            latitude: Joi.number().min(-90).max(90).required(),
            longitude: Joi.number().min(-180).max(180).required(),
            name: Joi.string().max(100).optional(),
          }),
          Joi.string().min(2).max(100) // Address string
        )
      ).min(1).required(),
    }).required(),
  }),

  // Date range query
  dateRange: Joi.object({
    startDate: Joi.date().iso().required(),
    endDate: Joi.date().iso().min(Joi.ref('startDate')).required(),
    timeZone: Joi.string().default('UTC'),
  }),

  // Pagination
  pagination: Joi.object({
    page: Joi.number().integer().min(1).default(1),
    limit: Joi.number().integer().min(1).max(100).default(20),
    sortBy: Joi.string().default('createdAt'),
    sortOrder: Joi.string().valid('asc', 'desc').default('desc'),
  }),
};

// Validation middleware factory
const validate = (schema, property = 'body') => {
  return (req, res, next) => {
    const { error, value } = schema.validate(req[property], {
      abortEarly: false,
      allowUnknown: false,
      stripUnknown: true,
    });

    if (error) {
      const errorDetails = error.details.map(detail => ({
        field: detail.path.join('.'),
        message: detail.message,
      }));

      return res.status(400).json({
        success: false,
        error: {
          message: 'Validation error',
          details: errorDetails,
          statusCode: 400,
        },
      });
    }

    // Replace req[property] with validated and sanitized value
    req[property] = value;
    next();
  };
};

// Quick validation functions
const validateEmail = (email) => {
  const emailSchema = Joi.string().email();
  return emailSchema.validate(email);
};

const validateCoordinates = (lat, lon) => {
  const coordSchema = Joi.object({
    latitude: Joi.number().min(-90).max(90).required(),
    longitude: Joi.number().min(-180).max(180).required(),
  });
  return coordSchema.validate({ latitude: lat, longitude: lon });
};

const validateObjectId = (id) => {
  const objectIdSchema = Joi.string().pattern(/^[0-9a-fA-F]{24}$/);
  return objectIdSchema.validate(id);
};

module.exports = {
  schemas,
  validate,
  validateEmail,
  validateCoordinates,
  validateObjectId,
};