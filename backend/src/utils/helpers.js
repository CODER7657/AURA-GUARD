// Common response helper functions
const successResponse = (res, data, message = 'Success', statusCode = 200) => {
  return res.status(statusCode).json({
    success: true,
    message,
    data,
    timestamp: new Date().toISOString(),
  });
};

const errorResponse = (res, message, statusCode = 500, details = null) => {
  const response = {
    success: false,
    error: {
      message,
      statusCode,
    },
    timestamp: new Date().toISOString(),
  };

  if (details) {
    response.error.details = details;
  }

  return res.status(statusCode).json(response);
};

const paginatedResponse = (res, data, pagination, message = 'Success') => {
  return res.status(200).json({
    success: true,
    message,
    data,
    pagination: {
      currentPage: pagination.page,
      totalPages: Math.ceil(pagination.total / pagination.limit),
      totalItems: pagination.total,
      itemsPerPage: pagination.limit,
      hasNextPage: pagination.page < Math.ceil(pagination.total / pagination.limit),
      hasPrevPage: pagination.page > 1,
    },
    timestamp: new Date().toISOString(),
  });
};

// Async handler wrapper to catch errors
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

// Generate cache key
const generateCacheKey = (prefix, ...parts) => {
  return `${prefix}:${parts.filter(Boolean).join(':')}`;
};

// Format coordinates
const formatCoordinates = (lat, lon, precision = 6) => {
  return {
    latitude: parseFloat(lat).toFixed(precision),
    longitude: parseFloat(lon).toFixed(precision),
  };
};

// Calculate distance between two coordinates (Haversine formula)
const calculateDistance = (lat1, lon1, lat2, lon2) => {
  const R = 6371; // Earth's radius in kilometers
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c; // Distance in kilometers
};

const toRadians = (degrees) => {
  return degrees * (Math.PI / 180);
};

// Delay function for testing/throttling
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Deep clone object
const deepClone = (obj) => {
  if (obj === null || typeof obj !== 'object') return obj;
  if (obj instanceof Date) return new Date(obj.getTime());
  if (obj instanceof Array) return obj.map(item => deepClone(item));
  if (typeof obj === 'object') {
    const clonedObj = {};
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        clonedObj[key] = deepClone(obj[key]);
      }
    }
    return clonedObj;
  }
};

// Remove sensitive fields from object
const sanitizeObject = (obj, sensitiveFields = ['password', 'token', 'secret']) => {
  const sanitized = deepClone(obj);
  
  const removeSensitiveFields = (current) => {
    if (typeof current === 'object' && current !== null) {
      sensitiveFields.forEach(field => {
        if (current.hasOwnProperty(field)) {
          delete current[field];
        }
      });
      
      Object.values(current).forEach(value => {
        if (typeof value === 'object') {
          removeSensitiveFields(value);
        }
      });
    }
  };
  
  removeSensitiveFields(sanitized);
  return sanitized;
};

// Generate random string
const generateRandomString = (length = 32) => {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
};

// Parse query string parameters
const parseQueryParams = (query) => {
  const params = {};
  
  // Convert string 'true'/'false' to boolean
  Object.keys(query).forEach(key => {
    if (query[key] === 'true') {
      params[key] = true;
    } else if (query[key] === 'false') {
      params[key] = false;
    } else if (!isNaN(query[key]) && query[key] !== '') {
      params[key] = Number(query[key]);
    } else {
      params[key] = query[key];
    }
  });
  
  return params;
};

module.exports = {
  successResponse,
  errorResponse,
  paginatedResponse,
  asyncHandler,
  generateCacheKey,
  formatCoordinates,
  calculateDistance,
  delay,
  deepClone,
  sanitizeObject,
  generateRandomString,
  parseQueryParams,
};