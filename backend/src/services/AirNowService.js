const axios = require('axios');
const config = require('../config/config');
const logger = require('../utils/logger');
const { setCache, getCache } = require('../config/redis');

class AirNowService {
  constructor() {
    this.baseURL = config.airnowApiBaseUrl;
    this.apiKey = config.airnowApiKey;
    this.timeout = 10000; // 10 seconds
    
    // Create axios instance
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: this.timeout,
      headers: {
        'User-Agent': 'NASA-Air-Quality-Forecasting/1.0',
      },
    });

    // Add request interceptor for API key
    this.client.interceptors.request.use((config) => {
      if (this.apiKey) {
        config.params = {
          ...config.params,
          API_KEY: this.apiKey,
        };
      }
      return config;
    });

    // Add response interceptor for logging
    this.client.interceptors.response.use(
      (response) => {
        logger.external('AirNow API Success', {
          url: response.config.url,
          status: response.status,
          dataLength: response.data?.length || 0,
        });
        return response;
      },
      (error) => {
        logger.external('AirNow API Error', {
          url: error.config?.url,
          status: error.response?.status,
          message: error.message,
        });
        return Promise.reject(error);
      }
    );
  }

  /**
   * Get current air quality observations by location
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @param {number} distance - Distance in miles (default: 25)
   * @returns {Promise<Object>} Air quality data
   */
  async getCurrentObservations(lat, lon, distance = 25) {
    try {
      const cacheKey = `airnow:current:${lat}:${lon}:${distance}`;
      const cached = await getCache(cacheKey);
      
      if (cached) {
        logger.cache('AirNow current observations cache hit', { lat, lon });
        return cached;
      }

      const response = await this.client.get('/aq/observation/latLong/current/', {
        params: {
          format: 'application/json',
          latitude: lat,
          longitude: lon,
          distance: distance,
          verbose: 1,
        },
      });

      const data = this.transformCurrentObservations(response.data);
      
      // Cache for 15 minutes
      await setCache(cacheKey, data, 900);
      
      return data;
    } catch (error) {
      logger.error('AirNow getCurrentObservations error:', error.message);
      throw new Error(`Failed to fetch current observations: ${error.message}`);
    }
  }

  /**
   * Get air quality forecast by location
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @param {string} date - Date in YYYY-MM-DD format
   * @param {number} distance - Distance in miles (default: 25)
   * @returns {Promise<Object>} Forecast data
   */
  async getForecast(lat, lon, date, distance = 25) {
    try {
      const cacheKey = `airnow:forecast:${lat}:${lon}:${date}:${distance}`;
      const cached = await getCache(cacheKey);
      
      if (cached) {
        logger.cache('AirNow forecast cache hit', { lat, lon, date });
        return cached;
      }

      const response = await this.client.get('/aq/forecast/latLong/', {
        params: {
          format: 'application/json',
          latitude: lat,
          longitude: lon,
          date: date,
          distance: distance,
          verbose: 1,
        },
      });

      const data = this.transformForecast(response.data);
      
      // Cache for 1 hour
      await setCache(cacheKey, data, 3600);
      
      return data;
    } catch (error) {
      logger.error('AirNow getForecast error:', error.message);
      throw new Error(`Failed to fetch forecast: ${error.message}`);
    }
  }

  /**
   * Get historical air quality data
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @param {string} startDate - Start date in YYYY-MM-DD format
   * @param {string} endDate - End date in YYYY-MM-DD format
   * @param {number} distance - Distance in miles (default: 25)
   * @returns {Promise<Object>} Historical data
   */
  async getHistoricalData(lat, lon, startDate, endDate, distance = 25) {
    try {
      const cacheKey = `airnow:historical:${lat}:${lon}:${startDate}:${endDate}:${distance}`;
      const cached = await getCache(cacheKey);
      
      if (cached) {
        logger.cache('AirNow historical data cache hit', { lat, lon, startDate, endDate });
        return cached;
      }

      // Note: Historical data endpoint might require different parameters
      // This is a placeholder implementation
      const response = await this.client.get('/aq/observation/latLong/historical/', {
        params: {
          format: 'application/json',
          latitude: lat,
          longitude: lon,
          startDate: startDate,
          endDate: endDate,
          distance: distance,
          verbose: 1,
        },
      });

      const data = this.transformHistoricalData(response.data);
      
      // Cache for 24 hours (historical data doesn't change)
      await setCache(cacheKey, data, 86400);
      
      return data;
    } catch (error) {
      logger.error('AirNow getHistoricalData error:', error.message);
      throw new Error(`Failed to fetch historical data: ${error.message}`);
    }
  }

  /**
   * Get monitoring sites within distance of coordinates
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @param {number} distance - Distance in miles (default: 25)
   * @returns {Promise<Object>} Monitoring sites
   */
  async getMonitoringSites(lat, lon, distance = 25) {
    try {
      const cacheKey = `airnow:sites:${lat}:${lon}:${distance}`;
      const cached = await getCache(cacheKey);
      
      if (cached) {
        logger.cache('AirNow monitoring sites cache hit', { lat, lon });
        return cached;
      }

      const response = await this.client.get('/aq/data/', {
        params: {
          format: 'application/json',
          latitude: lat,
          longitude: lon,
          distance: distance,
          verbose: 1,
        },
      });

      const data = this.transformMonitoringSites(response.data);
      
      // Cache for 6 hours (sites don't change frequently)
      await setCache(cacheKey, data, 21600);
      
      return data;
    } catch (error) {
      logger.error('AirNow getMonitoringSites error:', error.message);
      throw new Error(`Failed to fetch monitoring sites: ${error.message}`);
    }
  }

  /**
   * Transform current observations data to our standard format
   */
  transformCurrentObservations(data) {
    if (!Array.isArray(data)) {
      return { observations: [], metadata: { source: 'AirNow', timestamp: new Date().toISOString() } };
    }

    const observations = data.map(item => ({
      stationId: item.AQS_ID || item.SiteName,
      stationName: item.SiteName,
      latitude: parseFloat(item.Latitude),
      longitude: parseFloat(item.Longitude),
      measurementTime: new Date(item.DateObserved + 'T' + item.HourObserved + ':00:00Z').toISOString(),
      dataSource: 'EPA_AIRNOW',
      parameter: this.mapParameterName(item.ParameterName),
      value: parseFloat(item.AQI),
      unit: item.Unit,
      aqi: parseInt(item.AQI),
      aqiCategory: this.mapAQICategory(parseInt(item.AQI)),
      rawData: item,
    }));

    return {
      observations,
      metadata: {
        source: 'AirNow',
        timestamp: new Date().toISOString(),
        count: observations.length,
      },
    };
  }

  /**
   * Transform forecast data to our standard format
   */
  transformForecast(data) {
    if (!Array.isArray(data)) {
      return { forecasts: [], metadata: { source: 'AirNow', timestamp: new Date().toISOString() } };
    }

    const forecasts = data.map(item => ({
      stationId: item.AQS_ID || item.SiteName,
      stationName: item.SiteName,
      latitude: parseFloat(item.Latitude),
      longitude: parseFloat(item.Longitude),
      forecastDate: item.DateForecast,
      dataSource: 'EPA_AIRNOW',
      parameter: this.mapParameterName(item.ParameterName),
      aqi: parseInt(item.AQI),
      aqiCategory: this.mapAQICategory(parseInt(item.AQI)),
      discussion: item.Discussion,
      rawData: item,
    }));

    return {
      forecasts,
      metadata: {
        source: 'AirNow',
        timestamp: new Date().toISOString(),
        count: forecasts.length,
      },
    };
  }

  /**
   * Transform historical data to our standard format
   */
  transformHistoricalData(data) {
    // Similar to current observations but for historical data
    return this.transformCurrentObservations(data);
  }

  /**
   * Transform monitoring sites data to our standard format
   */
  transformMonitoringSites(data) {
    if (!Array.isArray(data)) {
      return { sites: [], metadata: { source: 'AirNow', timestamp: new Date().toISOString() } };
    }

    const sites = data.map(item => ({
      siteId: item.AQS_ID || item.SiteName,
      siteName: item.SiteName,
      latitude: parseFloat(item.Latitude),
      longitude: parseFloat(item.Longitude),
      parameters: [this.mapParameterName(item.ParameterName)],
      status: 'active',
      dataSource: 'EPA_AIRNOW',
      rawData: item,
    }));

    return {
      sites,
      metadata: {
        source: 'AirNow',
        timestamp: new Date().toISOString(),
        count: sites.length,
      },
    };
  }

  /**
   * Map AirNow parameter names to our standard names
   */
  mapParameterName(parameterName) {
    const mapping = {
      'PM2.5': 'PM2.5',
      'PM10': 'PM10',
      'OZONE': 'O3',
      'O3': 'O3',
      'NO2': 'NO2',
      'SO2': 'SO2',
      'CO': 'CO',
    };
    
    return mapping[parameterName?.toUpperCase()] || parameterName;
  }

  /**
   * Map AQI values to categories
   */
  mapAQICategory(aqi) {
    if (aqi <= 50) return 'good';
    if (aqi <= 100) return 'moderate';
    if (aqi <= 150) return 'unhealthy_sensitive';
    if (aqi <= 200) return 'unhealthy';
    if (aqi <= 300) return 'very_unhealthy';
    return 'hazardous';
  }

  /**
   * Test API connection
   */
  async testConnection() {
    try {
      // Test with a known location (Washington DC)
      await this.getCurrentObservations(38.9072, -77.0369, 25);
      return true;
    } catch (error) {
      logger.error('AirNow connection test failed:', error.message);
      return false;
    }
  }
}

module.exports = new AirNowService();