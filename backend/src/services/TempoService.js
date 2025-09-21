const axios = require('axios');
const config = require('../config/config');
const logger = require('../utils/logger');
const { setCache, getCache } = require('../config/redis');

class TempoService {
  constructor() {
    this.baseURL = config.tempoApiBaseUrl;
    this.apiKey = config.tempoApiKey;
    this.timeout = 15000; // 15 seconds (satellite data can be slower)
    
    // Create axios instance
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: this.timeout,
      headers: {
        'User-Agent': 'NASA-Air-Quality-Forecasting/1.0',
        'Accept': 'application/json',
      },
    });

    // Add request interceptor for API key
    this.client.interceptors.request.use((config) => {
      if (this.apiKey) {
        config.headers.Authorization = `Bearer ${this.apiKey}`;
      }
      return config;
    });

    // Add response interceptor for logging
    this.client.interceptors.response.use(
      (response) => {
        logger.external('TEMPO API Success', {
          url: response.config.url,
          status: response.status,
          dataSize: response.headers['content-length'] || 'unknown',
        });
        return response;
      },
      (error) => {
        logger.external('TEMPO API Error', {
          url: error.config?.url,
          status: error.response?.status,
          message: error.message,
        });
        return Promise.reject(error);
      }
    );
  }

  /**
   * Get latest TEMPO satellite data for a geographic region
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @param {number} radius - Radius in kilometers (default: 50)
   * @param {Array} parameters - Parameters to retrieve ['NO2', 'O3', 'HCHO', etc.]
   * @returns {Promise<Object>} TEMPO satellite data
   */
  async getLatestData(lat, lon, radius = 50, parameters = ['NO2', 'O3', 'HCHO']) {
    try {
      const cacheKey = `tempo:latest:${lat}:${lon}:${radius}:${parameters.join(',')}`;
      const cached = await getCache(cacheKey);
      
      if (cached) {
        logger.cache('TEMPO latest data cache hit', { lat, lon, radius });
        return cached;
      }

      const response = await this.client.get('/data/latest', {
        params: {
          latitude: lat,
          longitude: lon,
          radius: radius,
          parameters: parameters.join(','),
          format: 'json',
          quality_flag: 'good',
        },
      });

      const data = this.transformSatelliteData(response.data);
      
      // Cache for 30 minutes (satellite data updates are less frequent)
      await setCache(cacheKey, data, 1800);
      
      return data;
    } catch (error) {
      logger.error('TEMPO getLatestData error:', error.message);
      throw new Error(`Failed to fetch TEMPO latest data: ${error.message}`);
    }
  }

  /**
   * Get TEMPO data for a specific time range
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @param {string} startTime - Start time in ISO format
   * @param {string} endTime - End time in ISO format
   * @param {Array} parameters - Parameters to retrieve
   * @returns {Promise<Object>} Time series TEMPO data
   */
  async getTimeSeriesData(lat, lon, startTime, endTime, parameters = ['NO2', 'O3']) {
    try {
      const cacheKey = `tempo:timeseries:${lat}:${lon}:${startTime}:${endTime}:${parameters.join(',')}`;
      const cached = await getCache(cacheKey);
      
      if (cached) {
        logger.cache('TEMPO time series cache hit', { lat, lon, startTime, endTime });
        return cached;
      }

      const response = await this.client.get('/data/timeseries', {
        params: {
          latitude: lat,
          longitude: lon,
          start_time: startTime,
          end_time: endTime,
          parameters: parameters.join(','),
          format: 'json',
          quality_flag: 'good',
        },
      });

      const data = this.transformTimeSeriesData(response.data);
      
      // Cache for 2 hours
      await setCache(cacheKey, data, 7200);
      
      return data;
    } catch (error) {
      logger.error('TEMPO getTimeSeriesData error:', error.message);
      throw new Error(`Failed to fetch TEMPO time series data: ${error.message}`);
    }
  }

  /**
   * Get TEMPO data coverage and availability
   * @param {string} date - Date in YYYY-MM-DD format
   * @returns {Promise<Object>} Data coverage information
   */
  async getDataCoverage(date) {
    try {
      const cacheKey = `tempo:coverage:${date}`;
      const cached = await getCache(cacheKey);
      
      if (cached) {
        logger.cache('TEMPO coverage cache hit', { date });
        return cached;
      }

      const response = await this.client.get('/coverage', {
        params: {
          date: date,
          format: 'json',
        },
      });

      const data = this.transformCoverageData(response.data);
      
      // Cache for 6 hours
      await setCache(cacheKey, data, 21600);
      
      return data;
    } catch (error) {
      logger.error('TEMPO getDataCoverage error:', error.message);
      throw new Error(`Failed to fetch TEMPO coverage data: ${error.message}`);
    }
  }

  /**
   * Get TEMPO data quality metrics
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @param {string} parameter - Parameter name
   * @returns {Promise<Object>} Data quality information
   */
  async getDataQuality(lat, lon, parameter) {
    try {
      const cacheKey = `tempo:quality:${lat}:${lon}:${parameter}`;
      const cached = await getCache(cacheKey);
      
      if (cached) {
        logger.cache('TEMPO quality cache hit', { lat, lon, parameter });
        return cached;
      }

      const response = await this.client.get('/quality', {
        params: {
          latitude: lat,
          longitude: lon,
          parameter: parameter,
          format: 'json',
        },
      });

      const data = this.transformQualityData(response.data);
      
      // Cache for 1 hour
      await setCache(cacheKey, data, 3600);
      
      return data;
    } catch (error) {
      logger.error('TEMPO getDataQuality error:', error.message);
      throw new Error(`Failed to fetch TEMPO quality data: ${error.message}`);
    }
  }

  /**
   * Transform TEMPO satellite data to our standard format
   */
  transformSatelliteData(data) {
    if (!data || !data.measurements) {
      return { observations: [], metadata: { source: 'TEMPO', timestamp: new Date().toISOString() } };
    }

    const observations = data.measurements.map(measurement => ({
      stationId: `TEMPO_${measurement.pixel_id || measurement.lat}_${measurement.lon}`,
      stationName: `TEMPO Satellite ${measurement.pixel_id || 'Pixel'}`,
      latitude: parseFloat(measurement.latitude || measurement.lat),
      longitude: parseFloat(measurement.longitude || measurement.lon),
      measurementTime: new Date(measurement.measurement_time).toISOString(),
      dataSource: 'TEMPO',
      
      // Map TEMPO parameters to our standard format
      ...(measurement.NO2 && { no2: parseFloat(measurement.NO2.value) }),
      ...(measurement.O3 && { ozone: parseFloat(measurement.O3.value) }),
      ...(measurement.HCHO && { hcho: parseFloat(measurement.HCHO.value) }),
      ...(measurement.SO2 && { so2: parseFloat(measurement.SO2.value) }),
      
      // Data quality indicators
      dataQuality: this.mapQualityFlag(measurement.quality_flag),
      validationStatus: 'validated',
      
      // Satellite-specific metadata
      metadata: {
        satellite: 'TEMPO',
        pixel_id: measurement.pixel_id,
        solar_zenith_angle: measurement.solar_zenith_angle,
        viewing_zenith_angle: measurement.viewing_zenith_angle,
        cloud_fraction: measurement.cloud_fraction,
        surface_pressure: measurement.surface_pressure,
      },
      
      rawData: measurement,
    }));

    return {
      observations,
      metadata: {
        source: 'TEMPO',
        timestamp: new Date().toISOString(),
        count: observations.length,
        data_version: data.version,
        processing_level: data.processing_level,
      },
    };
  }

  /**
   * Transform time series data
   */
  transformTimeSeriesData(data) {
    if (!data || !data.time_series) {
      return { timeSeries: [], metadata: { source: 'TEMPO', timestamp: new Date().toISOString() } };
    }

    const timeSeries = data.time_series.map(point => ({
      timestamp: new Date(point.time).toISOString(),
      latitude: parseFloat(point.latitude),
      longitude: parseFloat(point.longitude),
      values: point.values,
      quality: this.mapQualityFlag(point.quality_flag),
    }));

    return {
      timeSeries,
      metadata: {
        source: 'TEMPO',
        timestamp: new Date().toISOString(),
        count: timeSeries.length,
        parameters: data.parameters,
      },
    };
  }

  /**
   * Transform coverage data
   */
  transformCoverageData(data) {
    return {
      coverage: {
        date: data.date,
        regions: data.coverage_regions || [],
        passes: data.satellite_passes || [],
        dataAvailability: data.data_availability || {},
      },
      metadata: {
        source: 'TEMPO',
        timestamp: new Date().toISOString(),
      },
    };
  }

  /**
   * Transform quality data
   */
  transformQualityData(data) {
    return {
      quality: {
        overall_score: data.overall_quality,
        parameter_quality: data.parameter_quality || {},
        cloud_coverage: data.cloud_coverage,
        atmospheric_conditions: data.atmospheric_conditions,
        instrument_status: data.instrument_status,
      },
      metadata: {
        source: 'TEMPO',
        timestamp: new Date().toISOString(),
      },
    };
  }

  /**
   * Map TEMPO quality flags to our standard format
   */
  mapQualityFlag(flag) {
    const mapping = {
      0: 'excellent',
      1: 'good',
      2: 'fair',
      3: 'poor',
      'excellent': 'excellent',
      'good': 'good',
      'fair': 'fair',
      'poor': 'poor',
    };
    
    return mapping[flag] || 'fair';
  }

  /**
   * Test API connection
   */
  async testConnection() {
    try {
      // Test with a small request
      await this.getDataCoverage(new Date().toISOString().split('T')[0]);
      return true;
    } catch (error) {
      logger.error('TEMPO connection test failed:', error.message);
      return false;
    }
  }

  /**
   * Get available parameters for TEMPO data
   */
  getAvailableParameters() {
    return [
      'NO2',    // Nitrogen Dioxide
      'O3',     // Ozone
      'HCHO',   // Formaldehyde
      'SO2',    // Sulfur Dioxide
      'CHOCHO', // Glyoxal
      'BrO',    // Bromine Oxide
      'H2O',    // Water Vapor
      'CLOUD',  // Cloud Properties
    ];
  }

  /**
   * Get parameter information
   */
  getParameterInfo(parameter) {
    const info = {
      'NO2': {
        name: 'Nitrogen Dioxide',
        unit: 'molecules/cm²',
        description: 'Tropospheric NO2 column density',
        health_impact: 'Respiratory irritant, contributes to ozone formation',
      },
      'O3': {
        name: 'Ozone',
        unit: 'DU',
        description: 'Total column ozone',
        health_impact: 'Respiratory health effects, UV protection indicator',
      },
      'HCHO': {
        name: 'Formaldehyde',
        unit: 'molecules/cm²',
        description: 'Tropospheric HCHO column density',
        health_impact: 'Carcinogenic, indicator of VOC emissions',
      },
      'SO2': {
        name: 'Sulfur Dioxide',
        unit: 'DU',
        description: 'Total column SO2',
        health_impact: 'Respiratory irritant, contributes to acid rain',
      },
    };

    return info[parameter] || { name: parameter, description: 'Unknown parameter' };
  }
}

module.exports = new TempoService();