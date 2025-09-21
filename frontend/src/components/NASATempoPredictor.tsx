import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useRealTimePrediction, useModelPerformance, useSystemHealth } from '../hooks/useApi';
import type { AirQualityData, ModelPerformance, SystemHealth } from '../types';

export const NASATempoPredictor: React.FC = () => {
  const [coordinates, setCoordinates] = useState<{ latitude: number; longitude: number }>({
    latitude: 34.0522, // Los Angeles default
    longitude: -118.2437
  });
  
  const [forecastHours, setForecastHours] = useState(1);
  
  // API hooks
  const { data: prediction, loading: predictionLoading, error: predictionError, refetch } = 
    useRealTimePrediction({ ...coordinates, forecast_hours: forecastHours });
  
  const { data: modelPerformance, loading: performanceLoading } = useModelPerformance();
  
  const { data: systemHealth, loading: healthLoading } = useSystemHealth();

  const handlePredict = () => {
    refetch();
  };

  const getAQIColor = (aqi: number) => {
    if (aqi <= 50) return 'text-green-600';
    if (aqi <= 100) return 'text-yellow-600';
    if (aqi <= 150) return 'text-orange-600';
    if (aqi <= 200) return 'text-red-600';
    if (aqi <= 300) return 'text-purple-600';
    return 'text-red-800';
  };

  const getAQIBgColor = (aqi: number) => {
    if (aqi <= 50) return 'bg-green-100 border-green-200';
    if (aqi <= 100) return 'bg-yellow-100 border-yellow-200';
    if (aqi <= 150) return 'bg-orange-100 border-orange-200';
    if (aqi <= 200) return 'bg-red-100 border-red-200';
    if (aqi <= 300) return 'bg-purple-100 border-purple-200';
    return 'bg-red-200 border-red-300';
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white dark:bg-gray-900 rounded-xl shadow-lg">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          🛰️ NASA TEMPO Enhanced LSTM Air Quality Predictor
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Real-time air quality predictions using NASA satellite data and Enhanced LSTM neural networks
        </p>
      </div>

      {/* Input Controls */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Latitude
          </label>
          <input
            type="number"
            value={coordinates.latitude}
            onChange={(e) => setCoordinates(prev => ({ ...prev, latitude: parseFloat(e.target.value) || 0 }))}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
            placeholder="34.0522"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Longitude
          </label>
          <input
            type="number"
            value={coordinates.longitude}
            onChange={(e) => setCoordinates(prev => ({ ...prev, longitude: parseFloat(e.target.value) || 0 }))}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
            placeholder="-118.2437"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Forecast Hours
          </label>
          <select
            value={forecastHours}
            onChange={(e) => setForecastHours(parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
          >
            <option value={1}>1 Hour</option>
            <option value={6}>6 Hours</option>
            <option value={12}>12 Hours</option>
            <option value={24}>24 Hours</option>
          </select>
        </div>
      </motion.div>

      <button
        onClick={handlePredict}
        disabled={predictionLoading}
        className="w-full md:w-auto px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white rounded-lg font-medium transition-colors mb-6"
      >
        {predictionLoading ? 'Generating Prediction...' : '🔮 Generate Prediction'}
      </button>

      {/* Error Display */}
      {predictionError && (
        <motion.div
          className="mb-6 p-4 bg-red-100 border border-red-200 rounded-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <p className="text-red-700">Error: {predictionError}</p>
        </motion.div>
      )}

      {/* Prediction Results */}
      {prediction && (
        <motion.div
          className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {/* Main Prediction Card */}
          <div className={`p-6 rounded-xl border-2 ${getAQIBgColor(prediction.prediction.aqi)}`}>
            <h3 className="text-xl font-bold mb-4 text-gray-900">
              Air Quality Prediction
            </h3>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="font-medium">AQI:</span>
                <span className={`font-bold text-2xl ${getAQIColor(prediction.prediction.aqi)}`}>
                  {prediction.prediction.aqi}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="font-medium">Category:</span>
                <span className={`font-semibold ${getAQIColor(prediction.prediction.aqi)}`}>
                  {prediction.prediction.category}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="font-medium">PM2.5:</span>
                <span className="font-semibold">{prediction.prediction.pm25_concentration.toFixed(2)} μg/m³</span>
              </div>
              
              <div className="flex justify-between">
                <span className="font-medium">Confidence:</span>
                <span className="font-semibold">{(prediction.prediction.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Health Impact Card */}
          <div className="p-6 bg-gray-50 dark:bg-gray-800 rounded-xl">
            <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">
              Health Impact
            </h3>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="font-medium text-gray-700 dark:text-gray-300">AQI Range:</span>
                <span className="font-semibold text-gray-900 dark:text-white">
                  {prediction.prediction.health_impact.aqi_range}
                </span>
              </div>
              
              <div className="text-sm text-gray-600 dark:text-gray-400">
                <p><strong>Description:</strong> {prediction.prediction.health_impact.description}</p>
                <p className="mt-2"><strong>Recommendations:</strong> {prediction.prediction.health_impact.recommendations}</p>
              </div>
              
              <div className="flex justify-between">
                <span className="font-medium text-gray-700 dark:text-gray-300">Alert Level:</span>
                <span className={`font-semibold ${prediction.prediction.alert_level === 'None' ? 'text-green-600' : 'text-red-600'}`}>
                  {prediction.prediction.alert_level}
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Model Performance & System Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Performance */}
        {modelPerformance && (
          <motion.div
            className="p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <h3 className="text-lg font-bold mb-4 text-blue-900 dark:text-blue-100">
              🧠 Model Performance
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>R² Score:</span>
                <span className="font-semibold">{modelPerformance.model_performance.r2_score}</span>
              </div>
              <div className="flex justify-between">
                <span>Inference Time:</span>
                <span className="font-semibold">{modelPerformance.model_performance.inference_time_ms.toFixed(1)}ms</span>
              </div>
              <div className="flex justify-between">
                <span>NASA Compliance:</span>
                <span className="font-semibold text-green-600">
                  {modelPerformance.nasa_compliance.compliance_percentage.toFixed(1)}%
                </span>
              </div>
            </div>
          </motion.div>
        )}

        {/* System Health */}
        {systemHealth && (
          <motion.div
            className="p-6 bg-green-50 dark:bg-green-900/20 rounded-xl"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <h3 className="text-lg font-bold mb-4 text-green-900 dark:text-green-100">
              ⚡ System Health
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Status:</span>
                <span className={`font-semibold ${systemHealth.status === 'OK' ? 'text-green-600' : 'text-yellow-600'}`}>
                  {systemHealth.status}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Database:</span>
                <span className={`font-semibold ${systemHealth.services?.postgresql?.status === 'connected' ? 'text-green-600' : 'text-red-600'}`}>
                  {systemHealth.services?.postgresql?.status || 'unknown'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Uptime:</span>
                <span className="font-semibold">{Math.floor(systemHealth.uptime / 60)}m {Math.floor(systemHealth.uptime % 60)}s</span>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Satellite Data Info */}
      {prediction?.satellite_data && (
        <motion.div
          className="mt-6 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">🛰️ Satellite Data Status</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600 dark:text-gray-400">Source:</span>
              <p className="font-medium">{prediction.satellite_data.source}</p>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Quality Score:</span>
              <p className="font-medium">{prediction.satellite_data.quality_score}</p>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Mode:</span>
              <p className={`font-medium ${prediction.satellite_data.fallback_mode ? 'text-yellow-600' : 'text-green-600'}`}>
                {prediction.satellite_data.fallback_mode ? 'Fallback' : 'Direct'}
              </p>
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Parameters:</span>
              <p className="font-medium">{prediction.satellite_data.parameters.length} loaded</p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default NASATempoPredictor;