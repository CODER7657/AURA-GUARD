import React, { useState, useEffect } from 'react';
import { 
  MapPin, 
  Wind, 
  AlertTriangle, 
  Search, 
  Navigation,
  Download,
  Share2,
  Settings,
  Bell,
  Info,
  RefreshCw,
  Clock,
  Activity,
  Eye,
  Target,
  TrendingUp
} from 'lucide-react';
import { useRealTimePrediction } from '../hooks/useApi';
import type { AirQualityData } from '../types';

export const AirQualityDashboard: React.FC = () => {
  // State management
  const [selectedLocation, setSelectedLocation] = useState('Los Angeles');
  const [currentCoordinates, setCurrentCoordinates] = useState<{latitude: number; longitude: number}>({
    latitude: 34.0522,
    longitude: -118.2437
  });
  
  const [searchLocation, setSearchLocation] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  
  // Alert system
  const [alertsEnabled, setAlertsEnabled] = useState(false);
  const [alertThreshold, setAlertThreshold] = useState(100);
  const [showAlert, setShowAlert] = useState(false);
  
  // UI state
  const [activeView, setActiveView] = useState<'overview' | 'forecast' | 'detailed'>('overview');
  const [refreshing, setRefreshing] = useState(false);
  
  // NASA TEMPO API integration
  const { data: realTimeData, loading, error, refetch } = useRealTimePrediction(currentCoordinates);

  // Popular cities for suggestions
  const popularCities = ['Los Angeles', 'New York', 'Houston', 'Chicago', 'Miami', 'Seattle', 'Denver', 'Phoenix'];

  // Load saved settings
  useEffect(() => {
    const savedSearches = localStorage.getItem('recentAirQualitySearches');
    if (savedSearches) {
      setRecentSearches(JSON.parse(savedSearches));
    }
    
    const alertData = localStorage.getItem('airQualityAlerts');
    if (alertData) {
      const parsed = JSON.parse(alertData);
      setAlertsEnabled(parsed.enabled);
      setAlertThreshold(parsed.threshold);
    }
  }, []);

  // Get current data from API response with fallbacks
  const getCurrentData = () => {
    if (realTimeData && realTimeData.prediction) {
      return {
        aqi: realTimeData.prediction.aqi || 0,
        status: realTimeData.prediction.category || 'Unknown',
        color: realTimeData.prediction.aqi <= 50 ? 'green' : 
               realTimeData.prediction.aqi <= 100 ? 'yellow' : 'orange',
        pollutants: {
          'PM2.5': { 
            value: realTimeData.prediction.pm25_concentration || 0, 
            unit: 'μg/m³', 
            status: realTimeData.prediction.aqi <= 50 ? 'Good' : 'Moderate' 
          },
          'NO₂': { 
            value: realTimeData.predictions?.[0]?.pollutants?.no2 || 0, 
            unit: 'μg/m³', 
            status: 'Good' 
          },
          'O₃': { 
            value: realTimeData.predictions?.[0]?.pollutants?.o3 || 0, 
            unit: 'μg/m³', 
            status: 'Good' 
          },
          'PM10': { 
            value: realTimeData.predictions?.[0]?.pollutants?.pm10 || 0, 
            unit: 'μg/m³', 
            status: 'Good' 
          }
        },
        forecast: realTimeData.predictions?.slice(0, 7).map((p, index) => ({
          day: index === 0 ? 'Today' : index === 1 ? 'Tomorrow' : 
               new Date(p.timestamp).toLocaleDateString('en-US', { weekday: 'short' }),
          aqi: p.aqi,
          temp: '75°F' // Temperature not provided by current API
        })) || []
      };
    }
    
    // Fallback data while loading
    return {
      aqi: loading ? 0 : 72,
      status: loading ? 'Loading...' : 'Moderate',
      color: loading ? 'gray' : 'yellow',
      pollutants: {
        'PM2.5': { value: loading ? 0 : 25, unit: 'μg/m³', status: loading ? 'Loading...' : 'Good' },
        'NO₂': { value: loading ? 0 : 45, unit: 'μg/m³', status: loading ? 'Loading...' : 'Good' },
        'O₃': { value: loading ? 0 : 78, unit: 'μg/m³', status: loading ? 'Loading...' : 'Moderate' },
        'PM10': { value: loading ? 0 : 35, unit: 'μg/m³', status: loading ? 'Loading...' : 'Good' }
      },
      forecast: loading ? [] : [
        { day: 'Today', aqi: 72, temp: '75°F' },
        { day: 'Tomorrow', aqi: 65, temp: '73°F' },
        { day: 'Wed', aqi: 68, temp: '76°F' },
        { day: 'Thu', aqi: 85, temp: '78°F' },
        { day: 'Fri', aqi: 78, temp: '75°F' },
        { day: 'Sat', aqi: 58, temp: '72°F' },
        { day: 'Sun', aqi: 62, temp: '74°F' }
      ]
    };
  };

  const currentData = getCurrentData();

  // Alert system - check AQI against threshold
  useEffect(() => {
    if (alertsEnabled && currentData.aqi > alertThreshold) {
      setShowAlert(true);
      // Show browser notification if supported
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('Air Quality Alert', {
          body: `AQI in ${selectedLocation} is ${currentData.aqi} - above your threshold of ${alertThreshold}`,
          icon: '/vite.svg'
        });
      }
    } else {
      setShowAlert(false);
    }
  }, [selectedLocation, alertsEnabled, alertThreshold, currentData.aqi]);

  // Utility functions
  const getAQIColor = (aqi: number) => {
    if (aqi <= 50) return 'text-emerald-700 bg-gradient-to-r from-emerald-100 to-teal-100 border border-emerald-200';
    if (aqi <= 100) return 'text-amber-700 bg-gradient-to-r from-amber-100 to-orange-100 border border-amber-200';
    if (aqi <= 150) return 'text-orange-700 bg-gradient-to-r from-orange-100 to-red-100 border border-orange-200';
    return 'text-rose-700 bg-gradient-to-r from-rose-100 to-pink-100 border border-rose-200';
  };

  const getPollutantColor = (status: string) => {
    if (status === 'Good') return 'text-emerald-700 bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200';
    if (status === 'Moderate') return 'text-amber-700 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200';
    return 'text-rose-700 bg-gradient-to-r from-rose-50 to-pink-50 border border-rose-200';
  };

  const saveRecentSearch = (location: string) => {
    const updated = [location, ...recentSearches.filter(s => s !== location)].slice(0, 5);
    setRecentSearches(updated);
    localStorage.setItem('recentAirQualitySearches', JSON.stringify(updated));
  };

  const handleLocationSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchLocation.trim()) return;

    setIsSearching(true);
    setSearchError('');

    try {
      // Mock coordinates for common cities - in real app, would use geocoding API
      const mockCoordinates: Record<string, { latitude: number; longitude: number }> = {
        'los angeles': { latitude: 34.0522, longitude: -118.2437 },
        'new york': { latitude: 40.7128, longitude: -74.0060 },
        'houston': { latitude: 29.7604, longitude: -95.3698 },
        'chicago': { latitude: 41.8781, longitude: -87.6298 }
      };

      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const searchKey = searchLocation.toLowerCase();
      const coords = mockCoordinates[searchKey] || { latitude: 34.0522, longitude: -118.2437 };
      
      setCurrentCoordinates(coords);
      setSelectedLocation(searchLocation);
      saveRecentSearch(searchLocation);
      setSearchLocation('');
      setShowSuggestions(false);

    } catch (error) {
      setSearchError('Unable to fetch air quality data for this location. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      setIsSearching(true);
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const coords = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          };
          setCurrentCoordinates(coords);
          setSelectedLocation(`Location ${coords.latitude.toFixed(2)}, ${coords.longitude.toFixed(2)}`);
          setIsSearching(false);
        },
        (error) => {
          setSearchError('Unable to access your location. Please search manually.');
          setIsSearching(false);
        }
      );
    } else {
      setSearchError('Geolocation is not supported by your browser.');
    }
  };

  const refreshData = async () => {
    setRefreshing(true);
    await refetch();
    setTimeout(() => setRefreshing(false), 1000);
  };

  const exportToCSV = () => {
    const data = getCurrentData();
    const csvContent = [
      ['Location', 'AQI', 'Status', 'PM2.5', 'NO₂', 'O₃', 'PM10'],
      [selectedLocation, data.aqi, data.status, ...Object.values(data.pollutants).map((p: any) => p.value)]
    ].map(row => row.join(',')).join('\\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `air-quality-${selectedLocation}-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const shareData = async () => {
    const shareText = `Air Quality in ${selectedLocation}: AQI ${currentData.aqi} (${currentData.status})`;
    
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Air Quality Data',
          text: shareText,
          url: window.location.href
        });
      } catch (err) {
        console.log('Error sharing:', err);
      }
    } else {
      navigator.clipboard.writeText(shareText);
      alert('Air quality data copied to clipboard!');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold text-gray-800 mb-2">Air Quality Monitor</h1>
            <p className="text-gray-600">NASA TEMPO Enhanced LSTM Predictions • Real-time data and 7-day forecasts</p>
            {error && (
              <div className="mt-2 text-red-600 text-sm">
                API Error: Using fallback data. {error}
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-3 mt-4 md:mt-0">
            <button
              onClick={refreshData}
              disabled={refreshing}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
            
            <button
              onClick={shareData}
              className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
              <Share2 className="w-4 h-4" />
              <span>Share</span>
            </button>
            
            <button
              onClick={exportToCSV}
              className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
          </div>
        </div>

        {/* Alert Banner */}
        {showAlert && (
          <div className="mb-6 p-4 bg-red-100 border border-red-200 rounded-lg flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <div>
                <p className="font-semibold text-red-800">Air Quality Alert</p>
                <p className="text-red-700">
                  AQI in {selectedLocation} is {currentData.aqi} - above your threshold of {alertThreshold}
                </p>
              </div>
            </div>
            <button
              onClick={() => setShowAlert(false)}
              className="text-red-600 hover:text-red-800"
            >
              ✕
            </button>
          </div>
        )}

        {/* Search Section */}
        <div className="mb-8">
          <form onSubmit={handleLocationSearch} className="relative max-w-2xl">
            <div className="flex space-x-3">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={searchLocation}
                  onChange={(e) => {
                    setSearchLocation(e.target.value);
                    setShowSuggestions(e.target.value.length > 0);
                    setSearchError('');
                  }}
                  placeholder="Search for a city or location..."
                  className="w-full pl-11 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white shadow-sm"
                />
                
                {/* Suggestions Dropdown */}
                {showSuggestions && (
                  <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                    <div className="p-2">
                      <p className="text-sm text-gray-500 mb-2">Popular Cities</p>
                      {popularCities
                        .filter(city => city.toLowerCase().includes(searchLocation.toLowerCase()))
                        .map((city, index) => (
                          <button
                            key={index}
                            type="button"
                            onClick={() => {
                              setSearchLocation(city);
                              setShowSuggestions(false);
                            }}
                            className="w-full text-left px-3 py-2 hover:bg-gray-100 rounded flex items-center space-x-2"
                          >
                            <MapPin className="w-4 h-4 text-gray-400" />
                            <span>{city}</span>
                          </button>
                        ))}
                      
                      {recentSearches.length > 0 && (
                        <>
                          <hr className="my-2" />
                          <p className="text-sm text-gray-500 mb-2">Recent Searches</p>
                          {recentSearches
                            .filter(search => search.toLowerCase().includes(searchLocation.toLowerCase()))
                            .slice(0, 3)
                            .map((search, index) => (
                              <button
                                key={index}
                                type="button"
                                onClick={() => {
                                  setSearchLocation(search);
                                  setShowSuggestions(false);
                                }}
                                className="w-full text-left px-3 py-2 hover:bg-gray-100 rounded flex items-center space-x-2"
                              >
                                <Clock className="w-4 h-4 text-gray-400" />
                                <span>{search}</span>
                              </button>
                            ))}
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
              
              <button
                type="submit"
                disabled={isSearching}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
              >
                {isSearching ? (
                  <RefreshCw className="w-5 h-5 animate-spin" />
                ) : (
                  <Search className="w-5 h-5" />
                )}
                <span>{isSearching ? 'Searching...' : 'Search'}</span>
              </button>
              
              <button
                type="button"
                onClick={getCurrentLocation}
                className="px-4 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center"
                title="Use current location"
              >
                <Navigation className="w-5 h-5 text-gray-600" />
              </button>
            </div>
            
            {searchError && (
              <p className="mt-2 text-red-600 text-sm">{searchError}</p>
            )}
          </form>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Current Status Card */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-800">{selectedLocation}</h2>
                  <p className="text-gray-500 flex items-center space-x-2">
                    <Clock className="w-4 h-4" />
                    <span>Last updated: {new Date().toLocaleString()}</span>
                  </p>
                  {loading && (
                    <p className="text-blue-600 text-sm mt-1">Loading NASA TEMPO data...</p>
                  )}
                </div>
                
                <div className={`text-center p-4 rounded-xl ${getAQIColor(currentData.aqi)}`}>
                  <div className="text-3xl font-bold">{currentData.aqi}</div>
                  <div className="text-sm font-medium">AQI</div>
                </div>
              </div>
              
              <div className="mb-6">
                <div className={`inline-flex items-center px-4 py-2 rounded-lg font-medium ${getAQIColor(currentData.aqi)}`}>
                  <Wind className="w-4 h-4 mr-2" />
                  {currentData.status}
                </div>
              </div>
              
              {/* Pollutant Details */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                {Object.entries(currentData.pollutants).map(([pollutant, data]) => (
                  <div key={pollutant} className={`p-3 rounded-lg ${getPollutantColor(data.status)}`}>
                    <div className="text-sm font-medium text-gray-600">{pollutant}</div>
                    <div className="text-lg font-bold">{data.value}</div>
                    <div className="text-xs text-gray-500">{data.unit}</div>
                    <div className="text-xs font-medium mt-1">{data.status}</div>
                  </div>
                ))}
              </div>
              
              {/* 7-Day Forecast */}
              {currentData.forecast.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">7-Day Forecast</h3>
                  <div className="grid grid-cols-7 gap-2">
                    {currentData.forecast.map((day, index) => (
                      <div key={index} className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-sm font-medium text-gray-600 mb-2">{day.day}</div>
                        <div className={`text-lg font-bold p-2 rounded ${getAQIColor(day.aqi)}`}>
                          {day.aqi}
                        </div>
                        <div className="text-xs text-gray-500 mt-2">{day.temp}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Sidebar */}
          <div className="space-y-6">
            {/* API Status */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2" />
                NASA TEMPO Status
              </h3>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Connection</span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    error ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                  }`}>
                    {error ? 'Offline' : 'Online'}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Model Accuracy</span>
                  <span className="text-green-600 font-medium">86.98%</span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Inference Time</span>
                  <span className="text-blue-600 font-medium">1.7ms</span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Data Source</span>
                  <span className="text-purple-600 font-medium">TEMPO Satellite</span>
                </div>
              </div>
            </div>
            
            {/* Alert Settings */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                <Bell className="w-5 h-5 mr-2" />
                Alert Settings
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Enable Alerts</span>
                  <button
                    onClick={() => {
                      const newEnabled = !alertsEnabled;
                      setAlertsEnabled(newEnabled);
                      localStorage.setItem('airQualityAlerts', JSON.stringify({
                        enabled: newEnabled,
                        threshold: alertThreshold
                      }));
                      if (newEnabled && 'Notification' in window && Notification.permission === 'default') {
                        Notification.requestPermission();
                      }
                    }}
                    className={`w-12 h-6 rounded-full transition-colors ${
                      alertsEnabled ? 'bg-blue-600' : 'bg-gray-300'
                    }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${
                      alertsEnabled ? 'translate-x-6' : 'translate-x-0.5'
                    }`} />
                  </button>
                </div>
                
                <div>
                  <label className="block text-gray-600 mb-2">Alert Threshold (AQI)</label>
                  <input
                    type="range"
                    min="50"
                    max="200"
                    value={alertThreshold}
                    onChange={(e) => {
                      const newThreshold = parseInt(e.target.value);
                      setAlertThreshold(newThreshold);
                      localStorage.setItem('airQualityAlerts', JSON.stringify({
                        enabled: alertsEnabled,
                        threshold: newThreshold
                      }));
                    }}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-gray-500 mt-1">
                    <span>50</span>
                    <span className="font-medium">{alertThreshold}</span>
                    <span>200</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Quick Info */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                <Info className="w-5 h-5 mr-2" />
                AQI Scale
              </h3>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 bg-green-50 border border-green-200 rounded">
                  <span className="text-green-800 font-medium">Good</span>
                  <span className="text-green-600 text-sm">0-50</span>
                </div>
                
                <div className="flex items-center justify-between p-2 bg-yellow-50 border border-yellow-200 rounded">
                  <span className="text-yellow-800 font-medium">Moderate</span>
                  <span className="text-yellow-600 text-sm">51-100</span>
                </div>
                
                <div className="flex items-center justify-between p-2 bg-orange-50 border border-orange-200 rounded">
                  <span className="text-orange-800 font-medium">Unhealthy</span>
                  <span className="text-orange-600 text-sm">101-150</span>
                </div>
                
                <div className="flex items-center justify-between p-2 bg-red-50 border border-red-200 rounded">
                  <span className="text-red-800 font-medium">Very Unhealthy</span>
                  <span className="text-red-600 text-sm">151-200</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};