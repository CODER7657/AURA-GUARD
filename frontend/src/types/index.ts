export interface ApiResponse<T> {
  success: boolean;
  message?: string;
  data: T;
  timestamp?: string;
}

export interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
}

// Air quality prediction response
export interface AirQualityData {
  location: {
    latitude: number;
    longitude: number;
  };
  forecastHours: number;
  timestamp: string;
  prediction: {
    pm25_concentration: number;
    confidence: number;
    aqi: number;
    category: string;
    health_impact: {
      category: string;
      aqi_range: string;
      description: string;
      recommendations: string;
    };
    alert_level: string;
  };
  model_performance: {
    live_forecasts: string;
    backtest_1h: HorizonScore | null;
  };
  satellite_data: {
    source: string;
    parameters: any[];
    quality_score: number;
    fallback_mode: boolean;
  };
  predictions: Array<{
    timestamp: string;
    hour_offset: number;
    aqi: number;
    category: string;
    color: string;
    confidence: number;
    pollutants: {
      no2: number;
      o3: number;
      pm25: number;
      pm10: number;
    };
  }>;
  // true when the AQI forecast above is simulated (more than 1 hour ahead)
  aqi_forecast_simulated?: boolean;
  // Real NO2 model forecast at the nearest EPA monitor within 80 km, or null
  no2_forecast?: NO2NearestForecast | null;
}

export interface NO2NearestForecast {
  status: string;
  distance_km: number;
  forecast: {
    station: string;
    place?: string;
    issued_at_utc: string;
    forecasts: Array<{
      horizon_h: number;
      target_time_local: string;
      no2_ppb: number;
      interval90_low_ppb: number;
      interval90_high_ppb: number;
      aqi_category: string;
    }>;
  };
}

// Legacy interface for backward compatibility
export interface AirQualityDataLegacy {
  id: string;
  location: string;
  coordinates: {
    lat: number;
    lng: number;
  };
  aqi: number;
  pm25: number;
  pm10: number;
  no2: number;
  so2: number;
  co: number;
  o3: number;
  timestamp: string;
  forecast?: AirQualityForecast[];
}

export interface AirQualityForecast {
  date: string;
  aqi: number;
  description: string;
  confidence: number;
}

// Scores for one forecast horizon from the walk-forward backtest on EPA data
export interface HorizonScore {
  mae_ppb: number;
  rmse_ppb: number;
  r2: number;
  skill_vs_persistence: number;
  persistence_mae_ppb: number;
  interval90_coverage: number;
}

// Shape of ml/results/model_performance.json, served by GET /predictions/accuracy
export interface ModelPerformance {
  status: string;
  live_forecasts: string;
  model?: string;
  target?: string;
  data?: string;
  evaluation?: string;
  horizons: Record<string, HorizonScore>;
  predict_ms_per_row?: number;
}

// System Health Interface
export interface SystemHealth {
  status: string;
  timestamp: string;
  uptime: number;
  version: string;
  environment: string;
  services: {
    postgresql: {
      status: string;
      host: string;
      database: string;
    };
    redis: {
      status: string;
    };
  };
  memory: {
    rss: string;
    heapTotal: string;
    heapUsed: string;
    external: string;
  };
  cpu: {
    user: number;
    system: number;
  };
}

// Framer component props types
export interface FramerComponentProps {
  onSubmit?: (data: any) => void;
  onClick?: () => void;
  isLoading?: boolean;
  data?: any;
  className?: string;
}