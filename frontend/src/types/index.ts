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

// NASA TEMPO Enhanced LSTM Air Quality Data Interface
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
    accuracy: number;
    mae: number;
    inference_time_ms: number;
  };
  satellite_data: {
    source: string;
    parameters: any[];
    quality_score: number;
    fallback_mode: boolean;
  };
  nasa_compliance: {
    accuracy_target: number;
    current_accuracy: number;
    compliance_status: string;
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

// NASA TEMPO Model Performance Interface
export interface ModelPerformance {
  model_performance: {
    r2_score: number;
    mae: number;
    rmse: number;
    inference_time_ms: number;
    architecture: string;
    parameters: number;
  };
  nasa_compliance: {
    accuracy_target: number;
    current_accuracy: number;
    gap: number;
    compliance_percentage: number;
    status: string;
  };
  benchmarks: {
    error_tolerance: string;
    latency_requirement: string;
    accuracy_requirement: string;
  };
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