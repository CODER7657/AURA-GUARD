import { useState, useEffect, useCallback } from 'react';
import { ApiService } from '../api/service';
import type { AirQualityData, User, ModelPerformance, SystemHealth } from '../types';

// Custom hook for API data fetching with loading and error states
export function useApi<T>(
  apiCall: () => Promise<T>,
  dependencies: any[] = []
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiCall();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
}

// Hook for NASA TEMPO real-time predictions
export function useRealTimePrediction(coordinates?: { latitude: number; longitude: number; forecast_hours?: number }) {
  return useApi(
    () => coordinates ? ApiService.getRealTimePrediction(coordinates) : Promise.resolve(null),
    [coordinates?.latitude, coordinates?.longitude, coordinates?.forecast_hours]
  );
}

// Hook for extended forecasts
export function useExtendedForecast(coordinates?: { latitude: number; longitude: number; duration?: number }) {
  return useApi(
    () => coordinates ? ApiService.getExtendedForecast(coordinates) : Promise.resolve(null),
    [coordinates?.latitude, coordinates?.longitude, coordinates?.duration]
  );
}

// Hook for model performance metrics
export function useModelPerformance() {
  return useApi<ModelPerformance>(
    () => ApiService.getModelAccuracy(),
    []
  );
}

// Hook for system health status
export function useSystemHealth() {
  return useApi<SystemHealth>(
    () => ApiService.getHealthStatus(),
    []
  );
}

// Legacy hook for backward compatibility
export function useAirQuality(location?: string) {
  console.warn('useAirQuality is deprecated, use useRealTimePrediction instead');
  return useApi(
    () => ApiService.getAirQualityData(location),
    [location]
  );
}

// Legacy hook for backward compatibility
export function useAirQualityForecast(coordinates?: { lat: number; lng: number }) {
  return useApi(
    () => coordinates ? ApiService.getAirQualityForecast(coordinates) : Promise.resolve(null),
    [coordinates?.lat, coordinates?.lng]
  );
}

// Hook for user data
export function useUser() {
  return useApi(() => ApiService.getCurrentUser());
}

// Hook for form submissions with Framer components
export function useFormSubmission() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  const submitForm = useCallback(async (endpoint: string, formData: Record<string, any>) => {
    try {
      setIsSubmitting(true);
      setSubmitError(null);
      setSubmitSuccess(false);
      
      await ApiService.submitForm(endpoint, formData);
      setSubmitSuccess(true);
      return true;
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : 'Submission failed');
      return false;
    } finally {
      setIsSubmitting(false);
    }
  }, []);

  const resetSubmissionState = useCallback(() => {
    setSubmitError(null);
    setSubmitSuccess(false);
  }, []);

  return {
    submitForm,
    isSubmitting,
    submitError,
    submitSuccess,
    resetSubmissionState
  };
}

// Hook for authentication state
export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (token) {
      ApiService.getCurrentUser()
        .then((userData) => {
          setUser(userData);
          setIsAuthenticated(true);
        })
        .catch(() => {
          localStorage.removeItem('authToken');
          setIsAuthenticated(false);
        })
        .finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    try {
      const { user: userData } = await ApiService.login(email, password);
      setUser(userData);
      setIsAuthenticated(true);
      return true;
    } catch (error) {
      return false;
    }
  }, []);

  const logout = useCallback(async () => {
    await ApiService.logout();
    setUser(null);
    setIsAuthenticated(false);
  }, []);

  return {
    user,
    isAuthenticated,
    isLoading,
    login,
    logout
  };
}