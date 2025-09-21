import apiClient from './config';
import type { ApiResponse, AirQualityData, User } from '../types';

export class ApiService {
  // NASA TEMPO Enhanced LSTM Air Quality API methods
  static async getRealTimePrediction(coordinates: { latitude: number; longitude: number; forecast_hours?: number }): Promise<AirQualityData> {
    try {
      const response = await apiClient.post<ApiResponse<AirQualityData>>('/predictions/realtime', coordinates);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching real-time prediction:', error);
      throw new Error('Failed to fetch real-time air quality prediction');
    }
  }

  static async getExtendedForecast(coordinates: { latitude: number; longitude: number; duration?: number }): Promise<AirQualityData> {
    try {
      const response = await apiClient.post<ApiResponse<AirQualityData>>('/predictions/forecast', coordinates);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching extended forecast:', error);
      throw new Error('Failed to fetch extended forecast');
    }
  }

  static async getModelAccuracy(): Promise<any> {
    try {
      const response = await apiClient.get<ApiResponse<any>>('/predictions/accuracy');
      return response.data.data;
    } catch (error) {
      console.error('Error fetching model accuracy:', error);
      throw new Error('Failed to fetch model accuracy metrics');
    }
  }

  // Legacy methods for backward compatibility
  static async getAirQualityData(location?: string): Promise<AirQualityData[]> {
    // Convert location to coordinates if needed, for now return empty array
    console.warn('getAirQualityData is deprecated, use getRealTimePrediction instead');
    return [];
  }

  static async getAirQualityForecast(coordinates: { lat: number; lng: number }): Promise<AirQualityData> {
    return this.getRealTimePrediction({ 
      latitude: coordinates.lat, 
      longitude: coordinates.lng 
    });
  }

  // User API methods
  static async getCurrentUser(): Promise<User> {
    try {
      const response = await apiClient.get<ApiResponse<User>>('/user/profile');
      return response.data.data;
    } catch (error) {
      console.error('Error fetching user profile:', error);
      throw new Error('Failed to fetch user profile');
    }
  }

  static async updateUserProfile(userData: Partial<User>): Promise<User> {
    try {
      const response = await apiClient.put<ApiResponse<User>>('/user/profile', userData);
      return response.data.data;
    } catch (error) {
      console.error('Error updating user profile:', error);
      throw new Error('Failed to update user profile');
    }
  }

  // Authentication methods
  static async register(userData: { name: string; email: string; password: string }): Promise<{ user: User; token: string }> {
    try {
      const response = await apiClient.post<ApiResponse<{ user: User; token: string }>>('/auth/register', userData);
      const { user, token } = response.data.data;
      localStorage.setItem('authToken', token);
      return { user, token };
    } catch (error) {
      console.error('Error during registration:', error);
      throw new Error('Registration failed');
    }
  }

  static async login(email: string, password: string): Promise<{ user: User; token: string }> {
    try {
      const response = await apiClient.post<ApiResponse<{ user: User; token: string }>>('/auth/login', {
        email,
        password,
      });
      const { user, token } = response.data.data;
      localStorage.setItem('authToken', token);
      return { user, token };
    } catch (error) {
      console.error('Error during login:', error);
      throw new Error('Login failed');
    }
  }

  static async logout(): Promise<void> {
    try {
      await apiClient.post('/auth/logout');
      localStorage.removeItem('authToken');
    } catch (error) {
      console.error('Error during logout:', error);
      // Still remove token even if logout API fails
      localStorage.removeItem('authToken');
    }
  }

  // Health and monitoring methods
  static async getHealthStatus(): Promise<any> {
    try {
      const response = await apiClient.get('/health/detailed', { 
        baseURL: import.meta.env.VITE_API_BASE_URL?.replace('/api/v1', '') || 'http://localhost:3000'
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching health status:', error);
      throw new Error('Failed to fetch system health status');
    }
  }

  // Generic form submission for Framer components
  static async submitForm(endpoint: string, formData: Record<string, any>): Promise<any> {
    try {
      const response = await apiClient.post<ApiResponse<any>>(endpoint, formData);
      return response.data.data;
    } catch (error) {
      console.error(`Error submitting form to ${endpoint}:`, error);
      throw new Error('Form submission failed');
    }
  }
}

export default ApiService;