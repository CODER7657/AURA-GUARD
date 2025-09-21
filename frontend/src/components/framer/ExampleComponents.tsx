// Example Framer component that would be exported from your Framer project
// This is a placeholder showing how to structure exported components

import React from 'react';
import { motion } from 'framer-motion';

interface AirQualityCardProps {
  location?: string;
  aqi?: number;
  status?: string;
  onLocationClick?: () => void;
  className?: string;
}

// This would typically be exported from Framer using the React Export plugin
export const AirQualityCard: React.FC<AirQualityCardProps> = ({
  location = "Sample Location",
  aqi = 0,
  status = "Good",
  onLocationClick,
  className = ""
}) => {
  const getAQIColor = (aqiValue: number) => {
    if (aqiValue <= 50) return "bg-green-500";
    if (aqiValue <= 100) return "bg-yellow-500";
    if (aqiValue <= 150) return "bg-orange-500";
    if (aqiValue <= 200) return "bg-red-500";
    if (aqiValue <= 300) return "bg-purple-500";
    return "bg-red-800";
  };

  return (
    <motion.div
      className={`bg-white rounded-lg shadow-lg p-6 cursor-pointer transition-transform hover:scale-105 ${className}`}
      onClick={onLocationClick}
      whileHover={{ y: -5 }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-lg font-semibold text-gray-800">{location}</h3>
        <div className={`w-4 h-4 rounded-full ${getAQIColor(aqi)}`} />
      </div>
      
      <div className="mb-2">
        <span className="text-3xl font-bold text-gray-900">{aqi}</span>
        <span className="text-gray-600 ml-2">AQI</span>
      </div>
      
      <p className="text-sm text-gray-600">{status}</p>
    </motion.div>
  );
};

// Example form component that might be exported from Framer
interface LocationSearchFormProps {
  onSubmit?: (location: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  className?: string;
}

export const LocationSearchForm: React.FC<LocationSearchFormProps> = ({
  onSubmit,
  isLoading = false,
  placeholder = "Enter location...",
  className = ""
}) => {
  const [location, setLocation] = React.useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (location.trim() && onSubmit) {
      onSubmit(location.trim());
    }
  };

  return (
    <motion.form
      onSubmit={handleSubmit}
      className={`bg-white rounded-lg shadow-md p-6 ${className}`}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
    >
      <div className="flex gap-2">
        <input
          type="text"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          placeholder={placeholder}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <motion.button
          type="submit"
          disabled={isLoading || !location.trim()}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isLoading ? 'Searching...' : 'Search'}
        </motion.button>
      </div>
    </motion.form>
  );
};

// Example dashboard component
interface DashboardLayoutProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  title = "Air Quality Dashboard",
  className = ""
}) => {
  return (
    <motion.div
      className={`min-h-screen bg-gray-50 ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-gray-900">{title}</h1>
            <div className="flex items-center space-x-4">
              {/* Add navigation or user menu here */}
            </div>
          </div>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </motion.div>
  );
};