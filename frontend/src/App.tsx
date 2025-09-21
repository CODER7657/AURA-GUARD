import React from 'react';
import { ThemeProvider } from './contexts/ThemeContext';
import { Navbar } from './components/framer/Navbar';
import { HeroSection } from './components/framer/HeroSection';
import { TrustedCompaniesSection } from './components/framer/TrustedCompaniesSection';
import { FeaturesSection } from './components/framer/FeaturesSection';
import { AirQualityDashboard } from './components/AirQualityDashboard';
import NASATempoPredictor from './components/NASATempoPredictor';
import { PricingSection } from './components/framer/PricingSection';
import { TestimonialsSection } from './components/framer/TestimonialsSection';
import { FAQSection } from './components/framer/FAQSection';
import './App.css';

function App() {
  return (
    <ThemeProvider>
      <div className="App">
        <Navbar />
        <HeroSection />
        <TrustedCompaniesSection />
        <FeaturesSection />
        {/* NASA TEMPO Enhanced LSTM Predictor */}
        <section className="py-12 bg-gray-50 dark:bg-gray-900">
          <NASATempoPredictor />
        </section>
        <AirQualityDashboard />
        <PricingSection />
        <TestimonialsSection />
        <FAQSection />
      </div>
    </ThemeProvider>
  );
}

export default App;
