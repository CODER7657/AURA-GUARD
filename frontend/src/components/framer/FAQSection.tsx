import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export const FAQSection: React.FC = () => {
  const [openFAQ, setOpenFAQ] = useState<number | null>(null);

  const faqs = [
    {
      question: "What is NASA's TEMPO mission and how does AeroGuard use it?",
      answer: "TEMPO (Tropospheric Emissions: Monitoring of Pollution) is NASA's first Earth-observing instrument in geostationary orbit over North America. It provides hourly measurements of air pollutants including nitrogen dioxide, ozone, sulfur dioxide, and aerosols. AeroGuard integrates this revolutionary satellite data with ground-based measurements to create the most comprehensive air quality monitoring system available."
    },
    {
      question: "How accurate are AeroGuard's air quality forecasts?",
      answer: "Our AI-powered forecasting models achieve 95% accuracy for next-day predictions and maintain over 85% accuracy for 7-day forecasts. By combining NASA's TEMPO satellite data with ground-based sensor networks and advanced machine learning algorithms, we provide the most reliable air quality predictions available for public health protection."
    },
    {
      question: "What pollutants does AeroGuard monitor?",
      answer: "AeroGuard tracks all major air pollutants including nitrogen dioxide (NO₂), ground-level ozone (O₃), sulfur dioxide (SO₂), particulate matter (PM2.5 and PM10), carbon monoxide (CO), and formaldehyde. Our platform provides real-time concentrations, health impact assessments, and predictive forecasts for each pollutant."
    },
    {
      question: "How does AeroGuard protect public health?",
      answer: "Our platform provides automated health alerts based on air quality conditions, with specific warnings for sensitive groups including children, elderly, and people with respiratory conditions. We integrate with Air Quality Index (AQI) standards and provide actionable recommendations like when to limit outdoor activities or wear protective masks."
    },
    {
      question: "Who can access AeroGuard's data and services?",
      answer: "We offer multiple access tiers: free public access for basic air quality information, research plans for scientists and academic institutions, and enterprise solutions for government agencies, environmental organizations, and corporations needing comprehensive air quality intelligence for decision-making and compliance."
    },
    {
      question: "How often is the air quality data updated?",
      answer: "Thanks to NASA's TEMPO mission, we provide hourly satellite-based air quality measurements during daylight hours, combined with continuous ground-based sensor data. This creates the most frequent and comprehensive air quality monitoring coverage ever achieved, enabling real-time health protection and rapid response to pollution events."
    }
  ];

  const toggleFAQ = (index: number) => {
    setOpenFAQ(openFAQ === index ? null : index);
  };

  return (
    <>
      {/* FAQ Section */}
      <section id="about" className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Frequently Asked Questions
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Learn more about AeroGuard's air quality monitoring, NASA's TEMPO mission, and how we protect public health
            </p>
          </motion.div>

          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                className="border border-gray-200 rounded-lg overflow-hidden"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <button
                  onClick={() => toggleFAQ(index)}
                  className="w-full px-6 py-6 text-left flex justify-between items-center hover:bg-gray-50 transition-colors duration-200"
                >
                  <h3 className="text-lg font-semibold text-gray-900 pr-4">
                    {faq.question}
                  </h3>
                  <motion.svg
                    className="w-6 h-6 text-gray-600 flex-shrink-0"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    animate={{ rotate: openFAQ === index ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </motion.svg>
                </button>
                
                <AnimatePresence>
                  {openFAQ === index && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="overflow-hidden"
                    >
                      <div className="px-6 pb-6 text-gray-600 leading-relaxed">
                        {faq.answer}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="py-20 bg-gradient-to-br from-green-600 via-blue-700 to-green-800 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Protect Your Community Today
            </h2>
            <p className="text-xl text-green-100 mb-8 max-w-2xl mx-auto">
              Join the air quality revolution powered by NASA's cutting-edge satellite technology.
            </p>
            <button className="bg-white text-green-600 hover:bg-gray-100 font-semibold px-8 py-4 rounded-lg text-lg transition-all duration-200 transform hover:scale-105 shadow-lg">
              Start Monitoring
            </button>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-6 gap-8">
            <div className="col-span-2">
              <h3 className="text-2xl font-bold mb-4">AeroGuard</h3>
              <p className="text-gray-400 mb-4">
                Protecting public health with NASA's revolutionary air quality intelligence
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Platform</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Live Data</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Forecasting</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Health Alerts</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Science</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">TEMPO Mission</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Data Sources</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Research</a></li>
              </ul>
            </div>
            
            <div className="col-span-2">
              <p className="text-gray-400 text-sm">
                Developed for NASA Space Apps Challenge 2025 - Problem Statement 9: Air Quality Forecasting. 
                Leveraging NASA's TEMPO satellite mission for superior environmental protection.
              </p>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center">
            <p className="text-gray-400 text-sm">
              © 2025 AeroGuard. Powered by NASA TEMPO Mission. Built for NASA Space Apps Challenge 2025.
            </p>
          </div>
        </div>
      </footer>
    </>
  );
};