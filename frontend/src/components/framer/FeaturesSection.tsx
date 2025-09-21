import React from 'react';
import { motion } from 'framer-motion';

export const FeaturesSection: React.FC = () => {
  return (
    <>
      {/* TEMPO Mission Integration Section */}
      <section id="tempo" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                Real-Time Satellite Data
              </h2>
              <p className="text-xl text-gray-600 mb-8 leading-relaxed">
                Powered by NASA's TEMPO mission, we deliver unprecedented hourly air quality measurements from space, covering North America with unmatched precision and coverage.
              </p>
              <button className="bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 text-white font-semibold px-6 py-3 rounded-lg transition-all duration-200 transform hover:scale-105">
                Explore TEMPO Data
              </button>
            </motion.div>
            
            <motion.div
              className="relative"
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              {/* Satellite visualization */}
              <div className="bg-gradient-to-br from-blue-50 to-green-50 rounded-2xl p-8 h-80 flex items-center justify-center">
                <div className="text-center">
                  <div className="w-24 h-24 bg-gradient-to-r from-green-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-12 h-12 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 2L3 7v11a1 1 0 001 1h12a1 1 0 001-1V7l-7-5zM8 15v-3a1 1 0 011-1h2a1 1 0 011 1v3H8z"/>
                    </svg>
                  </div>
                  <p className="text-gray-600 font-medium">TEMPO Satellite Monitoring</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Core Features Section */}
      <section id="forecasting" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Advanced Air Quality Intelligence
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Integrating NASA's revolutionary TEMPO satellite data with ground-based measurements for superior forecasting accuracy
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                title: "Real-Time Monitoring",
                description: "Hourly satellite measurements of NO₂, O₃, SO₂, and aerosols across North America.",
                icon: "🛰️"
              },
              {
                title: "Predictive Forecasting",
                description: "AI-powered models predict air quality up to 7 days ahead with 95% accuracy.",
                icon: "�"
              },
              {
                title: "Health Impact Alerts",
                description: "Automated notifications for sensitive groups and public health advisories.",
                icon: "🏥"
              },
              {
                title: "Data Integration",
                description: "Seamlessly combines satellite data with ground sensors for comprehensive coverage.",
                icon: "�"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="bg-white rounded-xl p-8 shadow-lg hover:shadow-xl transition-all duration-300"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -5 }}
              >
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Capabilities Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Cutting-Edge Environmental Technology
            </h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: "TEMPO Mission Integration",
                description: "First-ever geostationary satellite monitoring air quality over North America hourly.",
                icon: "🌍"
              },
              {
                title: "Multi-Pollutant Tracking",
                description: "Monitor nitrogen dioxide, ozone, sulfur dioxide, aerosols, and formaldehyde simultaneously.",
                icon: "🧪"
              },
              {
                title: "Public Health Protection",
                description: "Early warning systems protect vulnerable populations from harmful air quality events.",
                icon: "🛡️"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="text-center p-8"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
              >
                <div className="text-5xl mb-6">{feature.icon}</div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                <p className="text-gray-600 mb-6 leading-relaxed">{feature.description}</p>
                <button className="text-green-600 hover:text-green-700 font-semibold hover:underline transition-colors duration-200">
                  Learn More
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
};