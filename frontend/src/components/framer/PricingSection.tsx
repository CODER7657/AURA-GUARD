import React, { useState } from 'react';
import { motion } from 'framer-motion';

export const PricingSection: React.FC = () => {
  const [isYearly, setIsYearly] = useState(false);

  return (
    <>
      {/* Performance Metrics Section */}
      <section className="py-20 bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Proven Air Quality Intelligence
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Delivering actionable insights with NASA-grade satellite technology and advanced forecasting models
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { value: "95%", label: "Forecast Accuracy" },
              { value: "365M+", label: "People Protected" },
              { value: "24/7", label: "Continuous Monitoring" }
            ].map((metric, index) => (
              <motion.div
                key={index}
                className="text-center"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
              >
                <div className="text-5xl md:text-6xl font-bold text-green-400 mb-4">
                  {metric.value}
                </div>
                <div className="text-xl text-gray-300">{metric.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Data Access Plans Section */}
      <section id="data-plans" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Data Access Plans
            </h2>
            <p className="text-xl text-gray-600 mb-8">
              Choose the plan that meets your air quality monitoring and research needs
            </p>

            {/* Toggle */}
            <div className="flex items-center justify-center space-x-4 mb-12">
              <span className={`font-medium ${!isYearly ? 'text-green-600' : 'text-gray-500'}`}>
                Monthly
              </span>
              <button
                onClick={() => setIsYearly(!isYearly)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  isYearly ? 'bg-green-600' : 'bg-gray-200'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    isYearly ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
              <span className={`font-medium ${isYearly ? 'text-green-600' : 'text-gray-500'}`}>
                Yearly <span className="text-sm text-gray-400">(Save 20%)</span>
              </span>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Public Access Plan */}
            <motion.div
              className="bg-white border-2 border-gray-200 rounded-2xl p-8 hover:border-green-300 transition-all duration-300"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              whileHover={{ y: -5 }}
            >
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Public Access</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold text-gray-900">Free</span>
                <span className="text-gray-600 ml-2">Always</span>
              </div>
              
              <ul className="space-y-4 mb-8">
                {[
                  "Basic air quality index",
                  "Daily forecasts",
                  "Health alerts",
                  "Mobile app access",
                  "Community support"
                ].map((feature, index) => (
                  <li key={index} className="flex items-center">
                    <svg className="w-5 h-5 text-green-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>

              <button className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-lg transition-colors duration-200">
                Get Started Free
              </button>
            </motion.div>

            {/* Research Plan */}
            <motion.div
              className="bg-gradient-to-br from-green-500 to-blue-600 text-white rounded-2xl p-8 relative overflow-hidden"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1 }}
              whileHover={{ y: -5 }}
            >
              <div className="absolute top-4 right-4 bg-yellow-400 text-yellow-900 px-3 py-1 rounded-full text-sm font-semibold">
                Popular
              </div>
              
              <h3 className="text-2xl font-bold mb-4">Research</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold">${isYearly ? '199' : '249'}</span>
                <span className="text-green-100 ml-2">per month</span>
              </div>
              
              <ul className="space-y-4 mb-8">
                {[
                  "Full TEMPO satellite data",
                  "7-day forecasts",
                  "Historical data access",
                  "API integration",
                  "Research support",
                  "Custom alerts"
                ].map((feature, index) => (
                  <li key={index} className="flex items-center">
                    <svg className="w-5 h-5 text-green-300 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>

              <button className="w-full bg-white text-green-600 hover:bg-gray-50 font-semibold py-3 rounded-lg transition-colors duration-200">
                Start Research Plan
              </button>
            </motion.div>

            {/* Enterprise Plan */}
            <motion.div
              className="bg-white border-2 border-gray-200 rounded-2xl p-8 hover:border-blue-300 transition-all duration-300"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
              whileHover={{ y: -5 }}
            >
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Enterprise</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold text-gray-900">Custom</span>
                <span className="text-gray-600 ml-2">Pricing</span>
              </div>
              
              <ul className="space-y-4 mb-8">
                {[
                  "Real-time data streams",
                  "Custom model training",
                  "White-label solutions",
                  "SLA guarantees",
                  "Dedicated support",
                  "On-premise deployment"
                ].map((feature, index) => (
                  <li key={index} className="flex items-center">
                    <svg className="w-5 h-5 text-blue-500 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>

              <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors duration-200">
                Contact Sales
              </button>
            </motion.div>
          </div>

          {/* Additional Info */}
          <motion.div
            className="mt-16 text-center"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <p className="text-gray-600 mb-4">
              All plans include access to our mobile app and basic support
            </p>
            <div className="flex justify-center space-x-8 text-sm text-gray-500">
              <span>✓ 99.9% Uptime SLA</span>
              <span>✓ GDPR Compliant</span>
              <span>✓ 30-day Money Back</span>
            </div>
          </motion.div>
        </div>
      </section>
    </>
  );
};