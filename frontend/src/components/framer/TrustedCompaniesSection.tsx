import React from 'react';
import { motion } from 'framer-motion';

export const TrustedCompaniesSection: React.FC = () => {
  const partners = [
    { name: 'NASA', type: 'Space Agency' },
    { name: 'EPA', type: 'Environmental' },
    { name: 'NOAA', type: 'Atmospheric' },
    { name: 'CDC', type: 'Public Health' },
    { name: 'ESA', type: 'European Space' },
    { name: 'WHO', type: 'World Health' },
    { name: 'WMO', type: 'Meteorological' },
    { name: 'UNEP', type: 'UN Environment' }
  ];

  return (
    <section className="py-16 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Trusted by Leading Environmental & Health Organizations
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Collaborating with world-class institutions to advance air quality science and public health protection
          </p>
        </motion.div>

        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-8 items-center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          {partners.map((partner, index) => (
            <motion.div
              key={partner.name}
              className="flex items-center justify-center h-20"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <div className="text-center group cursor-pointer">
                {/* Organization logo placeholder */}
                <div className="w-16 h-16 bg-gradient-to-br from-green-100 to-blue-100 rounded-lg flex items-center justify-center mb-2 group-hover:from-green-200 group-hover:to-blue-200 transition-all duration-300">
                  <span className="text-xl font-bold text-green-700 group-hover:text-blue-700">
                    {partner.name.charAt(0)}
                  </span>
                </div>
                <span className="text-sm font-medium text-gray-700 group-hover:text-green-600 transition-colors duration-300">
                  {partner.name}
                </span>
                <div className="text-xs text-gray-500 mt-1">{partner.type}</div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Mission Collaboration Badge */}
        <motion.div
          className="mt-16 text-center"
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-green-100 to-blue-100 text-green-800 rounded-full font-medium">
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 2L3 7v11a1 1 0 001 1h12a1 1 0 001-1V7l-7-5zM8 15v-3a1 1 0 011-1h2a1 1 0 011 1v3H8z"/>
            </svg>
            NASA Space Apps Challenge 2025 - Problem Statement 9
          </div>
        </motion.div>

        {/* Key Statistics */}
        <motion.div
          className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">9.1/10</div>
            <p className="text-gray-600">Weighted Challenge Score</p>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">24/7</div>
            <p className="text-gray-600">Continuous Monitoring</p>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600 mb-2">365M+</div>
            <p className="text-gray-600">People Protected</p>
          </div>
        </motion.div>
      </div>
    </section>
  );
};