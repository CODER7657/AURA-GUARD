import React from 'react';
import { motion } from 'framer-motion';

export const TestimonialsSection: React.FC = () => {
  const testimonials = [
    {
      name: "Dr. Sarah Chen",
      role: "Environmental Scientist, EPA",
      content: "AeroGuard's TEMPO integration provides unparalleled accuracy in our air quality assessments. It's revolutionizing how we protect public health.",
      avatar: "SC"
    },
    {
      name: "Dr. Michael Rodriguez", 
      role: "Public Health Director",
      content: "The real-time health alerts have helped us prevent respiratory emergencies in vulnerable populations. This platform saves lives.",
      avatar: "MR"
    },
    {
      name: "Prof. Emily Watson",
      role: "Atmospheric Researcher", 
      content: "Having hourly satellite data integrated with ground measurements gives us insights we never had before. It's a game-changer for research.",
      avatar: "EW"
    },
    {
      name: "James Parker",
      role: "City Environmental Planner",
      content: "AeroGuard helps us make data-driven decisions for urban planning. The 7-day forecasts are incredibly accurate for policy planning.",
      avatar: "JP"
    },
    {
      name: "Dr. Lisa Thompson",
      role: "NASA Atmospheric Scientist",
      content: "The seamless integration of TEMPO data with ground-based measurements creates the most comprehensive air quality picture available.",
      avatar: "LT"
    },
    {
      name: "Maria Gonzalez",
      role: "School District Health Coordinator",
      content: "We use AeroGuard to make real-time decisions about outdoor activities. The health impact predictions are spot-on.",
      avatar: "MG"
    }
  ];

  return (
    <section className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Trusted by Environmental & Health Professionals
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Real stories from scientists, researchers, and public health officials using AeroGuard to protect communities worldwide.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ y: -5 }}
            >
              {/* Stars */}
              <div className="flex items-center mb-4">
                {[...Array(5)].map((_, i) => (
                  <svg
                    key={i}
                    className="w-5 h-5 text-green-400"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                ))}
              </div>

              {/* Testimonial Content */}
              <p className="text-gray-700 mb-6 leading-relaxed">
                "{testimonial.content}"
              </p>

              {/* Author */}
              <div className="flex items-center">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-blue-600 rounded-full flex items-center justify-center text-white font-bold mr-4">
                  {testimonial.avatar}
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">{testimonial.name}</h4>
                  <p className="text-sm text-gray-600">{testimonial.role}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Air Quality Impact Grid */}
        <motion.div
          className="mt-20 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <h3 className="text-3xl font-bold text-gray-900 mb-8 lg:col-span-3 text-center">
            Advanced Air Quality Intelligence for Everyone
          </h3>

          {[
            {
              title: "TEMPO Satellite Integration",
              description: "First-ever geostationary satellite providing hourly air quality data across North America.",
              icon: "🛰️"
            },
            {
              title: "Multi-Pollutant Monitoring",
              description: "Track NO₂, O₃, SO₂, aerosols, and formaldehyde simultaneously with precision.",
              icon: "🧪"
            },
            {
              title: "Predictive Forecasting",
              description: "AI-powered models predict air quality up to 7 days ahead with 95% accuracy.",
              icon: "�"
            },
            {
              title: "Health Impact Alerts",
              description: "Automated notifications protect sensitive groups from harmful air quality events.",
              icon: "🏥"
            },
            {
              title: "Real-Time Data Fusion",
              description: "Seamlessly combines satellite and ground-based measurements for comprehensive coverage.",
              icon: "�"
            },
            {
              title: "Public Health Protection",
              description: "Early warning systems safeguard communities and vulnerable populations worldwide.",
              icon: "�️"
            }
          ].map((feature, index) => (
            <motion.div
              key={index}
              className="text-center p-6"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h4 className="text-xl font-bold text-gray-900 mb-3">{feature.title}</h4>
              <p className="text-gray-600 leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};