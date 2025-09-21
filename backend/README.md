# NASA Air Quality Forecasting Backend

A Node.js backend API for NASA Air Quality Forecasting application integrating TEMPO satellite data and EPA air quality monitoring.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env

# Start development server
npm run dev

# Run tests
npm test
```

## 📁 Project Structure

```
backend/
├── src/
│   ├── controllers/     # Route controllers
│   ├── middleware/      # Custom middleware
│   ├── models/         # Database models
│   ├── routes/         # Express routes
│   ├── services/       # Business logic
│   ├── utils/          # Utility functions
│   ├── config/         # Configuration files
│   └── app.js          # Main application file
├── tests/              # Test files
├── docs/              # API documentation
└── scripts/           # Utility scripts
```

## 🔧 Technology Stack

- **Runtime**: Node.js 18+
- **Framework**: Express.js
- **Database**: PostgreSQL with Sequelize ORM
- **Cache**: Redis
- **Authentication**: JWT + Passport.js
- **Testing**: Jest + Supertest
- **Documentation**: Swagger/OpenAPI

## 📊 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh token

### Air Quality Data
- `GET /api/v1/air-quality/current` - Current air quality data
- `GET /api/v1/air-quality/historical` - Historical data queries
- `GET /api/v1/air-quality/location/:coords` - Location-based data

### Predictions
- `POST /api/v1/predictions/realtime` - Real-time air quality prediction
- `POST /api/v1/predictions/forecast` - Multi-hour forecasting

### Notifications
- `POST /api/v1/notifications/subscribe` - Subscribe to notifications
- `GET /api/v1/notifications/preferences` - Get user preferences

## 🌍 Environment Variables

See `.env.example` for required environment variables.

## 📝 License

MIT License - see LICENSE file for details.