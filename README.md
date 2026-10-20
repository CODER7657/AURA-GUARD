# AURA-GUARD: Air Quality Monitoring and NO₂ Forecasting

> Built for the **NASA Space Apps Challenge 2025**, Problem Statement 9: Air Quality Forecasting.

[![ml tests](https://github.com/CODER7657/AURA-GUARD/actions/workflows/ml-tests.yml/badge.svg)](https://github.com/CODER7657/AURA-GUARD/actions/workflows/ml-tests.yml)
[![Open backtest in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CODER7657/AURA-GUARD/blob/master/ml/notebooks/backtest_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A full-stack air quality app (React + Node.js) with an NO₂ forecaster that is
backtested on a year of real EPA monitoring data.

## What is real, and what is not yet

| Part | Status |
|---|---|
| Forecasting model ([`ml/`](ml/README.md)) | **Real.** Trained and scored on EPA AirData hourly NO₂ from 20 US monitors, 2024, with a monthly walk-forward backtest |
| Model performance in the API and dashboard | **Measured.** Served from `ml/results/model_performance.json`, which the backtest writes |
| Current readings | **Real when configured.** EPA AirNow, with an AirNow API key |
| Forecasts shown in the app | **Simulated.** No trained model is deployed yet; every API response says so in `simulated` / `live_forecasts` |
| NASA TEMPO satellite data | **Not integrated yet.** It needs a NASA Earthdata account and is the next step |

An earlier version of this README reported "86.98% accuracy" and "1.7 ms inference"
for an LSTM. Those numbers were written into the code, and the model only ever saw
randomly generated data. They have been removed everywhere.

## Forecasting results

Hourly NO₂ forecasts 1, 6, 12 and 24 hours ahead, scored month by month from July to
December 2024. Each month's model is trained only on data from before that month.

| Horizon | Persistence (next = now) | Gradient boosting | Improvement | 90% interval coverage |
|---|---|---|---|---|
| 1 h | 2.00 ppb | **1.85 ppb** | 7% | 88.4% |
| 6 h | 5.30 ppb | **3.69 ppb** | 30% | 87.6% |
| 12 h | 6.11 ppb | **4.11 ppb** | 33% | 87.5% |
| 24 h | 4.88 ppb | **4.32 ppb** | 11% | 87.6% |

Mean absolute error, lower is better. Persistence alone scores **R² 0.90** one hour
ahead, higher than the 0.87 the old README claimed, which is why every model here is
judged by how much it beats persistence rather than by R².

![Forecast error by horizon](ml/results/mae_by_horizon.png)

Full method, every baseline and the limitations: [`ml/README.md`](ml/README.md).

## System architecture

```
┌──────────────────┐     ┌──────────────────────┐     ┌──────────────────────────┐
│ React + Vite     │────▶│ Node.js + Express    │────▶│ EPA AirNow API           │
│ dashboard        │     │ /api/v1/predictions  │     │ (current readings)       │
└──────────────────┘     └──────────┬───────────┘     └──────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐     ┌──────────────────────────┐
                         │ Python bridge        │◀────│ ml/ forecaster           │
                         │ (simulated forecasts │     │ backtested on EPA data;  │
                         │  until deployment)   │     │ results in ml/results/   │
                         └──────────────────────┘     └──────────────────────────┘
```

## Technology stack

- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS
- **Backend:** Node.js, Express, PostgreSQL, Redis
- **Forecasting:** Python, pandas, scikit-learn (gradient boosting, ridge), split-conformal prediction intervals
- **Data:** EPA AirData (training and backtest), EPA AirNow (current readings)

More detail: [TECH_STACK.md](TECH_STACK.md).

## Quick Start

### Prerequisites

- Node.js 18.0+ (recommended: 22.17.0)
- Python 3.10+ (for AI/ML components)
- PostgreSQL database
- Redis server

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/CODER7657/AURA-GUARD.git
   cd AURA-GUARD
   ```

2. **Backend Setup**
   ```bash
   cd backend
   npm install
   cp .env.example .env
   # Configure your environment variables in .env
   npm start
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   cp .env.example .env.local
   # Configure your environment variables in .env.local
   npm run dev
   ```

4. **AI/ML Setup**
   ```bash
   cd ai-int
   pip install -r requirements.txt
   python models/lstm_air_quality.py
   ```

### Environment Configuration

Create `.env` files with your API keys:

**Backend (.env)**
```bash
NODE_ENV=development
PORT=3000

# Database

POSTGRESQL_URI=your_postgresql_connection_string
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_api_key

# Redis

UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token

# API Keys

TEMPO_API_KEY=your_tempo_api_key
AIRNOW_API_KEY=your_airnow_api_key
OPENWEATHERMAP_API_KEY=your_openweather_api_key

# Authentication

JWT_SECRET=your_jwt_secret_key
JWT_REFRESH_SECRET=your_refresh_token_secret
```

**Frontend (.env.local)**
```bash
VITE_API_BASE_URL=http://localhost:3000/api/v1
VITE_NASA_API_KEY=your_nasa_api_key
VITE_OPENWEATHER_API_KEY=your_openweather_api_key
VITE_DEBUG_MODE=true
```

---


---

## Project Structure

```
AURA-GUARD/
├── frontend/          React dashboard
├── backend/           Express API (routes, services, models, middleware)
├── ml/                NO2 forecaster: data loading, features, models, backtest
│   ├── notebooks/     Colab notebook that runs the full backtest
│   ├── results/       Measured results, charts, model_performance.json
│   └── tests/         Light tests, no data download needed
├── ai-int/models/     Python bridge the backend calls (simulated forecasts)
├── TECH_STACK.md
└── README.md
```

---

## API Documentation

### Core Endpoints

#### **Real-time Predictions**
```http
POST /api/v1/predictions/realtime
Content-Type: application/json

{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "forecast_hours": 1
}
```

#### **Extended Forecasting**
```http
POST /api/v1/predictions/forecast
Content-Type: application/json

{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "duration": 48
}
```

#### **Model Performance**
Returns the measured backtest results from `ml/results/model_performance.json`.
```http
GET /api/v1/predictions/accuracy
```

#### **System Health**
```http
GET /api/v1/health/detailed
```

---

## Security Features

- **🔐 JWT Authentication**: Secure user authentication with refresh tokens
- **🛡️ Rate Limiting**: API abuse prevention with configurable limits
- **🌐 CORS Protection**: Cross-origin request security
- **🔒 Input Validation**: Comprehensive data sanitization
- **🗄️ SQL Injection Prevention**: Parameterized database queries
- **🔑 Environment Variables**: Secure configuration management

---

## Contributing

We welcome contributions to AURA-GUARD! Here's how to get started:

### Development Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- **Frontend**: ESLint + Prettier with TypeScript strict mode
- **Backend**: ESLint with Airbnb configuration
- **AI/ML**: PEP 8 Python style guide
- **Testing**: Jest for JavaScript, pytest for Python

---


## Roadmap

1. **Deploy the forecaster.** Export the trained model, give the bridge the last 7 days of readings for a station, and replace the simulated forecasts.
2. **Add NASA TEMPO.** Use TEMPO NO₂ columns (NASA Earthdata) as an extra feature and measure whether they improve the backtest.
3. **More data.** More monitors, more years, and weather inputs (wind, boundary-layer height).

## License

MIT, see [LICENSE](LICENSE). EPA AirData and AirNow data are public, from the
U.S. Environmental Protection Agency.
