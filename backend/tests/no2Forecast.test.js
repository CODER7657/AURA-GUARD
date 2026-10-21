const express = require('express');
const request = require('supertest');

// The router's other dependencies (AirNow, Redis, the simulated AQI service) are not
// needed for the NO2 endpoints, so they are stubbed and nothing touches the network.
jest.mock('../src/services/ai/NASATempoAIService', () => jest.fn().mockImplementation(() => ({
  getModelPerformance: () => ({}),
})));
jest.mock('../src/services/ai/ForecastAnalystService', () => ({
  askAnalyst: jest.fn(async (q) => ({
    answer: `stub answer to: ${q}`, source: 'first', grounded: true, latency_s: 0.1,
    tool_calls: [{ name: 'list_stations', args: {} }, { name: 'list_stations', args: {}, by: 'fallback' }],
  })),
}));
jest.mock('../src/utils/logger', () => ({ info: jest.fn(), error: jest.fn(), warn: jest.fn() }));

const { NO2ForecastService, haversineKm } = require('../src/services/ai/NO2ForecastService');
const predictions = require('../src/routes/predictions');

const app = express();
app.use(express.json());
app.use('/api/v1/predictions', predictions);

describe('NO2ForecastService', () => {
  const service = new NO2ForecastService();

  it('loads the committed forecasts file', () => {
    expect(service.data.stations.length).toBeGreaterThan(0);
    expect(service.data.status).toMatch(/not live/);
  });

  it('finds the nearest station and refuses points far from every station', () => {
    const chicago = service.nearest(41.88, -87.63);
    expect(chicago.forecast.place).toBe('Cook, Illinois');
    expect(chicago.distance_km).toBeLessThan(80);
    expect(service.nearest(29.76, -95.37)).toBeNull(); // Houston: no station
  });

  it('computes great-circle distance', () => {
    expect(haversineKm(0, 0, 0, 1)).toBeCloseTo(111.2, 0);
  });
});

describe('NO2 routes', () => {
  it('lists stations without the forecast arrays', async () => {
    const res = await request(app).get('/api/v1/predictions/no2/stations').expect(200);
    expect(res.body.data.stations[0]).not.toHaveProperty('forecasts');
  });

  it('returns a forecast with 90% intervals by station id', async () => {
    const id = new NO2ForecastService().data.stations[0].station;
    const res = await request(app).get(`/api/v1/predictions/no2/forecast?station=${id}`).expect(200);
    const f = res.body.data.forecast.forecasts[0];
    expect(f).toHaveProperty('interval90_low_ppb');
    expect(f.interval90_low_ppb).toBeLessThanOrEqual(f.no2_ppb);
  });

  it('404s for an unknown station and 400s without a location', async () => {
    await request(app).get('/api/v1/predictions/no2/forecast?station=00-000-0000-0').expect(404);
    await request(app).get('/api/v1/predictions/no2/forecast').expect(400);
  });

  it('proxies questions to the analyst and hides fallback-only tool calls', async () => {
    const res = await request(app).post('/api/v1/predictions/ask').send({ question: 'Hi?' }).expect(200);
    expect(res.body.data.answer).toBe('stub answer to: Hi?');
    expect(res.body.data.tool_calls).toHaveLength(1);
    await request(app).post('/api/v1/predictions/ask').send({}).expect(400);
  });
});
