const request = require('supertest');
const app = require('../src/app');

describe('Health Check Endpoints', () => {
  describe('GET /health', () => {
    it('should return 200 status and basic health info', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.body).toHaveProperty('status', 'OK');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('uptime');
      expect(response.body).toHaveProperty('version');
      expect(response.body).toHaveProperty('environment');
    });
  });

  describe('GET /health/detailed', () => {
    it('should return detailed health information', async () => {
      const response = await request(app)
        .get('/health/detailed')
        .expect(200);

      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('services');
      expect(response.body).toHaveProperty('memory');
      expect(response.body).toHaveProperty('cpu');
      expect(response.body.services).toHaveProperty('mongodb');
      expect(response.body.services).toHaveProperty('redis');
    });
  });
});

describe('API Documentation', () => {
  describe('GET /api/docs', () => {
    it('should return API documentation', async () => {
      const response = await request(app)
        .get('/api/docs')
        .expect(200);

      expect(response.body).toHaveProperty('message');
      expect(response.body).toHaveProperty('version');
      expect(response.body).toHaveProperty('endpoints');
      expect(response.body.endpoints).toHaveProperty('auth');
      expect(response.body.endpoints).toHaveProperty('airQuality');
      expect(response.body.endpoints).toHaveProperty('predictions');
      expect(response.body.endpoints).toHaveProperty('notifications');
    });
  });
});

describe('404 Not Found', () => {
  it('should return 404 for non-existent routes', async () => {
    const response = await request(app)
      .get('/non-existent-route')
      .expect(404);

    expect(response.body).toHaveProperty('success', false);
    expect(response.body).toHaveProperty('error');
    expect(response.body.error).toHaveProperty('statusCode', 404);
    expect(response.body).toHaveProperty('availableEndpoints');
  });
});