// Global test setup
process.env.NODE_ENV = 'test';
process.env.MONGODB_TEST_URI = 'mongodb://localhost:27017/nasa-air-quality-test';
process.env.REDIS_URL = 'redis://localhost:6379/1';
process.env.JWT_SECRET = 'test-secret-key';
process.env.JWT_REFRESH_SECRET = 'test-refresh-secret';

// Suppress console logs during testing unless explicitly needed
if (!process.env.DEBUG_TESTS) {
  console.log = jest.fn();
  console.info = jest.fn();
  console.warn = jest.fn();
  console.error = jest.fn();
}

// Global test timeout
jest.setTimeout(10000);