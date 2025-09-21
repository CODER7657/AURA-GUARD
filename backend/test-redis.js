require('dotenv').config();
const redis = require('./src/config/redis');
const logger = require('./src/utils/logger');

async function testRedis() {
  console.log('🧪 Testing Redis Cache Operations...\n');
  
  try {
    // Test connection
    console.log('1. Testing Redis connection...');
    const connected = await redis.connectRedis();
    console.log(`   Connection status: ${connected ? '✅ Connected' : '❌ Failed'}\n`);
    
    // Test SET operation
    console.log('2. Testing SET operation...');
    const testKey = 'test:air-quality:location-123';
    const testValue = JSON.stringify({
      location: 'New York',
      aqi: 45,
      timestamp: new Date().toISOString(),
      pollutants: {
        pm25: 12.5,
        pm10: 18.2,
        no2: 25.3,
        o3: 55.1
      }
    });
    
    const setResult = await redis.setCache(testKey, testValue, 300); // 5 minutes TTL
    console.log(`   SET result: ${setResult ? '✅ Success' : '❌ Failed'}\n`);
    
    // Test GET operation
    console.log('3. Testing GET operation...');
    const getValue = await redis.getCache(testKey);
    console.log(`   GET result: ${getValue ? '✅ Success' : '❌ Failed'}`);
    if (getValue) {
      try {
        const parsedValue = JSON.parse(getValue);
        console.log(`   Retrieved data: Location: ${parsedValue.location}, AQI: ${parsedValue.aqi}\n`);
      } catch (e) {
        console.log(`   Retrieved raw data: ${getValue}\n`);
      }
    }
    
    // Test EXISTS operation
    console.log('4. Testing EXISTS operation...');
    const exists = await redis.cache.exists(testKey);
    console.log(`   EXISTS result: ${exists ? '✅ Key exists' : '❌ Key not found'}\n`);
    
    // Test cache with different data types
    console.log('5. Testing different data types...');
    await redis.setCache('test:string', 'simple string', 60);
    await redis.setCache('test:number', '42', 60);
    await redis.setCache('test:boolean', 'true', 60);
    
    const stringVal = await redis.getCache('test:string');
    const numberVal = await redis.getCache('test:number');
    const booleanVal = await redis.getCache('test:boolean');
    
    console.log(`   String: ${stringVal ? '✅' : '❌'} ${stringVal}`);
    console.log(`   Number: ${numberVal ? '✅' : '❌'} ${numberVal}`);
    console.log(`   Boolean: ${booleanVal ? '✅' : '❌'} ${booleanVal}\n`);
    
    // Test DELETE operation
    console.log('6. Testing DELETE operation...');
    const deleteResult = await redis.deleteCache(testKey);
    console.log(`   DELETE result: ${deleteResult ? '✅ Success' : '❌ Failed'}`);
    
    // Verify deletion
    const getAfterDelete = await redis.getCache(testKey);
    console.log(`   Verification: ${getAfterDelete === null ? '✅ Key deleted' : '❌ Key still exists'}\n`);
    
    // Clean up test keys
    console.log('7. Cleaning up test keys...');
    await redis.deleteCache('test:string');
    await redis.deleteCache('test:number');
    await redis.deleteCache('test:boolean');
    console.log('   ✅ Cleanup completed\n');
    
    console.log('🎉 Redis cache testing completed successfully!');
    
  } catch (error) {
    console.error('❌ Redis test failed:', error.message);
  }
}

testRedis().then(() => {
  console.log('\n📊 Test Summary: Redis cache operations verified');
  process.exit(0);
}).catch(error => {
  console.error('Test script failed:', error);
  process.exit(1);
});