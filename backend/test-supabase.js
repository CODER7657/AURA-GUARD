require('dotenv').config();
const { Client } = require('pg');

async function testSupabaseConnection() {
  const connectionString = process.env.POSTGRESQL_URI;
  
  console.log('🧪 Testing Supabase PostgreSQL connection...');
  console.log('Connection string:', connectionString.replace(/:[^:]*@/, ':***@'));
  
  const client = new Client({
    connectionString: connectionString,
    ssl: {
      rejectUnauthorized: false // Accept self-signed certificates from Supabase
    }
  });

  try {
    await client.connect();
    console.log('✅ Connected to Supabase PostgreSQL successfully!');
    
    const result = await client.query('SELECT version()');
    console.log('Database version:', result.rows[0].version);
    
    await client.end();
    console.log('✅ Connection test completed successfully!');
  } catch (error) {
    console.error('❌ Connection failed:', error.message);
    console.error('Error code:', error.code);
    console.error('Error details:', error);
  }
}

testSupabaseConnection();