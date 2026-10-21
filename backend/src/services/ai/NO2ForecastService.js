/**
 * Real NO2 forecasts from the trained gradient-boosting models.
 *
 * Reads ml/results/forecasts.json, which ml/scripts/deploy_forecaster.py writes. The
 * forecasts are issued from the last hour in the newest EPA AirData file, so they are
 * real model output but not live: every response carries `issued_at_utc` and
 * `status` saying so.
 */

const path = require('path');
const { readFileSync } = require('fs');

const FORECASTS_FILE = path.join(__dirname, '../../../../ml/results/forecasts.json');

function haversineKm(lat1, lon1, lat2, lon2) {
  const rad = (d) => (d * Math.PI) / 180;
  const a = Math.sin(rad(lat2 - lat1) / 2) ** 2
    + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(a));
}

class NO2ForecastService {
  constructor(file = FORECASTS_FILE) {
    try {
      this.data = JSON.parse(readFileSync(file, 'utf8'));
    } catch (error) {
      this.data = { status: 'no forecasts file found; run ml/scripts/deploy_forecaster.py', stations: [] };
    }
    this.byStation = new Map(this.data.stations.map((s) => [s.station, s]));
  }

  header() {
    const { stations, ...rest } = this.data;
    return rest;
  }

  listStations() {
    return {
      ...this.header(),
      stations: this.data.stations.map(({ forecasts, ...s }) => s),
    };
  }

  getStation(id) {
    const s = this.byStation.get(id);
    return s ? { ...this.header(), forecast: s } : null;
  }

  /** The station closest to a point, with the distance, or null if none within maxKm. */
  nearest(latitude, longitude, maxKm = 80) {
    let best = null;
    for (const s of this.data.stations) {
      if (typeof s.lat !== 'number') continue;
      const d = haversineKm(latitude, longitude, s.lat, s.lon);
      if (!best || d < best.distance_km) best = { distance_km: Math.round(d * 10) / 10, station: s };
    }
    if (!best || best.distance_km > maxKm) return null;
    return { ...this.header(), distance_km: best.distance_km, max_distance_km: maxKm, forecast: best.station };
  }
}

module.exports = { NO2ForecastService, haversineKm };
