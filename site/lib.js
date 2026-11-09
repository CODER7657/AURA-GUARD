// Pure helpers for the live forecast page. No DOM here, so they run under `node --test`.

export const SOURCE =
  "https://raw.githubusercontent.com/CODER7657/AURA-GUARD/live-data/ml/live/forecasts.json";
export const STALE_AFTER_H = 3; // the job runs hourly; three missed runs means something is wrong

// "2026-11-07 04:00" (UTC, as the pipeline writes it) -> Date
export function parseUtc(s) {
  if (typeof s !== "string") return null;
  const m = s.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/);
  if (!m) return null;
  const [, y, mo, d, h, mi] = m.map(Number);
  const t = Date.UTC(y, mo - 1, d, h, mi);
  return Number.isNaN(t) ? null : new Date(t);
}

export function ageHours(issuedUtc, now = new Date()) {
  const t = parseUtc(issuedUtc);
  return t ? (now.getTime() - t.getTime()) / 3_600_000 : null;
}

export function ageLabel(hours) {
  if (hours == null || Number.isNaN(hours)) return "unknown";
  if (hours < 1 / 60) return "just now";
  if (hours < 1) return `${Math.round(hours * 60)} min ago`;
  const h = Math.floor(hours);
  return `${h} h ago`;
}

// The page renders only what it can trust; anything else is reported, not guessed.
export function validateDoc(doc) {
  const errors = [];
  if (!doc || typeof doc !== "object") return ["response is not a JSON object"];
  if (!Array.isArray(doc.horizons_h) || doc.horizons_h.length === 0) errors.push("no horizons");
  if (!Array.isArray(doc.stations)) errors.push("no stations list");
  if (!parseUtc(doc.issued_at_utc)) errors.push("missing issue time");
  for (const s of Array.isArray(doc.stations) ? doc.stations : []) {
    if (typeof s?.station !== "string" || !Array.isArray(s?.forecasts)) {
      errors.push("malformed station entry");
      break;
    }
  }
  return errors;
}

export function forecastAt(station, horizon) {
  return (station.forecasts ?? []).find((f) => f.horizon_h === horizon) ?? null;
}

// One row per station for the chosen horizon, highest forecast first; stations without
// that horizon are dropped rather than shown as zero.
export function rowsFor(doc, horizon) {
  const rows = [];
  for (const s of Array.isArray(doc.stations) ? doc.stations : []) {
    const f = forecastAt(s, horizon);
    if (!f || !Number.isFinite(f.no2_ppb)) continue;
    rows.push({
      station: s.station,
      place: s.place ?? s.station,
      last: Number.isFinite(s.last_reading_ppb) ? s.last_reading_ppb : null,
      value: f.no2_ppb,
      low: f.interval90_low_ppb,
      high: f.interval90_high_ppb,
      category: f.aqi_category ?? "",
      local: f.target_time_local ?? "",
      utc: f.target_time_utc ?? "",
    });
  }
  return rows.sort((a, b) => b.value - a.value || a.place.localeCompare(b.place));
}

// Shared axis for the interval bars: a round number of ppb above the widest interval.
export function axisMax(rows) {
  const top = Math.max(10, ...rows.map((r) => (Number.isFinite(r.high) ? r.high : r.value)));
  const step = top > 50 ? 20 : 10;
  return Math.ceil(top / step) * step;
}

export function pct(v, max) {
  if (!Number.isFinite(v) || !(max > 0)) return 0;
  return Math.min(100, Math.max(0, (v / max) * 100));
}

export function stateOf(place) {
  const i = (place ?? "").lastIndexOf(",");
  return i === -1 ? "" : place.slice(i + 1).trim();
}

export function categorySlug(category) {
  return (category ?? "").toLowerCase().replace(/[^a-z]+/g, "-").replace(/^-|-$/g, "") || "none";
}

// "2026-11-06 21:00" -> "Fri 21:00"; the date is the station's own local date.
export function localLabel(s) {
  const t = parseUtc(s);
  if (!t) return "";
  const day = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][t.getUTCDay()];
  return `${day} ${String(t.getUTCHours()).padStart(2, "0")}:${String(t.getUTCMinutes()).padStart(2, "0")}`;
}

export function notForecast(doc) {
  const c = doc.data_contract ?? {};
  return [
    ...(c.stations_missing ?? []).map((s) => [s, "no readings from AirNow"]),
    ...(c.stations_stale ?? []).map((s) => [s, "readings too old"]),
    ...(c.stations_sparse ?? []).map((s) => [s, "too few hours in the last day"]),
  ];
}

// Live scoring (ml/scripts/score_live.py, published hourly next to the forecasts).
export const SCORE_SOURCE = SOURCE.replace("forecasts.json", "score.json");
export const MIN_SCORED = 20; // fewer than this and a comparison says nothing

// One row per horizon: what the model's error was against the readings that followed,
// next to persistence ("the next reading equals this one"). Never claims a win on a few points.
export function scoreRows(scoreDoc, window = "yesterday") {
  const horizons = scoreDoc?.windows?.[window]?.horizons;
  if (!Array.isArray(horizons)) return [];
  return horizons.map((h) => {
    const judged = h.n >= MIN_SCORED && h.mae_ppb != null && h.persistence_mae_ppb != null;
    const better = judged && h.mae_ppb < h.persistence_mae_ppb;
    const pctLower = judged && h.persistence_mae_ppb > 0
      ? Math.round(100 * (1 - h.mae_ppb / h.persistence_mae_ppb)) : null;
    return {
      horizon: `${h.horizon_h} h`,
      scored: h.n,
      model: h.mae_ppb == null ? "–" : `${h.mae_ppb.toFixed(2)} ppb`,
      persistence: h.persistence_mae_ppb == null ? "–" : `${h.persistence_mae_ppb.toFixed(2)} ppb`,
      verdict: !judged ? "too few to judge"
        : better ? `${pctLower}% lower error`
          : "no better",
    };
  });
}
