import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ageHours, ageLabel, axisMax, categorySlug, localLabel, notForecast, parseUtc, pct, rowsFor,
  stateOf, validateDoc,
} from "../lib.js";

const doc = () => ({
  issued_at_utc: "2026-11-07 04:00",
  data_as_of_utc: "2026-11-07 03:00",
  horizons_h: [1, 6],
  data_contract: { stations_missing: ["17-031-0119-1"], stations_stale: ["x"], stations_sparse: [] },
  stations: [
    { station: "a", place: "Maricopa, Arizona", last_reading_ppb: 25,
      forecasts: [{ horizon_h: 1, no2_ppb: 31.9, interval90_low_ppb: 26.6, interval90_high_ppb: 37.1, aqi_category: "Good" }] },
    { station: "b", place: "Cass, North Dakota", last_reading_ppb: null,
      forecasts: [{ horizon_h: 1, no2_ppb: 40, interval90_low_ppb: 30, interval90_high_ppb: 61, aqi_category: "Moderate" },
                  { horizon_h: 6, no2_ppb: 12, interval90_low_ppb: 1, interval90_high_ppb: 20 }] },
  ],
});

test("parseUtc reads the pipeline format and rejects anything else", () => {
  assert.equal(parseUtc("2026-11-07 04:00").toISOString(), "2026-11-07T04:00:00.000Z");
  assert.equal(parseUtc("2026-11-07T04:00").toISOString(), "2026-11-07T04:00:00.000Z");
  for (const bad of [null, undefined, "", "yesterday", 42]) assert.equal(parseUtc(bad), null);
});

test("ageHours and ageLabel", () => {
  const now = new Date("2026-11-07T06:30:00Z");
  assert.equal(ageHours("2026-11-07 04:00", now), 2.5);
  assert.equal(ageHours("garbage", now), null);
  assert.equal(ageLabel(null), "unknown");
  assert.equal(ageLabel(0), "just now");
  assert.equal(ageLabel(0.5), "30 min ago");
  assert.equal(ageLabel(2.5), "2 h ago");
});

test("validateDoc accepts a good document and names what is wrong with a bad one", () => {
  assert.deepEqual(validateDoc(doc()), []);
  assert.deepEqual(validateDoc(null), ["response is not a JSON object"]);
  assert.deepEqual(validateDoc("x"), ["response is not a JSON object"]);
  assert.ok(validateDoc({}).includes("no stations list"));
  assert.ok(validateDoc({ ...doc(), issued_at_utc: "?" }).includes("missing issue time"));
  assert.ok(validateDoc({ ...doc(), stations: [{ station: 1 }] }).includes("malformed station entry"));
  assert.deepEqual(validateDoc({ ...doc(), stations: "x" }), ["no stations list"]);
});

test("rowsFor sorts by forecast and drops stations without that horizon", () => {
  const r1 = rowsFor(doc(), 1);
  assert.deepEqual(r1.map((r) => r.station), ["b", "a"]);
  assert.equal(r1[0].last, null);
  const r6 = rowsFor(doc(), 6);
  assert.deepEqual(r6.map((r) => r.station), ["b"]);
  assert.equal(r6[0].category, "");
  assert.deepEqual(rowsFor(doc(), 24), []);
  assert.deepEqual(rowsFor({}, 1), []);
});

test("axisMax rounds above the widest interval; pct clamps", () => {
  assert.equal(axisMax(rowsFor(doc(), 1)), 80);
  assert.equal(axisMax([]), 10);
  assert.equal(axisMax([{ value: 7, high: NaN }]), 10);
  assert.equal(pct(20, 80), 25);
  assert.equal(pct(-3, 80), 0);
  assert.equal(pct(500, 80), 100);
  assert.equal(pct(NaN, 80), 0);
  assert.equal(pct(5, 0), 0);
});

test("small formatters", () => {
  assert.equal(stateOf("Saint Louis, Missouri"), "Missouri");
  assert.equal(stateOf("nowhere"), "");
  assert.equal(categorySlug("Unhealthy for Sensitive Groups"), "unhealthy-for-sensitive-groups");
  assert.equal(categorySlug(""), "none");
  assert.equal(localLabel("2026-11-06 21:00"), "Fri 21:00");
  assert.equal(localLabel("bad"), "");
});

test("notForecast lists every excluded station with its reason", () => {
  assert.deepEqual(notForecast(doc()), [
    ["17-031-0119-1", "no readings from AirNow"],
    ["x", "readings too old"],
  ]);
  assert.deepEqual(notForecast({}), []);
});

import { MIN_SCORED, SCORE_SOURCE, scoreRows } from "../lib.js";

const scoreDoc = {
  windows: {
    yesterday: {
      horizons: [
        { horizon_h: 1, n: 97, mae_ppb: 1.24, persistence_mae_ppb: 1.39 },
        { horizon_h: 12, n: 12, mae_ppb: 1.59, persistence_mae_ppb: 5.97 },
        { horizon_h: 24, n: 33, mae_ppb: 3.6, persistence_mae_ppb: 3.43 },
        { horizon_h: 6, n: 0, mae_ppb: null, persistence_mae_ppb: null },
      ],
    },
  },
};

test("score rows compare with persistence and never claim a win on a few points", () => {
  const [h1, h12, h24, h6] = scoreRows(scoreDoc);
  assert.equal(h1.verdict, "11% lower error");
  assert.equal(h1.model, "1.24 ppb");
  assert.ok(12 < MIN_SCORED);
  assert.equal(h12.verdict, "too few to judge");
  assert.equal(h24.verdict, "no better");
  assert.deepEqual([h6.model, h6.verdict], ["–", "too few to judge"]);
});

test("score rows tolerate a missing or malformed file", () => {
  assert.deepEqual(scoreRows(null), []);
  assert.deepEqual(scoreRows({ windows: {} }), []);
  assert.ok(SCORE_SOURCE.endsWith("/live-data/ml/live/score.json"));
});
