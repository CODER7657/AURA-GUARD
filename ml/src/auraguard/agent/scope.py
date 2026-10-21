"""Decide, without a model, whether a question is in scope for the forecast analyst.

The analyst only knows hourly NO2 forecasts at 20 EPA monitors. Three kinds of
question are out of scope and must get an honest refusal:

    other_pollutant   PM2.5, ozone, smoke, pollen and so on
    medical           personal health or medical advice
    no_station        a place more than ``NEAR_KM`` from every monitor, outside the US,
                      or a US state with no monitor

The place check uses a small built-in gazetteer, so a place it does not know is not
flagged; the grounding checker still rejects any state or county the answer names
that is not a monitor's or the question's.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

NEAR_KM = 80.0

OTHER_POLLUTANTS = [
    r"pm\s?2\.?5", r"pm\s?10", r"particulate", r"\bparticles?\b", r"\bozone\b", r"\bo3\b", r"\bso2\b",
    r"sulfur dioxide", r"sulphur", r"carbon monoxide", r"\bCO\b", r"\bsmoke\b", r"wildfire", r"\bpollen\b",
    r"\blead (levels?|pollution|exposure)\b", r"benzene", r"methane", r"\bvocs?\b", r"formaldehyde",
    r"\bradon\b", r"\bdust\b",
]
MEDICAL = [
    r"asthma", r"inhaler", r"medication", r"medicine", r"\bmedical\b", r"\bdoctor\b", r"symptoms?",
    r"\bcopd\b", r"pregnan", r"heart (condition|disease)", r"\blungs?\b", r"health advice",
    r"wear a mask", r"\bsafe (for|to)\b", r"\bis it safe\b", r"\bshould (i|my|we)\b.*\b(run|jog|exercise|walk|"
    r"go out|go outside|play|bike|cycle|open)\b", r"\b(my|our) (son|daughter|kids?|child|children|baby)\b",
]

# (name, state, lat, lon); state None = outside the US
PLACES = [
    ("New York City", "New York", 40.71, -74.01), ("New York", "New York", 40.71, -74.01),
    ("Los Angeles", "California", 34.05, -118.24), ("Chicago", "Illinois", 41.88, -87.63),
    ("Houston", "Texas", 29.76, -95.37), ("Phoenix", "Arizona", 33.45, -112.07),
    ("Philadelphia", "Pennsylvania", 39.95, -75.17), ("San Antonio", "Texas", 29.42, -98.49),
    ("San Diego", "California", 32.72, -117.16), ("Dallas", "Texas", 32.78, -96.80),
    ("San Jose", "California", 37.34, -121.89), ("Austin", "Texas", 30.27, -97.74),
    ("San Francisco", "California", 37.77, -122.42), ("Seattle", "Washington", 47.61, -122.33),
    ("Denver", "Colorado", 39.74, -104.99), ("Washington DC", "District of Columbia", 38.91, -77.04),
    ("Washington, D.C.", "District of Columbia", 38.91, -77.04), ("Boston", "Massachusetts", 42.36, -71.06),
    ("Detroit", "Michigan", 42.33, -83.05), ("Atlanta", "Georgia", 33.75, -84.39),
    ("Miami", "Florida", 25.76, -80.19), ("Minneapolis", "Minnesota", 44.98, -93.27),
    ("Las Vegas", "Nevada", 36.17, -115.14), ("Portland", "Oregon", 45.52, -122.68),
    ("Salt Lake City", "Utah", 40.76, -111.89), ("Kansas City", "Missouri", 39.10, -94.58),
    ("St. Louis", "Missouri", 38.63, -90.20), ("Saint Louis", "Missouri", 38.63, -90.20),
    ("Newark", "New Jersey", 40.74, -74.17), ("New Brunswick", "New Jersey", 40.49, -74.45),
    ("Fargo", "North Dakota", 46.88, -96.79), ("Bismarck", "North Dakota", 46.81, -100.78),
    ("Minot", "North Dakota", 48.23, -101.30), ("Williston", "North Dakota", 48.15, -103.62),
    ("Santa Monica", "California", 34.02, -118.49), ("Pasadena", "California", 34.15, -118.14),
    ("Long Beach", "California", 33.77, -118.19), ("Anaheim", "California", 33.84, -117.91),
    ("Palm Springs", "California", 33.83, -116.55), ("Ontario", "California", 34.06, -117.65),
    ("Fontana", "California", 34.09, -117.44), ("Gary", "Indiana", 41.59, -87.35),
    ("Tempe", "Arizona", 33.43, -111.94), ("Sacramento", "California", 38.58, -121.49),
    ("Tucson", "Arizona", 32.22, -110.97), ("Wichita", "Kansas", 37.69, -97.34),
    ("Chanute", "Kansas", 37.68, -95.46), ("Roosevelt", "Utah", 40.30, -109.99),
    ("Honolulu", "Hawaii", 21.31, -157.86), ("Anchorage", "Alaska", 61.22, -149.90),
    ("Nashville", "Tennessee", 36.16, -86.78), ("Pittsburgh", "Pennsylvania", 40.44, -80.00),
    ("Cleveland", "Ohio", 41.50, -81.69), ("Baltimore", "Maryland", 39.29, -76.61),
    ("Charlotte", "North Carolina", 35.23, -80.84), ("Orlando", "Florida", 28.54, -81.38),
    ("Milwaukee", "Wisconsin", 43.04, -87.91), ("Omaha", "Nebraska", 41.26, -95.93),
    ("Albuquerque", "New Mexico", 35.08, -106.65),
    ("London", None, 51.51, -0.13), ("Paris", None, 48.86, 2.35), ("New Delhi", None, 28.61, 77.21),
    ("Delhi", None, 28.70, 77.10), ("Mumbai", None, 19.08, 72.88), ("Beijing", None, 39.90, 116.40),
    ("Tokyo", None, 35.68, 139.69), ("Toronto", None, 43.65, -79.38), ("Mexico City", None, 19.43, -99.13),
    ("Sydney", None, -33.87, 151.21), ("Berlin", None, 52.52, 13.40), ("Lagos", None, 6.52, 3.38),
    ("Cairo", None, 30.04, 31.24), ("Sao Paulo", None, -23.55, -46.63), ("Ahmedabad", None, 23.02, 72.57),
    ("India", None, 22.0, 79.0), ("Canada", None, 56.0, -106.0), ("Mexico", None, 23.6, -102.5),
    ("Europe", None, 50.0, 10.0), ("China", None, 35.0, 103.0),
]


def km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


@dataclass
class Scope:
    in_scope: bool
    reasons: list[str] = field(default_factory=list)  # other_pollutant, medical, no_station
    places: list[dict] = field(default_factory=list)  # places recognised in the question


def _any(patterns: list[str], text: str, flags=re.IGNORECASE) -> list[str]:
    return [m.group(0) for p in patterns for m in [re.search(p, text, 0 if p == r"\bCO\b" else flags)] if m]


def classify(question: str, stations: list[dict], state_names: set[str]) -> Scope:
    """``stations`` are dicts with station, place ("County, State"), lat and lon."""
    reasons, places = [], []
    if _any(OTHER_POLLUTANTS, question):
        reasons.append("other_pollutant")
    if _any(MEDICAL, question):
        reasons.append("medical")

    rest = question
    for name, state, lat, lon in sorted(PLACES, key=lambda p: -len(p[0])):
        if re.search(rf"\b{re.escape(name)}\b", rest, re.IGNORECASE):
            rest = re.sub(rf"\b{re.escape(name)}\b", " ", rest, flags=re.IGNORECASE)
            nearest = min(stations, key=lambda s: km(lat, lon, s["lat"], s["lon"]))
            d = km(lat, lon, nearest["lat"], nearest["lon"])
            places.append({"name": name, "state": state, "nearest_station": nearest["station"],
                           "distance_km": round(d), "near": state is not None and d <= NEAR_KM})
    station_states = {s["place"].split(", ")[-1] for s in stations}
    county_names = {s["place"].split(", ")[0].lower() for s in stations}
    for state in sorted(state_names, key=len, reverse=True):
        if re.search(rf"\b{re.escape(state)}\b", rest):
            rest = re.sub(rf"\b{re.escape(state)}\b", " ", rest)
            if state not in station_states and state.lower() not in county_names:
                places.append({"name": state, "state": state, "near": False, "nearest_station": None})
    if places and not any(p["near"] for p in places):
        reasons.append("no_station")
    return Scope(in_scope=not reasons, reasons=reasons, places=places)
