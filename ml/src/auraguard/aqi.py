"""EPA Air Quality Index for 1-hour NO2.

Breakpoints from the EPA Technical Assistance Document for the Reporting of Daily Air
Quality (AQI), NO2 1-hour, in parts per billion. Concentrations are truncated to whole
ppb before the index is computed, as that document specifies for NO2.

Plain standard library, so the agent can use it without numpy or pandas.
"""

from __future__ import annotations

import math

# (low ppb, high ppb, low index, high index, category)
NO2_BREAKPOINTS = (
    (0, 53, 0, 50, "Good"),
    (54, 100, 51, 100, "Moderate"),
    (101, 360, 101, 150, "Unhealthy for Sensitive Groups"),
    (361, 649, 151, 200, "Unhealthy"),
    (650, 1249, 201, 300, "Very Unhealthy"),
    (1250, 2049, 301, 500, "Hazardous"),
)


def no2_aqi(no2_ppb: float) -> dict:
    """AQI value and category for a 1-hour NO2 concentration in ppb."""
    if no2_ppb is None or not math.isfinite(no2_ppb):
        raise ValueError("NO2 concentration must be a finite number")
    c = max(0, math.floor(no2_ppb))
    for lo, hi, ilo, ihi, name in NO2_BREAKPOINTS:
        if lo <= c <= hi:
            index = round((ihi - ilo) / (hi - lo) * (c - lo) + ilo)
            return {"no2_ppb": no2_ppb, "truncated_ppb": c, "aqi": index, "category": name,
                    "band_ppb": [lo, hi]}
    return {"no2_ppb": no2_ppb, "truncated_ppb": c, "aqi": None, "category": "Beyond the AQI (above 2049 ppb)",
            "band_ppb": [2050, None]}
