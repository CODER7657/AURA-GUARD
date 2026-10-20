"""Load EPA AirData hourly files into one tidy table.

Source: https://aqs.epa.gov/aqsweb/airdata/download_files.html, e.g.
``hourly_42602_2024.zip`` (parameter 42602 = NO2, parts per billion). These are
public files and need no account.

A monitor is identified by state, county and site number plus POC (the instrument
at that site). Times are kept in GMT; EPA reports local *standard* time, so each
monitor's offset from GMT is constant and is kept for calendar features.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = {
    "State Code": "state",
    "County Code": "county",
    "Site Num": "site",
    "POC": "poc",
    "Latitude": "lat",
    "Longitude": "lon",
    "Date Local": "date_local",
    "Time Local": "time_local",
    "Date GMT": "date_gmt",
    "Time GMT": "time_gmt",
    "Sample Measurement": "value",
    "State Name": "state_name",
    "County Name": "county_name",
}


def read_hourly(path: str) -> pd.DataFrame:
    """One row per monitor-hour: station, time_gmt, utc_offset_h, value, lat, lon, place."""
    # Codes stay text so leading zeros survive; everything else is parsed normally,
    # which keeps a ~4 million row national file within Colab's memory.
    codes = {"State Code": str, "County Code": str, "Site Num": str, "POC": str}
    raw = pd.read_csv(path, usecols=list(COLUMNS), dtype=codes)
    df = raw.rename(columns=COLUMNS)
    df["station"] = (
        df["state"].str.zfill(2) + "-" + df["county"].str.zfill(3) + "-" + df["site"].str.zfill(4) + "-" + df["poc"]
    )
    df["time_gmt"] = pd.to_datetime(df["date_gmt"] + " " + df["time_gmt"], format="%Y-%m-%d %H:%M")
    local = pd.to_datetime(df["date_local"] + " " + df["time_local"], format="%Y-%m-%d %H:%M")
    df["utc_offset_h"] = ((local - df["time_gmt"]).dt.total_seconds() / 3600).round().astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["place"] = df["county_name"] + ", " + df["state_name"]
    out = df[["station", "time_gmt", "utc_offset_h", "value", "lat", "lon", "place"]]
    return out.dropna(subset=["value"]).drop_duplicates(["station", "time_gmt"])


def most_complete(df: pd.DataFrame, n: int = 20, one_per_site: bool = True) -> pd.DataFrame:
    """Keep the ``n`` monitors with the most valid hours.

    With ``one_per_site`` only the most complete instrument at each site is kept, so
    two instruments in the same place are not counted as two stations.
    """
    counts = df.groupby("station").size().rename("hours").reset_index()
    counts["site_key"] = counts["station"].str.rsplit("-", n=1).str[0]
    if one_per_site:
        counts = counts.sort_values("hours", ascending=False).drop_duplicates("site_key")
    keep = counts.nlargest(n, "hours")["station"]
    return df[df["station"].isin(keep)].copy()


def stations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per monitor with its place, location and coverage, for the README."""
    g = df.groupby("station")
    return pd.DataFrame(
        {
            "place": g["place"].first(),
            "lat": g["lat"].first(),
            "lon": g["lon"].first(),
            "hours": g.size(),
            "mean_ppb": g["value"].mean().round(2),
        }
    ).reset_index()
