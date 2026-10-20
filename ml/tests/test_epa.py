import pandas as pd

from auraguard import epa

HEADER = [
    "State Code", "County Code", "Site Num", "Parameter Code", "POC", "Latitude", "Longitude", "Datum",
    "Parameter Name", "Date Local", "Time Local", "Date GMT", "Time GMT", "Sample Measurement",
    "Units of Measure", "MDL", "Uncertainty", "Qualifier", "Method Type", "Method Code", "Method Name",
    "State Name", "County Name", "Date of Last Change",
]


def write(tmp_path, rows):
    path = tmp_path / "hourly.csv"
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)
    return str(path)


def row(site="0016", poc="1", hour="05:00", gmt_hour="13:00", value="12.5"):
    return ["06", "037", site, "42602", poc, "34.0", "-118.2", "WGS84", "Nitrogen dioxide (NO2)",
            "2024-03-01", hour, "2024-03-01", gmt_hour, value, "Parts per billion", "", "", "", "", "", "",
            "California", "Los Angeles", "2024-06-01"]


def test_station_id_keeps_leading_zeros_and_offset_is_local_standard_time(tmp_path):
    df = epa.read_hourly(write(tmp_path, [row()]))
    assert df.loc[0, "station"] == "06-037-0016-1"
    assert df.loc[0, "utc_offset_h"] == -8
    assert df.loc[0, "place"] == "Los Angeles, California"


def test_missing_measurements_are_dropped(tmp_path):
    df = epa.read_hourly(write(tmp_path, [row(), row(hour="06:00", gmt_hour="14:00", value="")]))
    assert len(df) == 1


def test_most_complete_keeps_one_instrument_per_site(tmp_path):
    rows = [row(poc="1", hour=f"0{h}:00", gmt_hour=f"1{h}:00") for h in range(5)]
    rows += [row(poc="2", hour=f"0{h}:00", gmt_hour=f"1{h}:00") for h in range(3)]
    rows += [row(site="0099", hour=f"0{h}:00", gmt_hour=f"1{h}:00") for h in range(4)]
    keep = epa.most_complete(epa.read_hourly(write(tmp_path, rows)), n=5)
    assert sorted(keep["station"].unique()) == ["06-037-0016-1", "06-037-0099-1"]
