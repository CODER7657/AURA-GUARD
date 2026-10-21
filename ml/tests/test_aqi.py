import pytest

from auraguard.aqi import no2_aqi


@pytest.mark.parametrize("ppb, aqi, category", [
    (0, 0, "Good"),
    (53, 50, "Good"),
    (53.9, 50, "Good"),  # NO2 is truncated to whole ppb before the index is computed
    (54, 51, "Moderate"),
    (100, 100, "Moderate"),
    (101, 101, "Unhealthy for Sensitive Groups"),
    (360, 150, "Unhealthy for Sensitive Groups"),
    (361, 151, "Unhealthy"),
    (649, 200, "Unhealthy"),
    (650, 201, "Very Unhealthy"),
    (1250, 301, "Hazardous"),
    (2049, 500, "Hazardous"),
])
def test_epa_no2_breakpoints(ppb, aqi, category):
    out = no2_aqi(ppb)
    assert out["aqi"] == aqi
    assert out["category"] == category


def test_negative_readings_count_as_zero_and_nan_is_rejected():
    assert no2_aqi(-0.4)["aqi"] == 0
    with pytest.raises(ValueError):
        no2_aqi(float("nan"))


def test_above_the_scale():
    assert no2_aqi(2500)["aqi"] is None
