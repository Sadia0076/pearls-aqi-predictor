# src/aqi_engine/nowcast_aqi.py
from .standard_aqi import calculate_standard_aqi

def calculate_nowcast(pm_last_12h: list, pollutant: str = "pm25") -> int:
    """
    Calculate NowCast AQI for a pollutant from last 12 hourly concentrations.
    
    :param pm_last_12h: list of last 12 hourly concentrations (most recent last)
    :param pollutant: pollutant type
    :return: NowCast AQI value
    """
    if len(pm_last_12h) == 0:
        return None

    c_min = min(pm_last_12h)
    c_max = max(pm_last_12h)

    # Weighted factor
    if c_max > 0:
        scaled_rate = (c_max - c_min) / c_max
        w = 1 - scaled_rate
        w = max(0.5, min(1.0, w))
    else:
        w = 1.0

    # Weighted average (NowCast concentration)
    numerator = sum(w**t * c for t, c in enumerate(reversed(pm_last_12h)))
    denominator = sum(w**t for t in range(len(pm_last_12h)))
    nowcast_concentration = numerator / denominator

    # Convert to AQI
    return calculate_standard_aqi(nowcast_concentration, pollutant=pollutant)
