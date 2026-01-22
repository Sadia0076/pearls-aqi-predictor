# src/aqi_engine/compare.py
from .standard_aqi import calculate_standard_aqi
from .nowcast_aqi import calculate_nowcast

def compare_aqi(pm_last_12h: list, pm_24h_avg: float = None, pollutant: str = "pm25") -> dict:
    """
    Compare Standard AQI (24h avg) vs NowCast AQI (real-time)
    
    :param pm_last_12h: last 12 hourly concentrations
    :param pm_24h_avg: optional 24-hour average concentration
    :param pollutant: pollutant type
    :return: dict with 'standard_aqi' and 'nowcast_aqi'
    """
    if pm_24h_avg is None:
        pm_24h_avg = sum(pm_last_12h)/len(pm_last_12h)

    standard = calculate_standard_aqi(pm_24h_avg, pollutant=pollutant)
    nowcast = calculate_nowcast(pm_last_12h, pollutant=pollutant)

    return {
        "standard_aqi": standard,
        "nowcast_aqi": nowcast
    }
