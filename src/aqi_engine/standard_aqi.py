# src/aqi_engine/standard_aqi.py

def calculate_standard_aqi(concentration: float, pollutant: str = "pm25") -> int:
    """
    Calculate standard 24-hour AQI for a given pollutant using EPA breakpoints.
    
    :param concentration: 24-hour average concentration of the pollutant
    :param pollutant: pollutant type ("pm25", "pm10", "o3", "no2", "co", "so2")
    :return: AQI value (integer)
    """

    breakpoints = {
        "pm25": [
            (0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500)
        ],
        "pm10": [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 604, 301, 500)
        ],
        "o3": [
            (0, 0.054, 0, 50),
            (0.055, 0.070, 51, 100),
            (0.071, 0.085, 101, 150),
            (0.086, 0.105, 151, 200),
            (0.106, 0.200, 201, 300)
        ],
        # You can add CO, NO2, SO2 similarly if needed
    }

    if pollutant not in breakpoints:
        raise ValueError(f"No breakpoints defined for pollutant: {pollutant}")

    for bplo, bphi, ilo, ihi in breakpoints[pollutant]:
        if bplo <= concentration <= bphi:
            aqi = ((ihi - ilo) / (bphi - bplo)) * (concentration - bplo) + ilo
            return round(aqi)
    return 500
