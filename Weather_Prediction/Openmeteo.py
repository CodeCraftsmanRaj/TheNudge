import requests
import pandas as pd

def get_weather_forecast(lat, lon, days=7):
    """
    Fetches daily weather forecast (temp, rainfall, humidity) for given lat/lon using Open-Meteo.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        days (int): Number of forecast days (max 16)

    Returns:
        pd.DataFrame: Weather forecast data
    """
    assert 1 <= days <= 16, "Open-Meteo supports 1 to 16 days forecast only."

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "windspeed_10m_max",
            "et0_fao_evapotranspiration",  # optional
            "weathercode"
        ]),
        "forecast_days": days,
        "timezone": "auto"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data["daily"])
    return df

# Example usage
if __name__ == "__main__":
    lat = 20.5937  # India
    lon = 78.9629
    days = 10

    forecast_df = get_weather_forecast(lat, lon, days)
    print(forecast_df)
