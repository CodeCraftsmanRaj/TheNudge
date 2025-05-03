import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor # For parallel historical requests

def get_weather_forecast(lat, lon, days=16):
    """
    Fetches daily weather forecast (up to 16 days) using Open-Meteo.
    """
    # Ensure days are within the API limit
    days = min(max(1, days), 16)

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
            "et0_fao_evapotranspiration",
            "weathercode"
        ]),
        "forecast_days": days,
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=15) # Add timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)
        data = response.json()
        df = pd.DataFrame(data["daily"])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"API Error fetching forecast: {e}")
        return pd.DataFrame() # Return empty dataframe on error
    except Exception as e:
        print(f"Error processing forecast data: {e}")
        return pd.DataFrame()

def get_historical_weather_for_period(lat, lon, start_date, end_date):
    """
    Fetches historical daily weather data for a specific period using Open-Meteo.
    """
    url = "https://archive-api.open-meteo.com/v1/archive" # Note: different URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "windspeed_10m_max",
            "et0_fao_evapotranspiration",
            "weathercode"
        ]),
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data or 'daily' not in data or not data['daily']['time']:
             print(f"No historical data found for {start_date.year}")
             return pd.DataFrame() # Return empty if no data
        df = pd.DataFrame(data["daily"])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
         # Keep only columns that are also in the forecast API (names match)
        common_cols = [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "rain_sum", "windspeed_10m_max", "et0_fao_evapotranspiration",
            "weathercode"
        ]
        df = df[[col for col in common_cols if col in df.columns]]
        return df
    except requests.exceptions.RequestException as e:
        print(f"API Error fetching historical data for {start_date.year}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing historical data for {start_date.year}: {e}")
        return pd.DataFrame()

def get_climatology(lat, lon, start_day_offset, end_day_offset, years_back=10):
    """
    Calculates climatology (average weather) for a future period based on past years.

    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_day_offset (int): Starting day relative to today (e.g., 17 for day 17)
        end_day_offset (int): Ending day relative to today (e.g., 30 for day 30)
        years_back (int): How many past years to average over.

    Returns:
        pd.DataFrame: Average daily weather for the period, indexed by future date.
                       Returns empty DataFrame if insufficient data.
    """
    today = datetime.now().date()
    target_start_date = today + timedelta(days=start_day_offset)
    target_end_date = today + timedelta(days=end_day_offset)

    historical_dfs = []

    # --- Fetch historical data in parallel ---
    def fetch_year(year_offset):
        past_start_date = target_start_date.replace(year=target_start_date.year - year_offset)
        past_end_date = target_end_date.replace(year=target_end_date.year - year_offset)
        # Handle potential date issues like leap years if necessary (often minor for averages)
        # Ensure end date is not before start date if period crosses year boundary
        if past_end_date < past_start_date:
             past_end_date = past_end_date.replace(year=past_start_date.year + 1)

        # Check if the historical date range is actually in the past
        if past_end_date >= today:
            print(f"Skipping year {past_start_date.year}: Date range not entirely in the past.")
            return None

        print(f"Fetching historical data for: {past_start_date} to {past_end_date}")
        return get_historical_weather_for_period(lat, lon, past_start_date, past_end_date)

    with ThreadPoolExecutor(max_workers=5) as executor: # Adjust max_workers as needed
        # Create list of offsets [1, 2, ..., years_back]
        year_offsets = list(range(1, years_back + 1))
        results = executor.map(fetch_year, year_offsets)
        historical_dfs = [df for df in results if df is not None and not df.empty]
    # --- End parallel fetch ---

    if not historical_dfs:
        print("No valid historical data fetched to calculate climatology.")
        return pd.DataFrame()

    # Align data by day of the year (ignoring year) before averaging
    aligned_data = []
    target_date_range = pd.date_range(target_start_date, target_end_date, name='time')

    for df in historical_dfs:
        # Create a common index based on month and day
        df['month_day'] = df.index.strftime('%m-%d')
        # Create a temporary target index for this historical year's data
        temp_target_index = pd.date_range(df.index.min(), periods=len(df), name='time')
        # Reindex using the target range structure but with historical data aligned by month-day
        df_reindexed = df.set_index('month_day')
        # Create a reference index based on the target future dates' month-day
        target_month_day = target_date_range.strftime('%m-%d')
        # Select data matching the target month-day sequence
        try:
            aligned_df = df_reindexed.loc[target_month_day].copy()
            # Assign the correct target future dates as index
            aligned_df.index = target_date_range
            aligned_data.append(aligned_df)
        except KeyError:
            print(f"Warning: Could not align all dates for year {df.index.min().year}. Missing days?")
            # Handle missing days if necessary, e.g., fill with NaN or skip year

    if not aligned_data:
        print("Could not align any historical data.")
        return pd.DataFrame()

    # Concatenate and calculate the mean, ignoring NaN values
    combined_historical = pd.concat(aligned_data)
    # Group by the target date index and calculate mean
    climatology_df = combined_historical.groupby(combined_historical.index).mean(numeric_only=True)

    # Optional: Round weathercode to nearest integer if needed
    if 'weathercode' in climatology_df:
        climatology_df['weathercode'] = climatology_df['weathercode'].round().astype(int)

    return climatology_df

# --- Example Usage ---
if __name__ == "__main__":
    lat = 20.5937  # India
    lon = 78.9629
    total_days_needed = 50 # Example: User wants 30 days of data
    forecast_limit = 16    # API's actual limit
    historical_years = 10  # Average over last 10 years for climatology

    print(f"Fetching forecast for the first {forecast_limit} days...")
    forecast_df = get_weather_forecast(lat, lon, days=forecast_limit)

    if total_days_needed > forecast_limit:
        print(f"\nFetching climatology for days {forecast_limit + 1} to {total_days_needed}...")
        climatology_df = get_climatology(
            lat,
            lon,
            start_day_offset=forecast_limit, # Start from the day after forecast ends
            end_day_offset=total_days_needed - 1, # End on the last day needed
            years_back=historical_years
        )

        if not forecast_df.empty and not climatology_df.empty:
            # Combine forecast and climatology data
            # Ensure columns match before concatenating (use intersection)
            common_cols = forecast_df.columns.intersection(climatology_df.columns)
            combined_df = pd.concat([forecast_df[common_cols], climatology_df[common_cols]])
            print("\nCombined Forecast (first 16 days) and Climatology (subsequent days):")
            print(combined_df)
        elif not forecast_df.empty:
            print("\nOnly forecast data available:")
            print(forecast_df)
        elif not climatology_df.empty:
             print("\nOnly climatology data available (forecast failed?):")
             print(climatology_df)
        else:
            print("\nFailed to retrieve both forecast and climatology data.")

    elif not forecast_df.empty:
        print("\nForecast Data (within 16-day limit):")
        print(forecast_df)
    else:
        print("\nFailed to retrieve forecast data.")