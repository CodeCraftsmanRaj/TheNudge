import requests
import pandas as pd
from datetime import datetime, timedelta, date
import logging

# --- Try to import Prophet ---
from prophet import Prophet

# --- Configure Logging (Optional: Reduces Prophet's output) ---
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# --- Function to get Historical Data (Unchanged) ---

def get_historical_weather_for_period(lat, lon, start_date, end_date):
    """
    Fetches historical daily weather data for a specific period using Open-Meteo.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    # Define the variables we want - keep consistent
    daily_variables = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "rain_sum", "windspeed_10m_max", "et0_fao_evapotranspiration",
        "weathercode"
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "daily": ",".join(daily_variables),
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=45) # Longer timeout for potentially large historical requests
        response.raise_for_status()
        data = response.json()
        if not data or 'daily' not in data or not data['daily']['time']:
             print(f"Warning: No historical data found for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
             return pd.DataFrame()
        df = pd.DataFrame(data["daily"])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        # Ensure only requested columns are returned, handling potential missing data from API
        available_cols = [col for col in daily_variables if col in df.columns]
        return df[available_cols]
    except requests.exceptions.RequestException as e:
        print(f"API Error fetching historical data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}): {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing historical data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}): {e}")
        return pd.DataFrame()

# --- Modified Function using Prophet for ALL requested days ---

def generate_full_prophet_forecast(lat, lon, forecast_days, historical_years=5):
    """
    Uses Prophet trained on historical data to forecast weather variables
    for the specified number of future days, starting from tomorrow.

    WARNING: Prophet is a statistical model and not physics-based.
             Weather forecasts generated this way are highly speculative,
             especially beyond a few days, and likely unreliable for specific
             daily values (particularly precipitation and extreme events).

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        forecast_days (int): The total number of future days to forecast.
        historical_years (int): Number of past years of data to fetch for training.

    Returns:
        pd.DataFrame: Prophet forecast data for the specified future period.
                      Returns empty DataFrame on failure.
    """
    if forecast_days <= 0:
        print("Error: Number of forecast days must be positive.")
        return pd.DataFrame()

    today = date.today()
    # Forecast starts tomorrow and lasts for 'forecast_days'
    forecast_start_date = today + timedelta(days=1)
    forecast_end_date = forecast_start_date + timedelta(days=forecast_days - 1)

    print(f"\n--- Generating Full Prophet Forecast ---")
    print(f"--- Forecast Period: {forecast_start_date} to {forecast_end_date} ({forecast_days} days) ---")
    print(f"--- Training on {historical_years} years of historical data ending {today} ---")
    print("--- WARNING: This forecast relies SOLELY on Prophet statistical extrapolation. Use with extreme caution. ---")

    # 1. Determine historical data range (ends today)
    hist_end_date = today
    hist_start_date = hist_end_date.replace(year=hist_end_date.year - historical_years)

    print(f"Fetching historical data: {hist_start_date} to {hist_end_date}...")
    historical_df = get_historical_weather_for_period(lat, lon, hist_start_date, hist_end_date)

    if historical_df.empty or len(historical_df) < 365: # Need sufficient data for yearly patterns
        print("ERROR: Insufficient historical data obtained to train Prophet.")
        return pd.DataFrame()

    # 2. Prepare for Prophet forecasting
    variables_to_forecast = historical_df.columns.tolist()
    all_forecasts = {}
    future_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, name='ds')

    if future_dates.empty:
        print("Error: Failed to create future date range.")
        return pd.DataFrame()

    future = pd.DataFrame({'ds': future_dates})

    # 3. Iterate through variables, train Prophet, predict
    for variable in variables_to_forecast:
        print(f"  Training Prophet model for: {variable}...")
        prophet_df = historical_df[[variable]].reset_index()
        prophet_df.rename(columns={'time': 'ds', variable: 'y'}, inplace=True)
        prophet_df = prophet_df.dropna(subset=['y'])

        if len(prophet_df) < 30: # Increased minimum slightly
             print(f"  Skipping {variable}: Not enough non-NaN data points ({len(prophet_df)}) after cleaning.")
             continue

        # Initialize and fit model
        # Consider adding more seasonality if needed, but yearly is key for weather
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

        try:
            model.fit(prophet_df)
        except Exception as e:
            print(f"  ERROR fitting Prophet for {variable}: {e}")
            print(f"  Skipping forecast for {variable}.")
            continue

        # Predict
        try:
            forecast = model.predict(future)
        except Exception as e:
            print(f"  ERROR predicting with Prophet for {variable}: {e}")
            print(f"  Skipping forecast for {variable}.")
            continue

        # Extract forecast ('yhat') and align index
        forecast_subset = pd.merge(future, forecast[['ds', 'yhat']], on='ds', how='left')
        forecast_subset.set_index('ds', inplace=True)

        # Post-processing/Sanity Checks
        if variable in ["precipitation_sum", "rain_sum", "windspeed_10m_max", "et0_fao_evapotranspiration"]:
            forecast_subset['yhat'] = forecast_subset['yhat'].clip(lower=0)
        elif variable == "weathercode":
            forecast_subset['yhat'] = forecast_subset['yhat'].round().astype(int).clip(lower=0)
        # No specific clips on temperature by default

        all_forecasts[variable] = forecast_subset['yhat']
        print(f"  Completed Prophet forecast for: {variable}")

    # 4. Combine forecasts
    if not all_forecasts:
        print("ERROR: Prophet forecasting failed for all variables.")
        return pd.DataFrame()

    prophet_results_df = pd.DataFrame(all_forecasts)
    prophet_results_df.index.name = 'time'

    print("--- Full Prophet Forecast Generation Complete ---")
    return prophet_results_df


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- User Configuration ---
    lat = 20.5937  # India
    lon = 78.9629
    total_days_to_forecast = 50 # <--- SET HOW MANY DAYS YOU WANT FORECASTED
    historical_years_for_training = 5 # How many years of history to train Prophet on
    # --- End Configuration ---


    print(f"--- Prophet Weather Forecast Request ---")
    print(f"Location: Lat={lat}, Lon={lon}")
    print(f"Requesting forecast for: {total_days_to_forecast} days (starting tomorrow)")
    print(f"Model: Prophet (trained on {historical_years_for_training} years of historical data)")
    print("="*40)

    # Generate the full forecast using Prophet
    full_forecast_df = generate_full_prophet_forecast(
        lat,
        lon,
        forecast_days=total_days_to_forecast,
        historical_years=historical_years_for_training
    )

    # Display results
    print("\n--- Prophet Weather Forecast Results ---")
    if not full_forecast_df.empty:
        print(f"\nForecast Data ({len(full_forecast_df)} days):")
        # Display options for large dataframes
        pd.set_option('display.max_rows', 100) # Show more rows if needed
        print(full_forecast_df)
        pd.reset_option('display.max_rows') # Reset display option
    else:
        print("\nERROR: Failed to generate Prophet forecast.")

    print("\n" + "="*40)
    print("--- IMPORTANT REMINDER ---")
    print("The forecast above was generated entirely by the Prophet statistical model.")
    print("It identifies trends and seasonality from past data but DOES NOT understand")
    print("real-time atmospheric physics or predict specific weather events accurately.")
    print("Reliability decreases significantly for longer forecast horizons.")
    print("Use this data for exploratory purposes only and with extreme caution.")
    print("--- End of Report ---")