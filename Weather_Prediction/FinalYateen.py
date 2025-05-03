# import requests
# import pandas as pd
# import numpy as np
# import logging # Optional: for better feedback

# # Configure logging (optional)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # A more comprehensive list of potential base variables (check API docs for full list)
# DEFAULT_VARIABLES_BASE = [
#     "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
#     "precipitation_sum", "precipitation_mean",
#     "relative_humidity_2m_mean",
#     "surface_pressure_mean",
#     "cloud_cover_mean",
#     "et0_fao_evapotranspiration_sum",
#     "soil_temperature_0_to_10cm_mean",
#     "soil_moisture_0_to_10cm_mean",
#     "snow_depth_mean",
#     "wind_speed_10m_mean", "wind_direction_10m_dominant", # Note: Averaging dominant direction might not be meaningful
#     "shortwave_radiation_sum"
# ]

# def get_ensemble_averaged_forecast(lat, lon, start_date, end_date,
#                                    variables_base=DEFAULT_VARIABLES_BASE,
#                                    timezone="auto"):
#     """
#     Fetches seasonal forecast data for a given period, averaging across
#     ensemble members for each requested base variable on each day.

#     Args:
#         lat (float): Latitude.
#         lon (float): Longitude.
#         start_date (str): Start date in YYYY-MM-DD format.
#         end_date (str): End date in YYYY-MM-DD format.
#         variables_base (list, optional): List of base variable names to request.
#                                          Defaults to a predefined list.
#                                          See Open-Meteo Seasonal API docs for options.
#         timezone (str, optional): Timezone for dates. Defaults to "auto".

#     Returns:
#         pandas.DataFrame: A DataFrame with dates as the index and columns
#                           representing the ensemble-averaged values for each
#                           requested base variable. Returns an empty DataFrame
#                           on failure after logging an error.
#     """
#     logging.info(f"Requesting seasonal forecast for {lat}, {lon} from {start_date} to {end_date}")
#     logging.info(f"Requesting variables: {variables_base}")

#     # 1) Construct API request parameters
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "daily": ",".join(variables_base),  # Use the provided base variable names
#         "start_date": start_date,
#         "end_date": end_date,
#         "timezone": timezone
#     }
#     url = "https://seasonal-api.open-meteo.com/v1/seasonal"

#     # 2) Call the API
#     try:
#         resp = requests.get(url, params=params, timeout=60) # Added timeout
#         resp.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
#     except requests.exceptions.Timeout:
#         logging.error("API request timed out.")
#         return pd.DataFrame() # Return empty DataFrame on timeout
#     except requests.exceptions.RequestException as e:
#         logging.error(f"API request failed: {e}")
#         # Attempt to get more details from response if available
#         try:
#             error_details = resp.json() if resp else "No response object"
#             logging.error(f"API response content: {error_details}")
#         except Exception as json_e:
#              logging.error(f"Could not parse error response JSON: {json_e}. Raw text: {resp.text if resp else 'N/A'}")
#         return pd.DataFrame() # Return empty DataFrame on other request errors

#     # 3) Process the response JSON
#     try:
#         data = resp.json()
#         if "daily" not in data or "time" not in data["daily"]:
#             logging.error("API response missing 'daily' or 'daily.time' section.")
#             logging.debug(f"Received data: {data}")
#             return pd.DataFrame() # Return empty DataFrame if structure is wrong

#         daily = data["daily"]
#         dates = pd.to_datetime(daily["time"]) # Convert dates immediately
#         num_dates = len(dates)

#         if num_dates == 0:
#              logging.warning("API returned 0 dates.")
#              return pd.DataFrame()

#         processed_data = {"time": dates}

#         # 4) For each base variable, average its member arrays across all dates
#         for var in variables_base:
#             # Find returned keys like var_member01, var_member02, …
#             member_keys = sorted([k for k in daily.keys() if k.startswith(f"{var}_member")])

#             if not member_keys:
#                 logging.warning(f"No ensemble member data found for variable '{var}'. Filling with NaN.")
#                 # Fill with NaNs if no members found for this variable
#                 processed_data[var] = [np.nan] * num_dates
#                 continue # Move to the next variable

#             # Collect the values for ALL dates for each member
#             # Creates a list of lists (or numpy array) where rows are members, columns are dates
#             try:
#                  # Ensure all member arrays have the expected length
#                 member_values = []
#                 for k in member_keys:
#                     if len(daily[k]) != num_dates:
#                          raise ValueError(f"Length mismatch for {k}: expected {num_dates}, got {len(daily[k])}")
#                     # Handle potential None values from API if they occur (though unlikely for numbers)
#                     member_values.append([v if v is not None else np.nan for v in daily[k]])

#                 # Convert to numpy array for easier averaging
#                 member_array = np.array(member_values, dtype=float) # Use float for potential NaNs

#                 # Calculate the mean across members (axis=0) for each date
#                 # np.nanmean safely ignores NaNs during calculation
#                 averaged_values = np.nanmean(member_array, axis=0)
#                 processed_data[var] = averaged_values.tolist() # Convert back to list for DataFrame
#                 logging.debug(f"Processed variable '{var}' with {len(member_keys)} members.")

#             except Exception as e:
#                 logging.error(f"Error processing variable '{var}': {e}")
#                 processed_data[var] = [np.nan] * num_dates # Fill with NaN on error


#         # 5) Create and return DataFrame
#         df = pd.DataFrame(processed_data)
#         df = df.set_index("time")
#         logging.info(f"Successfully processed forecast data. Shape: {df.shape}")
#         return df

#     except ValueError as e:
#          logging.error(f"Error converting data or date mismatch: {e}")
#          logging.debug(f"Problematic 'daily' data section: {daily}")
#          return pd.DataFrame()
#     except Exception as e:
#         logging.error(f"An unexpected error occurred during data processing: {e}")
#         logging.debug(f"Raw JSON data: {data}")
#         return pd.DataFrame()


# # Example usage:
# if __name__ == "__main__":
#     lat, lon = 19.0760, 72.8777         # Mumbai, India
#     # Use a shorter, more recent range for testing seasonal forecasts
#     start_date, end_date = "2024-07-01", "2024-09-30" # Example: Q3 2024 forecast

#     # Example with default variables:
#     print("--- Fetching forecast with default variables ---")
#     forecast_df = get_ensemble_averaged_forecast(lat, lon, start_date, end_date)

#     if not forecast_df.empty:
#         print(f"Forecast DataFrame shape: {forecast_df.shape}")
#         print("First 5 rows:")
#         print(forecast_df.head())
#         print("\nLast 5 rows:")
#         print(forecast_df.tail())
#         print("\nDataFrame Info:")
#         forecast_df.info()
#     else:
#         print("Failed to retrieve or process forecast data.")

#     # Example with a custom, smaller set of variables:
#     # print("\n--- Fetching forecast with custom variables ---")
#     # custom_vars = ["temperature_2m_mean", "precipitation_sum"]
#     # forecast_custom_df = get_ensemble_averaged_forecast(lat, lon, start_date, end_date,
#     #                                                    variables_base=custom_vars)
#     # if not forecast_custom_df.empty:
#     #     print(forecast_custom_df)
#     # else:
#     #     print("Failed to retrieve or process forecast data for custom variables.")


import requests
import pandas as pd
import numpy as np

def get_forecast_on_end_date(lat, lon, start_date, end_date,
                             variables_base=["temperature_2m_max",
                                             "temperature_2m_min",
                                             "precipitation_sum"],
                             timezone="auto"):
    """
    Fetch only the final forecast on `end_date` by:
      1. Requesting base daily variables (the API returns ensemble-member arrays).
      2. Filtering out None values.
      3. Averaging remaining members for the end_date.
    """

    url = "https://seasonal-api.open-meteo.com/v1/seasonal"
    # 1) Request only the base variable names (not memberXX names) 
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(variables_base),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise Exception(f"API Error {resp.status_code}: {resp.text}")

    data = resp.json()
    daily = data.get("daily")
    if not daily or "time" not in daily:
        raise KeyError("No 'daily' time series returned.")

    # 2) Find the index of the desired end_date
    times = daily["time"]
    if end_date not in times:
        raise ValueError(f"End date {end_date} not in returned time array.")
    idx = times.index(end_date)

    result = {"date": end_date}
    # 3) For each base variable, average its ensemble members—filtering out None :contentReference[oaicite:0]{index=0}
    for var in variables_base:
        member_keys = [k for k in daily.keys() if k.startswith(f"{var}_member")]
        # collect values at the end_date index, filter None 
        vals = [daily[k][idx] for k in member_keys]
        clean = [v for v in vals if v is not None]
        result[var] = float(np.mean(clean)) if clean else None

    return result

# Example Usage
if __name__ == "__main__":
    lat, lon = 19.0760, 72.8777
    start_date, end_date = "2025-05-01", "2025-12-12"
    forecast = get_forecast_on_end_date(lat, lon, start_date, end_date)
    print(forecast)
