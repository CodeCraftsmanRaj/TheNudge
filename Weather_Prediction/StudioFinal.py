# import requests
# import numpy as np

# def get_enhanced_forecast_on_end_date(
#     lat,
#     lon,
#     start_date,
#     end_date,
#     timezone="auto"
# ):
#     # Expanded list of available base variables
#     variables_base = [
#         "temperature_2m_max",
#         "temperature_2m_min",
#         "precipitation_sum",
#         "wind_speed_10m_max",
#         "shortwave_radiation_sum"
#     ]

#     url = "https://seasonal-api.open-meteo.com/v1/seasonal"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "daily": ",".join(variables_base),
#         "start_date": start_date,
#         "end_date": end_date,
#         "timezone": timezone
#     }

#     resp = requests.get(url, params=params)
#     if resp.status_code != 200:
#         raise Exception(f"API Error {resp.status_code}: {resp.text}")

#     data = resp.json()
#     daily = data.get("daily")
#     if not daily or "time" not in daily:
#         raise KeyError("No 'daily' time series returned.")

#     # Find index of the requested end_date
#     times = daily["time"]
#     if end_date not in times:
#         raise ValueError(f"End date {end_date} not in returned time array.")
#     idx = times.index(end_date)

#     result = {"date": end_date}

#     # For each variable, average across ensemble members (filtering out None)
#     for var in variables_base:
#         member_keys = [k for k in daily.keys() if k.startswith(f"{var}_member")]
#         vals = [daily[k][idx] for k in member_keys]
#         clean = [v for v in vals if v is not None]
#         result[var] = float(np.mean(clean)) if clean else None

#     return result


# # Example usage
# if __name__ == "__main__":
#     lat, lon = 19.0760, 72.8777  # Mumbai
#     start_date = "2025-05-01"
#     end_date = "2025-12-18"

#     forecast = get_enhanced_forecast_on_end_date(lat, lon, start_date, end_date)
#     print(forecast)

import requests
import numpy as np

def get_enhanced_forecast_on_end_date(
    lat,
    lon,
    start_date,
    end_date,
    timezone="auto" # Original default
):
    # Expanded list of available base variables (from your initial code)
    variables_base = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "shortwave_radiation_sum"
    ]

    url = "https://seasonal-api.open-meteo.com/v1/seasonal"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(variables_base),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone # Uses the timezone passed to the function
    }

    # Original API call logic
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        # NOTE: This is where the 400 error from the API bug will likely be caught
        print(f"--- Potential API Error Encountered ---")
        print(f"Status Code: {resp.status_code}")
        print(f"Reason (if available): {resp.text[:500]}...") # Print part of the response text
        print(f"This may be due to the known Open-Meteo Seasonal API bug when using the 'daily' parameter.")
        print(f"--------------------------------------")
        raise Exception(f"API Error {resp.status_code}. Check logs and potential API bug. Response: {resp.text}")

    data = resp.json()
    daily = data.get("daily")
    if not daily or "time" not in daily:
        raise KeyError("No 'daily' time series returned.")

    # Find index of the requested end_date
    times = daily["time"]
    if end_date not in times:
        raise ValueError(f"End date {end_date} not in returned time array.")
    idx = times.index(end_date)

    result = {"date": end_date}

    # For each variable, average across ensemble members (filtering out None)
    for var in variables_base:
        member_keys = [k for k in daily.keys() if k.startswith(f"{var}_member")]
        # Original logic had potential error if no members found, handle slightly better
        if not member_keys:
            print(f"Warning: No ensemble members found for variable '{var}' in the response.")
            result[var] = None
            continue # Skip this variable if no members

        # Original logic assumes index 'idx' is valid for all member arrays
        # Adding a basic check, though more robust checks were in later versions
        vals = []
        for k in member_keys:
             if k in daily and isinstance(daily[k], list) and idx < len(daily[k]):
                 vals.append(daily[k][idx])
             else:
                 # Log or handle cases where member data is missing/malformed for the index
                 # print(f"Warning: Data for member '{k}' missing or too short at index {idx}")
                 pass # Original code implicitly skipped Nones later

        clean = [v for v in vals if v is not None]
        result[var] = float(np.mean(clean)) if clean else None

    return result


# Example usage (from your initial code, modified ONLY for timezone)
if __name__ == "__main__":
    lat, lon = 19.0760, 72.8777  # Mumbai
    start_date = "2025-05-01" # Original example date
    end_date = "2025-12-18"   # Original example date
    target_timezone = "Asia/Kolkata" # Define IST timezone identifier

    print(f"Attempting to fetch forecast for {end_date} in timezone {target_timezone}...")
    print("NOTE: This request will likely fail due to a known Open-Meteo Seasonal API bug!")

    try:
        # Pass the target_timezone explicitly to the function
        forecast = get_enhanced_forecast_on_end_date(
            lat, lon,
            start_date, end_date,
            timezone=target_timezone # Explicitly set timezone here
            )
        # This part will only run if the API call somehow succeeds despite the bug
        print(f"\n--- Forecast Data (if successful) ---")
        print(forecast)
        print(f"--------------------------------------")
    except Exception as e:
        print(f"\nAn error occurred as expected (due to API bug or other issue):")
        print(e)