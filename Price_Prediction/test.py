# -*- coding: utf-8 -*-
# predict_evaluate_crops_multi_location.py

import pandas as pd
import numpy as np
import requests
import torch
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
import datetime
import json
import os
import warnings
import gc
import joblib
import time
from dateutil.relativedelta import relativedelta
import re # For parsing filenames

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
MODEL_DIR = "output_nbeats_trained_v3"  # Directory WHERE TRAINED MODELS/SCALERS ARE SAVED
DATA_DIR = "."                         # Directory containing the original CROP CSV data
FUTURE_WEATHER_CSV_DIR = "/home/raj_99/Projects/PragatiAI/DONE/csvs" # <<<<<<<< CORRECTED PATH
OUTPUT_DIR = "predictions_output_multi_location" # Main output directory
# JSON filenames will be saved within location subdirectories
OUTPUT_JSON_FILE = "crop_predictions_forecast.json"
OUTPUT_SIMPLE_JSON_FILE = "crop_predictions_simple.json"

CROP_FILES = {
    "Jowar": "Jowar.csv", "Maize": "Maize.csv", "Mango": "Mango.csv",
    "Onion": "Onion.csv", "Potato": "Potato.csv", "Rice": "Rice.csv",
    "Wheat": "Wheat.csv",
}

# --- Parameters (Match Training) ---
INPUT_CHUNK_LENGTH = 90
EXPECTED_HISTORICAL_WEATHER_COLS = [
    'weather_code', 'temp_max', 'temp_min', 'precip_sum', 'radiation_sum', 'wind_max'
]
EXPECTED_FUTURE_MONTHLY_COLS = [
    'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
    'wind_speed_10m_max', 'shortwave_radiation_sum'
]

# --- Prediction Settings ---
PREDICT_START_DATE_STR = "2025-05-03" # Universal prediction start date
N_MONTHS_PREDICT = 8
N_DAYS_PREDICT = N_MONTHS_PREDICT * 30 + 4 # Approx 8 months (use 244 for better average)

HISTORY_PLOT_DAYS = 365

# --- Helper Functions (Keep them as they were) ---
def get_historical_weather(lat, lon, start_date, end_date):
    """Fetches historical DAILY weather data from Open-Meteo Archive API."""
    print(f"  Fetching historical DAILY weather: {start_date} to {end_date} for Lat={lat:.4f}, Lon={lon:.4f}...")
    api_cols = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum,wind_speed_10m_max"
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = { "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date, "daily": api_cols, "timezone": "auto"}
    try:
        response = requests.get(url, params=params, timeout=60); response.raise_for_status(); data = response.json()
        if 'daily' not in data or 'time' not in data['daily']: print("  Warn: No 'daily' hist data."); return None
        df = pd.DataFrame(data['daily']); df['time'] = pd.to_datetime(df['time']); df = df.set_index('time')
        rename_map = { 'temperature_2m_max': 'temp_max', 'temperature_2m_min': 'temp_min', 'precipitation_sum': 'precip_sum', 'shortwave_radiation_sum': 'radiation_sum', 'wind_speed_10m_max': 'wind_max', 'weather_code': 'weather_code'}
        df = df.rename(columns=rename_map)
        for col in EXPECTED_HISTORICAL_WEATHER_COLS:
             if col in df.columns:
                 if col == 'radiation_sum': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.01); df[col] = df[col].replace(0, 0.01)
                 elif col == 'weather_code': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                 elif col in ['temp_max', 'temp_min']: df[col] = pd.to_numeric(df[col], errors='coerce')
                 else: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
             else: df[col] = 0
        if 'temp_max' in df.columns: df['temp_max'] = df['temp_max'].ffill().bfill()
        if 'temp_min' in df.columns: df['temp_min'] = df['temp_min'].ffill().bfill()
        df = df.fillna(0)
        print("  Historical DAILY weather fetched."); return df[EXPECTED_HISTORICAL_WEATHER_COLS]
    except Exception as e: print(f"  Error fetching historical DAILY weather: {e}"); return None

def load_crop_data_for_prediction(filepath, required_end_date, history_needed):
    """Loads crop CSV, processes, ffills, returns LAST 'history_needed' rows ending ON OR BEFORE 'required_end_date'."""
    try:
        # print(f"  Loading crop data from {os.path.basename(filepath)}...") # Less verbose now
        df = pd.read_csv(filepath)
        date_col = None; price_col = None
        if 't' in df.columns: date_col = 't'
        elif 'dt' in df.columns: date_col = 'dt'
        elif 'date' in df.columns: date_col = 'date'
        elif len(df.columns) > 0: date_col = df.columns[0]
        else: raise ValueError("No date col.")
        if 'p_modal' in df.columns: price_col = 'p_modal'
        elif 'modal_price' in df.columns: price_col = 'modal_price'
        elif 'price' in df.columns: price_col = 'price'
        else: pot_cols = [c for c in df.columns if 'modal' in c.lower() or 'price' in c.lower()]; price_col = pot_cols[0] if pot_cols else None
        if not price_col: raise ValueError("No price col.")

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce'); df = df.dropna(subset=[date_col])
        if df.empty: return None, None # Return None if empty early
        if df.duplicated(subset=[date_col]).any(): df = df.drop_duplicates(subset=[date_col], keep='last')
        df = df.set_index(date_col); df = df.sort_index()
        if not df.index.is_unique: raise ValueError(f"Index not unique: {filepath}")
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df = df[[price_col]].rename(columns={price_col: 'price'})
        df = df.asfreq('D'); df['price'] = df['price'].ffill();
        df = df.dropna(subset=['price'])
        required_end_dt = pd.to_datetime(required_end_date)
        df_filtered = df[df.index <= required_end_dt]
        if len(df_filtered) < history_needed:
            # print(f"  Warn: Not enough historical ({len(df_filtered)}) for {os.path.basename(filepath)} before {required_end_date.date()}. Trying latest.") # Less verbose
             if len(df) >= history_needed:
                  # print("  Using the latest available data chunk instead.")
                  df_final_chunk = df.tail(history_needed)
             else: raise ValueError(f"Insufficient historical data overall ({len(df)}).")
        else: df_final_chunk = df_filtered.tail(history_needed)
        actual_hist_end_date = df_final_chunk.index.max()
        # print(f"  Using price data ending {actual_hist_end_date.date()} ({len(df_final_chunk)} records).") # Less verbose
        return df_final_chunk, actual_hist_end_date
    except FileNotFoundError: print(f"  Error: Crop file not found: {filepath}"); return None, None
    except Exception as e: print(f"  Error processing {filepath}: {e}"); return None, None

def load_future_monthly_weather(lat, lon, weather_filename, csv_dir): # Pass filename directly
    """Loads the specific future monthly weather CSV."""
    filepath = os.path.join(csv_dir, weather_filename)
    print(f"  Loading future MONTHLY weather for {lat:.4f},{lon:.4f} from: {weather_filename}")
    if not os.path.exists(filepath):
        print(f"  Warning: File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        predict_start_dt = pd.to_datetime(PREDICT_START_DATE_STR)
        current_year = predict_start_dt.year
        current_month = predict_start_dt.month
        dates = []
        temp_month = current_month -1;
        if temp_month == 0: temp_month = 12
        temp_year = current_year if current_month > 1 else current_year -1
        first_csv_month = df['month'].iloc[0]
        if first_csv_month < current_month and not (current_month == 12 and first_csv_month == 1) : temp_year = current_year
        elif first_csv_month == current_month: temp_year = current_year
        else: temp_year = current_year
        for m in df['month']:
             if m < temp_month: temp_year += 1
             temp_month = m
             try: dates.append(pd.Timestamp(year=temp_year, month=m, day=1))
             except ValueError: print(f"  Warn: Parse fail Year={temp_year}, Month={m}."); dates.append(pd.NaT)
        df['time'] = dates
        df = df.dropna(subset=['time'])
        df = df.set_index('time')
        cols_to_return = [col for col in EXPECTED_FUTURE_MONTHLY_COLS if col in df.columns]
        missing_cols = [col for col in EXPECTED_FUTURE_MONTHLY_COLS if col not in df.columns]
        if missing_cols: print(f"    Warning: Missing expected future weather columns: {missing_cols}")
        return df[cols_to_return]
    except Exception as e: print(f"  Error loading/processing {filepath}: {e}"); return None

def map_monthly_to_daily(daily_dates_index, df_monthly_weather):
    """Maps monthly average weather to a daily index."""
    if df_monthly_weather is None or df_monthly_weather.empty: return None
    try:
        df_daily = pd.DataFrame(index=daily_dates_index); df_daily['year'] = df_daily.index.year; df_daily['month'] = df_daily.index.month
        df_monthly_weather_reset = df_monthly_weather.copy(); df_monthly_weather_reset['year'] = df_monthly_weather_reset.index.year; df_monthly_weather_reset['month'] = df_monthly_weather_reset.index.month
        df_mapped = pd.merge(df_daily.reset_index(), df_monthly_weather_reset.reset_index(drop=True), on=['year', 'month'], how='left')
        index_name = daily_dates_index.name if daily_dates_index.name else 'index'; df_mapped = df_mapped.set_index(index_name)
        weather_cols = [col for col in df_monthly_weather.columns if col not in ['year', 'month']]; df_result = df_mapped[weather_cols]
        weather_dict = {}
        for col in df_result.columns: filled_values = df_result[col].ffill().bfill(); weather_dict[col] = np.round(filled_values.fillna(0).tolist(), 2).tolist()
        return weather_dict
    except Exception as e: print(f"  Error during monthly weather mapping: {e}"); return None

def plot_forecast(hist_series, forecast_series, crop_name, lat, lon, filename): # Added lat/lon args
    """Generates and saves a plot of historical data and forecast."""
    plt.figure(figsize=(12, 6))
    if hist_series is not None and not hist_series.empty: hist_series.plot(label=f'Historical Price (Last {len(hist_series)} Days)')
    if forecast_series is not None and not forecast_series.empty: forecast_series.plot(label=f'Forecast Price ({len(forecast_series)} days)', color='red')
    plt.title(f'{crop_name} Price Forecast (Lat: {lat:.4f}, Lon: {lon:.4f})') # Use passed lat/lon
    plt.xlabel('Date'); plt.ylabel('Price'); plt.legend(); plt.grid(True); plt.tight_layout()
    try: plt.savefig(filename); print(f"  Forecast plot saved: {filename}")
    except Exception as e: print(f"  Error saving plot {filename}: {e}")
    plt.close()

def parse_lat_lon_from_filename(filename):
    """Extracts lat and lon from filenames like LAT_LON.csv using regex."""
    # Regex to capture floating point numbers separated by underscore before .csv
    match = re.match(r"^(-?\d+\.?\d*)_(-?\d+\.?\d*)\.csv$", filename)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return lat, lon
        except ValueError:
            return None, None # Handle case where parts are not valid floats
    return None, None


# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting Multi-Location Crop Price Prediction Script ---")
    overall_start_time = time.time()

    predict_start_date = pd.to_datetime(PREDICT_START_DATE_STR)
    hist_required_end_date = predict_start_date - datetime.timedelta(days=1)
    hist_required_start_date = hist_required_end_date - datetime.timedelta(days=INPUT_CHUNK_LENGTH - 1)

    print(f"Prediction Start Date: {predict_start_date.date()}")
    print(f"Required Historical Window: {hist_required_start_date.date()} to {hist_required_end_date.date()}")
    print(f"Prediction Horizon: {N_DAYS_PREDICT} days (approx {N_MONTHS_PREDICT} months)")

    # --- Create Base Output Directory ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created base output directory: {OUTPUT_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Calculate Required Future Covariate End Date (Conservative) ---
    required_covariate_future_end_date = predict_start_date + relativedelta(days=N_DAYS_PREDICT - 1)
    print(f"Calculating required covariate end date conservatively: {required_covariate_future_end_date.date()}")

    # --- Outer Loop: Iterate through Locations (Weather CSVs) ---
    # Use the corrected path here
    if not os.path.isdir(FUTURE_WEATHER_CSV_DIR):
        print(f"Error: Specified FUTURE_WEATHER_CSV_DIR is not a valid directory: {FUTURE_WEATHER_CSV_DIR}")
        exit()

    location_files = [f for f in os.listdir(FUTURE_WEATHER_CSV_DIR) if re.match(r"^(-?\d+\.?\d*)_(-?\d+\.?\d*)\.csv$", f)]
    print(f"Found {len(location_files)} potential location weather files in '{FUTURE_WEATHER_CSV_DIR}'.")

    if not location_files:
         print("Error: No weather CSV files found in the specified directory matching the LAT_LON.csv pattern. Exiting.")
         exit()

    for weather_filename in location_files:
        # --- Parse Lat/Lon from filename ---
        predict_lat, predict_lon = parse_lat_lon_from_filename(weather_filename)
        if predict_lat is None or predict_lon is None:
            print(f"\n---> Skipping file: Could not parse lat/lon from '{weather_filename}'")
            continue

        location_id = f"{predict_lat}_{predict_lon}" # Use for directory naming
        location_output_dir = os.path.join(OUTPUT_DIR, location_id)

        print(f"\n{'='*15} Processing Location: Lat={predict_lat:.4f}, Lon={predict_lon:.4f} {'='*15}")
        print(f"Output will be saved to: {location_output_dir}")

        # --- Create Location-Specific Output Directory ---
        if not os.path.exists(location_output_dir):
            os.makedirs(location_output_dir)

        # --- Initialize outputs for THIS location ---
        location_all_predictions = {}
        location_simple_predictions = {}

        # --- Load Future Monthly Weather for THIS location ---
        df_future_weather_monthly = load_future_monthly_weather(predict_lat, predict_lon, weather_filename, FUTURE_WEATHER_CSV_DIR)
        if df_future_weather_monthly is None:
            print(f"Warning: Could not load future monthly weather for {location_id}. Predictions for this location will lack weather context.")

        # --- Inner Loop: Iterate through Crops ---
        for crop_name, csv_filename in CROP_FILES.items():
            print(f"\n  ----- Processing Crop: {crop_name} for Location: {location_id} -----")
            model_path = os.path.join(MODEL_DIR, f"{crop_name}_nbeats_model_best.pt")
            scaler_target_path = os.path.join(MODEL_DIR, f"{crop_name}_scaler_target.pkl")
            scaler_covariate_path = os.path.join(MODEL_DIR, f"{crop_name}_scaler_covariate.pkl")
            csv_path = os.path.join(DATA_DIR, csv_filename) # Crop data path remains the same

            required_files = [model_path, scaler_target_path, scaler_covariate_path, csv_path]
            if not all(os.path.exists(p) for p in required_files):
                missing = [p for p in required_files if not os.path.exists(p)]
                print(f"    Skipping {crop_name}: Missing file(s): {', '.join(os.path.basename(mf) for mf in missing)}")
                continue

            # Initialize inner loop variables
            model, target_scaler, covariate_scaler = None, None, None
            df_price_hist_input, df_weather_hist_input = None, None
            series_target_input_scaled, series_past_covariates_extended_scaled = None, None
            prediction_scaled, prediction, forecast_pd_series = None, None, None

            try:
                # --- Load Model & Scalers ---
                # print(f"    Loading artifacts for {crop_name}...") # Less verbose
                model = NBEATSModel.load(model_path, map_location=device)
                target_scaler = joblib.load(scaler_target_path)
                covariate_scaler = joblib.load(scaler_covariate_path)

                # --- Load Historical Price Data ---
                df_price_hist_input, actual_hist_end_date = load_crop_data_for_prediction(
                    csv_path, hist_required_end_date, INPUT_CHUNK_LENGTH)
                if df_price_hist_input is None: raise ValueError("Failed load price data.")

                # --- Fetch Historical Daily Weather (for this location) ---
                hist_weather_start_date = df_price_hist_input.index.min().strftime('%Y-%m-%d')
                hist_weather_end_date = actual_hist_end_date.strftime('%Y-%m-%d')
                df_weather_hist_input = get_historical_weather(
                    predict_lat, predict_lon, # Use current location's lat/lon
                    hist_weather_start_date, hist_weather_end_date)
                if df_weather_hist_input is None: raise ValueError("Failed fetch weather.")

                # --- Align Price and Weather History ---
                common_start = max(df_price_hist_input.index.min(), df_weather_hist_input.index.min())
                common_end = min(df_price_hist_input.index.max(), df_weather_hist_input.index.max())
                df_price_hist_input = df_price_hist_input[(df_price_hist_input.index >= common_start) & (df_price_hist_input.index <= common_end)]
                df_weather_hist_input = df_weather_hist_input.loc[df_price_hist_input.index]
                if len(df_price_hist_input) < INPUT_CHUNK_LENGTH: raise ValueError(f"Input price short ({len(df_price_hist_input)}). Need {INPUT_CHUNK_LENGTH}.")
                if len(df_weather_hist_input) != len(df_price_hist_input): raise ValueError("Input price/weather len mismatch.")
                # print(f"    Aligned historical inputs: {len(df_price_hist_input)} recs ending {df_price_hist_input.index.max().date()}.") # Less verbose

                # --- WORKAROUND: Create Extended Past Covariates ---
                last_hist_cov_date = df_weather_hist_input.index.max()
                future_end_needed = required_covariate_future_end_date # Use pre-calculated date
                df_weather_extended = df_weather_hist_input # Start with historical
                if future_end_needed > last_hist_cov_date:
                    # print(f"    Extending covariates with dummy data to {future_end_needed:%Y-%m-%d}") # Less verbose
                    future_date_range = pd.date_range(start=last_hist_cov_date + relativedelta(days=1), end=future_end_needed, freq='D')
                    df_dummy_future_weather = pd.DataFrame(0, index=future_date_range, columns=EXPECTED_HISTORICAL_WEATHER_COLS)
                    for col in EXPECTED_HISTORICAL_WEATHER_COLS:
                        if col == 'weather_code': df_dummy_future_weather[col] = df_dummy_future_weather[col].astype(int)
                        else: df_dummy_future_weather[col] = df_dummy_future_weather[col].astype(float)
                    df_weather_extended = pd.concat([df_weather_hist_input, df_dummy_future_weather])
                # else: print("    Past covariates already cover required future period.") # Less verbose

                series_past_covariates_extended = TimeSeries.from_dataframe(df_weather_extended[EXPECTED_HISTORICAL_WEATHER_COLS], freq='D')
                # --- END WORKAROUND ---

                # --- Prepare Target TimeSeries & Scale ---
                series_target_input = TimeSeries.from_dataframe(df_price_hist_input, value_cols=['price'], freq='D')
                series_target_input_scaled = target_scaler.transform(series_target_input)
                series_past_covariates_extended_scaled = covariate_scaler.transform(series_past_covariates_extended)
                # print("    Input data scaled.") # Less verbose

                # --- Generate Prediction ---
                # print(f"    Generating {N_DAYS_PREDICT}-day forecast...") # Less verbose
                prediction_start_time = time.time()
                prediction_scaled = model.predict(
                    n=N_DAYS_PREDICT,
                    series=series_target_input_scaled,
                    past_covariates=series_past_covariates_extended_scaled # Provide EXTENDED covariates
                )
                # print(f"    Prediction generated in {time.time() - prediction_start_time:.2f} sec.") # Less verbose

                # --- Inverse Transform & Format ---
                prediction = target_scaler.inverse_transform(prediction_scaled)
                prediction_index = prediction.time_index
                prediction_values = prediction.values(copy=False).flatten()
                if prediction_index is None or prediction_values is None: raise ValueError("Pred missing index/values.")
                forecast_pd_series = pd.Series(prediction_values, index=prediction_index, name='price')
                forecast_dates = forecast_pd_series.index.strftime('%Y-%m-%d').tolist()
                forecast_prices = np.round(forecast_pd_series.values, 2).tolist()

                # --- Map Monthly Weather ---
                # print("    Mapping monthly weather...") # Less verbose
                mapped_weather_dict = map_monthly_to_daily(forecast_pd_series.index, df_future_weather_monthly)

                # --- Store Results for this location ---
                crop_result = {"dates": forecast_dates, "prices": forecast_prices}
                if mapped_weather_dict: crop_result["future_monthly_weather_estimates"] = mapped_weather_dict
                location_all_predictions[crop_name] = crop_result
                location_simple_predictions[crop_name] = {"dates": forecast_dates, "prices": forecast_prices} # Added
                # print(f"    Formatted output for {crop_name} stored.") # Less verbose

                # --- Generate Plot (Save in location subdirectory) ---
                plot_filename = os.path.join(location_output_dir, f"{crop_name}_forecast_plot.png")
                history_to_plot = pd.Series(dtype=float)
                try:
                     df_full_hist_for_plot, _ = load_crop_data_for_prediction(csv_path, hist_required_end_date, HISTORY_PLOT_DAYS)
                     if df_full_hist_for_plot is not None and not df_full_hist_for_plot.empty: history_to_plot = df_full_hist_for_plot['price']
                     if not history_to_plot.empty or not forecast_pd_series.empty: # Plot if at least one has data
                          plot_forecast(history_to_plot, forecast_pd_series, crop_name, predict_lat, predict_lon, plot_filename) # Pass lat/lon
                     # else: print(f"    Skipping plot for {crop_name} (no data).") # Less verbose
                except Exception as plot_e: print(f"    Warn: Could not generate plot for {crop_name}: {plot_e}")


            except Exception as e:
                print(f"\n    !!!!!! Error processing {crop_name} for {location_id}: {e} !!!!!!")
                # import traceback; traceback.print_exc() # Optionally uncomment for full trace
            finally:
                # Cleanup inner loop vars
                del model, target_scaler, covariate_scaler
                if 'series_target_input_scaled' in locals(): del series_target_input_scaled
                if 'prediction_scaled' in locals(): del prediction_scaled
                if 'prediction' in locals(): del prediction
                if 'df_price_hist_input' in locals(): del df_price_hist_input
                if 'df_weather_hist_input' in locals(): del df_weather_hist_input
                if 'series_target_input' in locals(): del series_target_input
                if 'series_past_covariates_extended_scaled' in locals(): del series_past_covariates_extended_scaled
                if 'forecast_pd_series' in locals(): del forecast_pd_series
                if 'df_weather_extended' in locals(): del df_weather_extended
                if 'series_past_covariates_extended' in locals(): del series_past_covariates_extended
                gc.collect();
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- End of Inner Crop Loop ---

        # --- Save JSON files for THIS location ---
        print(f"\n  --- Saving Predictions for Location: {location_id} ---")
        output_json_path = os.path.join(location_output_dir, OUTPUT_JSON_FILE)
        try:
            with open(output_json_path, 'w') as f: json.dump(location_all_predictions, f, indent=4)
            print(f"  Detailed predictions saved to {output_json_path}")
        except Exception as e: print(f"  Error saving detailed prediction JSON: {e}")

        output_simple_json_path = os.path.join(location_output_dir, OUTPUT_SIMPLE_JSON_FILE)
        try:
            with open(output_simple_json_path, 'w') as f: json.dump(location_simple_predictions, f, indent=4)
            print(f"  Simple predictions saved to {output_simple_json_path}")
        except Exception as e: print(f"  Error saving simple prediction JSON: {e}")

        # --- End of Outer Location Loop ---

    print(f"\n{'='*20} Multi-Location Prediction Finished {'='*20}")
    print(f"Outputs saved in subdirectories within: {OUTPUT_DIR}")
    print(f"Total script execution time: {time.time() - overall_start_time:.2f} seconds")
