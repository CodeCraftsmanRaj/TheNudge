# -*- coding: utf-8 -*-
# --- IMPORTS ---
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger # For logging train/val loss
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import datetime
import json
import os
import warnings
import gc
import time
import joblib # For saving scalers

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
CROP_FILES = {
    "Jowar": "Jowar.csv", "Maize": "Maize.csv", "Mango": "Mango.csv",
    "Onion": "Onion.csv", "Potato": "Potato.csv", "Rice": "Rice.csv",
    "Wheat": "Wheat.csv",
}
DATA_DIR = "." # Directory containing crop CSVs
OUTPUT_DIR = "output_nbeats_trained_v3" # Models, scalers, logs, plots saved here

# Model Hyperparameters
INPUT_CHUNK_LENGTH = 90
OUTPUT_CHUNK_LENGTH = 16 # How many steps the model outputs at once.
N_EPOCHS = 42 # Set requested number of epochs
# Define weather columns expected from the *historical* weather source
EXPECTED_HISTORICAL_WEATHER_COLS = [
    'weather_code',         # Example column from historical API
    'temp_max',             # Renamed from 'temperature_2m_max'
    'temp_min',             # Renamed from 'temperature_2m_min'
    'precip_sum',           # Renamed from 'precipitation_sum'
    'radiation_sum',        # Renamed from 'shortwave_radiation_sum'
    'wind_max'              # Renamed from 'wind_speed_10m_max'
]
# Location for fetching HISTORICAL weather (change if needed)
FETCH_LAT = 19.0760 # Mumbai Latitude
FETCH_LON = 72.8777 # Mumbai Longitude

VALIDATION_SPLIT_RATIO = 0.20 # Use 20% of the data for validation

# --- Weather Data Function (Historical Only) ---
def get_historical_weather(lat, lon, start_date, end_date):
    """
    Fetches historical daily weather data from Open-Meteo Archive API.
    Uses columns defined in EXPECTED_HISTORICAL_WEATHER_COLS.
    """
    print(f"Fetching historical weather: {start_date} to {end_date} for Lat={lat}, Lon={lon}...")
    api_cols = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum,wind_speed_10m_max"
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "daily": api_cols, "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        if 'daily' not in data or 'time' not in data['daily']:
            print("Warning: No 'daily' data found in historical weather response.")
            return None
        df = pd.DataFrame(data['daily']); df['time'] = pd.to_datetime(df['time']); df = df.set_index('time')
        rename_map = {
            'temperature_2m_max': 'temp_max', 'temperature_2m_min': 'temp_min',
            'precipitation_sum': 'precip_sum', 'shortwave_radiation_sum': 'radiation_sum',
            'wind_speed_10m_max': 'wind_max', 'weather_code': 'weather_code'
        }
        df = df.rename(columns=rename_map)
        for col in EXPECTED_HISTORICAL_WEATHER_COLS:
            if col in df.columns:
                if col == 'radiation_sum': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.01); df[col] = df[col].replace(0, 0.01)
                elif col == 'weather_code': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                elif col in ['temp_max', 'temp_min']: df[col] = pd.to_numeric(df[col], errors='coerce')
                else: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else: print(f"  Warning: Historical weather column '{col}' not found. Filling with 0."); df[col] = 0
        if 'temp_max' in df.columns: df['temp_max'] = df['temp_max'].ffill().bfill()
        if 'temp_min' in df.columns: df['temp_min'] = df['temp_min'].ffill().bfill()
        df = df.fillna(0)
        print("Historical weather fetched successfully.")
        return df[EXPECTED_HISTORICAL_WEATHER_COLS]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical weather: {e}"); import traceback; traceback.print_exc(); return None
    except Exception as e:
        print(f"Error processing historical weather data: {e}"); import traceback; traceback.print_exc(); return None

# --- Data Loading (Crop Prices) ---
def load_and_preprocess_crop_data(filepath):
    """Loads and preprocesses crop price data from a CSV file."""
    try:
        print(f"Loading crop data: {os.path.basename(filepath)}...")
        df = pd.read_csv(filepath); date_col = None; price_col = None
        # Detect date column
        if 't' in df.columns: date_col = 't'
        elif 'dt' in df.columns: date_col = 'dt'
        elif 'date' in df.columns: date_col = 'date'
        elif len(df.columns) > 0: date_col = df.columns[0]; print(f"  Warning: Assuming date col '{date_col}'.")
        else: raise ValueError("No date col.")
        # Detect price column
        if 'p_modal' in df.columns: price_col = 'p_modal'
        elif 'modal_price' in df.columns: price_col = 'modal_price'
        elif 'price' in df.columns: price_col = 'price'
        else:
            potential_price_cols = [c for c in df.columns if 'modal' in c.lower() or 'price' in c.lower()]
            if potential_price_cols: price_col = potential_price_cols[0]; print(f"  Warning: Assuming price col '{price_col}'.")
            else: raise ValueError("No price col.")
        print(f"  Identified Date Col: '{date_col}', Price Col: '{price_col}'")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce'); df = df.dropna(subset=[date_col])
        if df.empty: print("  Warning: Empty after date conversion."); return None
        print(f"  Shape before dedupe: {df.shape}")
        if df.duplicated(subset=[date_col]).any():
            num_dupes = df[df.duplicated(subset=[date_col], keep=False)].shape[0]; print(f"  Found {num_dupes} duplicate dates. Keeping last."); df = df.drop_duplicates(subset=[date_col], keep='last')
        print(f"  Shape after dedupe: {df.shape}"); df = df.set_index(date_col); df = df.sort_index()
        if not df.index.is_unique: raise ValueError(f"Index not unique: {filepath}")
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce'); df = df[[price_col]].rename(columns={price_col: 'price'})
        print(f"  Resampling to daily ('D')..."); df = df.asfreq('D'); missing_dates_count = df['price'].isnull().sum()
        if missing_dates_count > 0: print(f"  Found {missing_dates_count} missing dates after resampling.")
        df['price'] = df['price'].ffill(); remaining_nans = df['price'].isnull().sum()
        if remaining_nans > 0: print(f"  Warning: {remaining_nans} NaNs remain after ffill. Dropping."); df = df.dropna(subset=['price'])
        else: print(f"  Forward filled missing prices."); print(f"  Final shape: {df.shape}")
        if df.empty: print("Warning: DataFrame empty after processing."); return None
        return df
    except FileNotFoundError: print(f"Error: Crop data file not found: {filepath}"); return None
    except ValueError as ve: print(f"Error processing crop data {filepath}: {ve}"); return None
    except Exception as e: print(f"Unexpected error processing {filepath}: {e}"); import traceback; traceback.print_exc(); return None

# --- Plotting Function for Training Logs ---
def plot_training_logs(log_file, plot_filename, crop_name):
    """Reads CSV log file and plots training/validation loss."""
    try:
        log_data = pd.read_csv(log_file)
        plt.figure(figsize=(10, 6))
        # Plot validation loss (logged per epoch)
        if 'val_loss' in log_data.columns and 'epoch' in log_data.columns:
            val_data = log_data.dropna(subset=['val_loss'])
            plt.plot(val_data['epoch'], val_data['val_loss'], label='Validation Loss', marker='.')
        else: print(f"  Warning: 'val_loss' or 'epoch' column not found in {log_file}")
        # Plot training loss (prefer epoch-level, fallback to step-level if needed)
        train_loss_col = None
        if 'train_loss_epoch' in log_data.columns: train_loss_col = 'train_loss_epoch'
        elif 'train_loss' in log_data.columns: train_loss_col = 'train_loss'; print(f"  Info: Plotting 'train_loss' (may be step/epoch).")
        if train_loss_col and 'epoch' in log_data.columns:
             train_data = log_data.dropna(subset=[train_loss_col, 'epoch'])
             # Group by epoch and take the mean if train_loss is step-wise
             if train_loss_col == 'train_loss' and train_data.groupby('epoch').size().max() > 1:
                 print("  Info: Averaging step-wise training loss per epoch for plotting.")
                 train_data_epoch = train_data.groupby('epoch')[train_loss_col].mean().reset_index()
                 plt.plot(train_data_epoch['epoch'], train_data_epoch[train_loss_col], label='Avg Training Loss per Epoch', marker='.', linestyle='--')
             else: # Plot directly if it's already epoch-level or only one point per epoch
                 plt.plot(train_data['epoch'], train_data[train_loss_col], label='Training Loss', marker='.', linestyle='--')
        elif 'epoch' in log_data.columns: print(f"  Warning: Training loss column not found in {log_file}")
        plt.title(f'{crop_name} Model Training History (Epochs: {N_EPOCHS})')
        plt.xlabel('Epoch'); plt.ylabel('Loss (MAE)'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(plot_filename); plt.close()
        print(f"  Training plot saved to {plot_filename}")
    except FileNotFoundError: print(f"  Error: Log file not found for plotting: {log_file}")
    except Exception as e: print(f"  Error generating training plot from {log_file}: {e}")


# --- N-BEATS Training Function ---
def train_nbeats_model(
    df_hist_price, df_hist_weather, crop_name, output_dir,
    input_chunk_length, output_chunk_length, n_epochs, validation_split_ratio
    ):
    """Trains N-BEATS model, logs metrics, saves best model, plots training."""
    print(f"--- Starting N-BEATS Training for {crop_name} ---")
    training_successful = False # Assume failure initially
    log_file_path = None
    gpu_available = torch.cuda.is_available()

    try:
        # --- 1. Data Preparation and TimeSeries Conversion ---
        df_hist_price = df_hist_price.asfreq('D'); df_hist_weather = df_hist_weather.asfreq('D')
        series_target = TimeSeries.from_dataframe(df_hist_price, value_cols=['price'], freq='D')
        series_past_covariates = TimeSeries.from_dataframe(df_hist_weather, freq='D', fill_missing_dates=True, fillna_value=0)
        print(f"  TimeSeries created. Target: {len(series_target)}, Past Cov: {len(series_past_covariates)}")

        # --- 2. Align historical series ---
        series_past_covariates = series_past_covariates.slice_intersect(series_target)
        series_target = series_target.slice_intersect(series_past_covariates)
        if len(series_target) != len(series_past_covariates): raise ValueError("Lengths differ after align.")
        if len(series_target) == 0: raise ValueError("TimeSeries empty after align.")
        print(f"  TimeSeries aligned. Length: {len(series_target)} ({series_target.start_time()} to {series_target.end_time()})")

        # --- 3. Check Minimum Data Length ---
        min_length_required = input_chunk_length + output_chunk_length + 1
        if len(series_target) < min_length_required: raise ValueError(f"Insufficient data ({len(series_target)}). Need {min_length_required}.")

        # --- 4. Create Validation Set ---
        split_index = int(len(series_target) * (1 - validation_split_ratio)); split_index = max(input_chunk_length, split_index)
        split_index = min(split_index, len(series_target) - (input_chunk_length + output_chunk_length))
        if split_index <= input_chunk_length: raise ValueError(f"Split index ({split_index}) too early.")
        split_timestamp = series_target.time_index[split_index]
        train_target, val_target = series_target.split_after(split_timestamp)
        train_past_cov, val_past_cov = series_past_covariates.split_after(split_timestamp)
        if len(train_target) < input_chunk_length: raise ValueError(f"Train target ({len(train_target)}) < input_chunk ({input_chunk_length}).")
        if len(val_target) < output_chunk_length + 1: raise ValueError(f"Val target ({len(val_target)}) too short.")
        print(f"  Data split for validation at {split_timestamp}. Train: {len(train_target)}, Val: {len(val_target)}")

        # --- 5. Scaling ---
        target_scaler = Scaler(); covariate_scaler = Scaler()
        train_target_scaled = target_scaler.fit_transform(train_target)
        train_past_cov_scaled = covariate_scaler.fit_transform(train_past_cov)
        val_target_scaled = target_scaler.transform(val_target)
        val_past_cov_scaled = covariate_scaler.transform(val_past_cov)
        print("  Data scaling applied.")

        # --- 6. Model Setup and Training ---
        print(f"  Configuring N-BEATS model (Input: {input_chunk_length}, Output: {output_chunk_length}, Epochs: {n_epochs})...")
        print(f"  CUDA available: {gpu_available}")

        # Logger setup
        log_dir = os.path.join(output_dir, "logs"); csv_logger = CSVLogger(save_dir=log_dir, name=crop_name)
        print(f"  CSVLogger initialized. Logs will be saved in: {csv_logger.log_dir}")

        # Checkpoint setup (saves best model based on val_loss)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

        # Trainer setup
        trainer_kwargs = {
            "accelerator": "gpu" if gpu_available else "cpu", "devices": 1 if gpu_available else "auto",
            "enable_progress_bar": True, "callbacks": [checkpoint_callback], "enable_checkpointing": True,
            "logger": csv_logger }
        print(f"  Trainer kwargs: {trainer_kwargs}")

        # Model definition
        model = NBEATSModel(
            input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=n_epochs,
            random_state=42, generic_architecture=True, num_stacks=30, num_blocks=1, num_layers=4,
            layer_widths=256, loss_fn=nn.L1Loss(), optimizer_cls=torch.optim.Adam, optimizer_kwargs={'lr': 1e-3},
            lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau, lr_scheduler_kwargs={'patience': 5},
            batch_size=128, pl_trainer_kwargs=trainer_kwargs)

        # --- Training ---
        print("  Starting model training...")
        start_train_time = time.time()
        model.fit(
            series=train_target_scaled, past_covariates=train_past_cov_scaled,
            val_series=val_target_scaled, val_past_covariates=val_past_cov_scaled, verbose=True)
        print(f"  Training completed in {time.time() - start_train_time:.2f} seconds.")
        print("  Best model based on validation loss automatically loaded.")
        training_successful = True # Mark success only after fit completes
        log_file_path = os.path.join(csv_logger.log_dir, "metrics.csv")

        # --- 7. Save Model/Scalers ---
        model_save_path = os.path.join(output_dir, f"{crop_name}_nbeats_model_best.pt")
        scaler_target_save_path = os.path.join(output_dir, f"{crop_name}_scaler_target.pkl")
        scaler_covariate_save_path = os.path.join(output_dir, f"{crop_name}_scaler_covariate.pkl")
        try:
            print(f"  Saving best model to {model_save_path}"); model.save(model_save_path)
            print(f"  Saving target scaler to {scaler_target_save_path}"); joblib.dump(target_scaler, scaler_target_save_path)
            print(f"  Saving covariate scaler to {scaler_covariate_save_path}"); joblib.dump(covariate_scaler, scaler_covariate_save_path)
        except Exception as e: print(f"  Error saving model or scalers: {e}"); training_successful = False # Mark failure

        # --- 8. Plot Training Logs ---
        if log_file_path and os.path.exists(log_file_path):
            plot_filename = os.path.join(output_dir, f"{crop_name}_training_plot.png")
            plot_training_logs(log_file_path, plot_filename, crop_name)
        else: print(f"  Warning: Log file path not found or invalid ({log_file_path}), skipping training plot.")

        # --- 9. Optional Simulation ---
        print("  Simulating prediction from end of training data (for verification)...")
        try:
            sim_prediction_scaled = model.predict(n=output_chunk_length, series=train_target_scaled, past_covariates=train_past_cov_scaled)
            sim_prediction = target_scaler.inverse_transform(sim_prediction_scaled)
            prediction_index = sim_prediction.time_index; prediction_values = sim_prediction.values(copy=False).flatten()
            if prediction_index is not None and prediction_values is not None and len(prediction_values) > 0:
                sim_prediction_series = pd.Series(prediction_values, index=prediction_index); example_value = sim_prediction_series.iloc[0]
                print(f"  Simulated prediction successful. Example value: {example_value:.2f}")
            else: print("  Simulated prediction ran but no valid index/values found.")
        except AttributeError as ae: print(f"  Warning: Error accessing simulation components: {ae}")
        except Exception as e: print(f"  Warning: Error during prediction simulation: {e}")

    except Exception as e: # Catch errors during setup/training
        print(f"  *** Error during training setup or execution for {crop_name}: {e} ***")
        import traceback; traceback.print_exc(); training_successful = False
    finally:
        # --- 10. Cleanup ---
        print("  Cleaning up training resources...")
        # Safely delete objects if they exist
        if 'model' in locals() and model is not None: del model
        if 'target_scaler' in locals() and target_scaler is not None: del target_scaler
        if 'covariate_scaler' in locals() and covariate_scaler is not None: del covariate_scaler
        if 'csv_logger' in locals() and csv_logger is not None: del csv_logger
        if 'train_target_scaled' in locals(): del train_target_scaled
        if 'train_past_cov_scaled' in locals(): del train_past_cov_scaled
        if 'val_target_scaled' in locals(): del val_target_scaled
        if 'val_past_cov_scaled' in locals(): del val_past_cov_scaled
        if 'sim_prediction_scaled' in locals(): del sim_prediction_scaled
        if 'sim_prediction' in locals(): del sim_prediction
        gc.collect()
        if gpu_available: torch.cuda.empty_cache()

    print(f"--- Finished N-BEATS Training for {crop_name} ---")
    return training_successful # Return success status

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Crop Price Model Training Script ---")
    start_time = time.time()
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR); print(f"Creating output directory: {OUTPUT_DIR}")
    all_crop_data = {}; min_date_hist = None; max_date_hist = None
    print("\n--- Loading All Crop Price Data ---")
    for crop, filename in CROP_FILES.items():
        filepath = os.path.join(DATA_DIR, filename); df_crop = load_and_preprocess_crop_data(filepath)
        if df_crop is not None and not df_crop.empty:
            all_crop_data[crop] = df_crop; current_min = df_crop.index.min(); current_max = df_crop.index.max()
            if min_date_hist is None or current_min < min_date_hist: min_date_hist = current_min
            if max_date_hist is None or current_max > max_date_hist: max_date_hist = current_max
            print(f"  Loaded {crop}: {len(df_crop)} records from {current_min.date()} to {current_max.date()}")
        else: print(f"  Skipping {crop}.")
    if not all_crop_data: print("\nError: No crop data loaded. Exiting."); exit()
    print(f"\nOverall historical data range needed: {min_date_hist.date()} to {max_date_hist.date()}")
    hist_weather_start_date = min_date_hist - datetime.timedelta(days=INPUT_CHUNK_LENGTH + 5)
    hist_weather_end_date = max_date_hist
    print(f"\n--- Fetching Historical Weather for Training ({FETCH_LAT}, {FETCH_LON}) ---")
    print(f"  Requesting data from {hist_weather_start_date.date()} to {hist_weather_end_date.date()}")
    historical_weather_all = get_historical_weather(FETCH_LAT, FETCH_LON, hist_weather_start_date.strftime('%Y-%m-%d'), hist_weather_end_date.strftime('%Y-%m-%d'))
    if historical_weather_all is None or historical_weather_all.empty: print("\nError: Failed fetch weather. Exiting."); exit()
    print(f"  Hist weather fetched: {len(historical_weather_all)} records from {historical_weather_all.index.min().date()} to {historical_weather_all.index.max().date()}")
    print(f"  Hist weather columns: {historical_weather_all.columns.tolist()}")
    print("\n--- Training Models for Each Crop ---")
    successful_trainings = []; failed_trainings = []
    for crop, df_price_hist in all_crop_data.items():
        print(f"\n===== Processing Crop: {crop} =====")
        common_start = max(df_price_hist.index.min(), historical_weather_all.index.min()); common_end = min(df_price_hist.index.max(), historical_weather_all.index.max())
        df_price_aligned = df_price_hist[(df_price_hist.index >= common_start) & (df_price_hist.index <= common_end)]
        df_weather_aligned = historical_weather_all[(historical_weather_all.index >= common_start) & (historical_weather_all.index <= common_end)]
        final_start = max(df_price_aligned.index.min(), df_weather_aligned.index.min()); final_end = min(df_price_aligned.index.max(), df_weather_aligned.index.max())
        df_price_aligned = df_price_aligned[(df_price_aligned.index >= final_start) & (df_price_aligned.index <= final_end)]
        df_weather_aligned = df_weather_aligned[EXPECTED_HISTORICAL_WEATHER_COLS].loc[final_start:final_end]
        if df_price_aligned.empty or df_weather_aligned.empty: print(f"  Skipping {crop}: Empty after align."); failed_trainings.append(crop); continue
        if len(df_price_aligned) != len(df_weather_aligned): print(f"  Skipping {crop}: Lengths differ after align."); failed_trainings.append(crop); continue
        print(f"  Aligned Data for Training: {len(df_price_aligned)} records from {final_start.date()} to {final_end.date()}")
        min_len_needed = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH + 1
        if len(df_price_aligned) < min_len_needed: print(f"  Skipping {crop}: Insufficient data ({len(df_price_aligned)}), need {min_len_needed}."); failed_trainings.append(crop); continue

        training_success = train_nbeats_model(
            df_hist_price=df_price_aligned[['price']], df_hist_weather=df_weather_aligned, crop_name=crop,
            output_dir=OUTPUT_DIR, input_chunk_length=INPUT_CHUNK_LENGTH, output_chunk_length=OUTPUT_CHUNK_LENGTH,
            n_epochs=N_EPOCHS, validation_split_ratio=VALIDATION_SPLIT_RATIO)

        if training_success: successful_trainings.append(crop)
        else: failed_trainings.append(crop); print(f"  Training failed for {crop}.")
    print("\n--- Training Summary ---")
    print(f"Output directory: {OUTPUT_DIR}")
    if successful_trainings: print(f"Successfully trained models/plots saved for: {', '.join(successful_trainings)}")
    else: print("No models were trained successfully.")
    if failed_trainings: print(f"Failed to train models for: {', '.join(failed_trainings)}")
    end_time = time.time()
    print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds")
    print("--- Script Finished ---")
