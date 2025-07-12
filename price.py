import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import warnings
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent harmless Tkinter errors
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- 1. MODULAR FUNCTIONS ---

def load_and_prepare_daily_data(df_main, state, district, market, crop):
    """Filters, aggregates, and prepares a daily time series for a specific combination."""
    df_filtered = df_main[(df_main['State'] == state) & 
                          (df_main['District'] == district) &
                          (df_main['Market'] == market) & 
                          (df_main['Commodity'] == crop)]
    
    if df_filtered.empty:
        return None, "Combination not found."
    
    daily_prices = df_filtered.groupby('Arrival_Date')['Modal_Price'].mean()
    daily_index = pd.date_range(start=daily_prices.index.min(), end=daily_prices.index.max(), freq='D')
    ts = daily_prices.reindex(daily_index).ffill().rename('price')
    
    if len(ts) < 60:
        return None, "Not enough historical data (< 60 days)."
        
    return pd.DataFrame(ts), None

def create_features(df):
    """Creates time-based and daily lag features."""
    df['day_of_year'] = df.index.dayofyear
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    for i in range(1, 15):
        df[f'lag_{i}'] = df['price'].shift(i)
    return df.dropna()

def evaluate_models(X, y):
    """Trains and evaluates multiple models, adding MAPE as a metric."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results, predictions_dict = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions_dict[name] = predictions
        
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-9))) * 100
        results.append({
            "Model": name,
            "R-squared": r2_score(y_test, predictions),
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "MAPE (%)": mape
        })
        
    return pd.DataFrame(results), y_test, predictions_dict

def plot_and_save_performance(y_test, predictions_dict, output_dir, combo_name):
    """Generates and saves a plot comparing model predictions on the test set."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 8))
    plt.plot(y_test, label='Actual Price', color='black', linewidth=2.5, alpha=0.8)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        plt.plot(y_test.index, preds, label=f'{model_name} Prediction', color=colors[i], linestyle='--', alpha=0.9)
        
    plt.title(f'Model Performance for {combo_name}', fontsize=16)
    plt.xlabel('Date'); plt.ylabel('Price'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{combo_name}.png")); plt.close()

def forecast_future_daily(model, full_feature_df, start_date, days_to_predict=275):
    """Forecasts future prices iteratively starting from a specific date."""
    X_full, y_full = full_feature_df.drop('price', axis=1), full_feature_df['price']
    model.fit(X_full, y_full)

    latest_features = X_full.iloc[-1:].copy()
    current_date = start_date
    
    future_predictions = []
    
    for _ in range(days_to_predict):
        latest_features['day_of_year'] = current_date.dayofyear
        latest_features['day_of_week'] = current_date.dayofweek
        latest_features['month'] = current_date.month
        latest_features['year'] = current_date.year
        
        next_pred = model.predict(latest_features)[0]
        future_predictions.append(next_pred)
        
        new_lags = [next_pred] + list(latest_features.values[0][4:])
        for j in range(1, 15):
            latest_features[f'lag_{j}'] = new_lags[j-1]
        
        current_date += pd.DateOffset(days=1)
            
    forecast_dates = pd.date_range(start=start_date, periods=days_to_predict)
    return pd.Series(future_predictions, index=forecast_dates, name='Forecasted_Price')

def create_summary_plots(summary_df, output_dir):
    """Creates and saves high-level summary plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Average R-squared per Model
    avg_r2 = summary_df.groupby('Model')['R-squared'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    avg_r2.plot(kind='bar', color=['#2ca02c', '#ff7f0e', '#1f77b4'])
    plt.title('Average R-squared Score Across All Combinations', fontsize=16)
    plt.ylabel('Average R-squared'); plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_avg_r2_per_model.png')); plt.close()

    # Plot 2: Best Model's R-squared for each combination
    pivoted_summary = summary_df.pivot_table(index='Combination', columns='Model', values='R-squared')
    best_r2 = pivoted_summary.max(axis=1).sort_values(ascending=False).head(20) # Top 20 for readability
    plt.figure(figsize=(12, 10))
    best_r2.plot(kind='barh', color='#007acc')
    plt.title("Top 20 Most Predictable Combinations (by Best Model's R²)", fontsize=16)
    plt.xlabel('Highest R-squared Score'); plt.ylabel('Combination')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_best_r2_per_combination.png')); plt.close()

# --- 2. MAIN ORCHESTRATION SCRIPT ---

def run_full_analysis(start_date):
    """Orchestrates the entire workflow for all combinations."""
    RESULTS_DIR = 'results_all_combinations'
    PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("--- Starting Full Analysis for All Combinations ---")
    
    try:
        df_main = pd.read_csv('all_commodities_data.csv')
        df_main['Arrival_Date'] = pd.to_datetime(df_main['Arrival_Date'], format='%d/%m/%Y')
    except FileNotFoundError:
        print("Error: 'all_commodities_data.csv' not found. Exiting."); return

    combinations = df_main[['State', 'District', 'Market', 'Commodity']].drop_duplicates()
    print(f"Found {len(combinations)} unique State-District-Market-Commodity combinations.\n")
    
    all_metrics_records, all_forecasts_dict = [], {}

    for index, row in combinations.iterrows():
        state, district, market, crop = row['State'], row['District'], row['Market'], row['Commodity']
        combo_name = f"{state}_{district}_{market}_{crop}".replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        print(f"--- Processing: {combo_name} ---")
        
        ts_df, error = load_and_prepare_daily_data(df_main, state, district, market, crop)
        if error:
            print(f"  > Skipping: {error}\n"); continue

        features_df = create_features(ts_df)
        if len(features_df) < 20:
             print(f"  > Skipping: Not enough data after feature creation.\n"); continue
        X, y = features_df.drop('price', axis=1), features_df['price']

        comparison_df, y_test, predictions = evaluate_models(X, y)
        comparison_df['Combination'] = combo_name
        all_metrics_records.extend(comparison_df.to_dict('records'))
        print("  > Models evaluated.")

        plot_and_save_performance(y_test, predictions, PLOTS_DIR, combo_name)
        print("  > Performance plot saved.")

        best_model_name = comparison_df.sort_values(by="R-squared", ascending=False).iloc[0]['Model']
        if best_model_name == "RandomForest": best_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif best_model_name == "XGBoost": best_model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
        else: best_model = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
        daily_forecast = forecast_future_daily(best_model, features_df, start_date)
        all_forecasts_dict[f"{combo_name}_price"] = daily_forecast
        print(f"  > Forecast generated with {best_model_name}.\n")

    print("--- Aggregating Final Results ---")
    
    # Create and save the pivoted, organized summary
    full_summary_df = pd.DataFrame(all_metrics_records)
    if not full_summary_df.empty:
        pivoted_summary = full_summary_df.pivot_table(
            index='Combination', 
            columns='Model', 
            values=['R-squared', 'MAE', 'RMSE', 'MAPE (%)']
        )
        pivoted_summary.columns = [f'{val}_{model}' for val, model in pivoted_summary.columns]
        pivoted_summary.reset_index(inplace=True)
        
        summary_path = os.path.join(RESULTS_DIR, 'model_comparison_pivoted_summary.csv')
        pivoted_summary.to_csv(summary_path, index=False)
        print(f"> Pivoted model comparison summary saved to: {summary_path}")

        # Create overall summary plots
        create_summary_plots(full_summary_df, RESULTS_DIR)
        print(f"> High-level summary plots saved in: {RESULTS_DIR}")
    
    # Save the big combined forecast CSV
    if all_forecasts_dict:
        final_forecast_df = pd.DataFrame(all_forecasts_dict)
        final_forecast_df.index.name = 'Date'
        forecast_path = os.path.join(RESULTS_DIR, 'all_combinations_daily_forecast_9_months.csv')
        final_forecast_df.to_csv(forecast_path)
        print(f"> Combined daily forecast for all combinations saved to: {forecast_path}")
    
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    # --- USER INPUT: Set the start date for all forecasts ---
    START_DATE = pd.to_datetime('2025-07-06') # Example: Start all forecasts from June 1st, 2024
    # --------------------------------------------------------
    
    run_full_analysis(start_date=START_DATE)