import requests
import numpy as np
import pandas as pd # Added for data handling
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs # For map plotting
import cartopy.feature as cfeature # For map features
from PIL import Image # Used implicitly by animation saving sometimes
import imageio # For saving animation as GIF
import time
import argparse # Added for command-line control
import os

# --- Original API Function (with bug warning) ---
def get_enhanced_forecast_on_end_date(
    lat,
    lon,
    start_date,
    end_date,
    timezone="auto" # Original default
):
    variables_base = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "wind_speed_10m_max", "shortwave_radiation_sum"
    ]
    url = "https://seasonal-api.open-meteo.com/v1/seasonal"
    params = {
        "latitude": lat, "longitude": lon, "daily": ",".join(variables_base),
        "start_date": start_date, "end_date": end_date, "timezone": timezone
    }

    print(f"\n--- Attempting REAL API Call (Expected to Fail Due to Bug) ---")
    print(f"URL: {url}")
    print(f"Params: {params}")
    try:
        resp = requests.get(url, params=params, timeout=20) # Added timeout
        print(f"API Response Status Code: {resp.status_code}")
        if resp.status_code != 200:
            print(f"API Error Body (partial): {resp.text[:500]}...")
            raise Exception(f"API Error {resp.status_code} (Likely the known 'daily' bug). Response: {resp.text}")

        data = resp.json()
        daily = data.get("daily")
        if not daily or "time" not in daily:
            raise KeyError("No 'daily' time series returned from API.")

        times = daily["time"]
        if end_date not in times:
            raise ValueError(f"End date {end_date} not in returned API time array.")
        idx = times.index(end_date)

        result = {"date": end_date, "latitude": lat, "longitude": lon} # Added coords
        for var in variables_base:
            member_keys = [k for k in daily.keys() if k.startswith(f"{var}_member")]
            if not member_keys:
                result[var] = None
                continue
            vals = [daily[k][idx] for k in member_keys if k in daily and daily[k] and idx < len(daily[k])]
            clean = [v for v in vals if v is not None]
            result[var] = float(np.mean(clean)) if clean else None
        print("--- Real API Call Succeeded (Unexpected!) ---")
        return result

    except requests.exceptions.RequestException as req_err:
         print(f"--- Real API Call Failed (Network/Timeout Error) ---")
         print(f"Error: {req_err}")
         return None # Indicate failure
    except Exception as e:
        print(f"--- Real API Call Failed (As Expected or Other Error) ---")
        print(f"Error: {e}")
        return None # Indicate failure

# --- Simulation Functions ---

def simulate_forecast_grid(
    lat_range=(8, 37), lon_range=(68, 98), grid_res=2.0, date="2025-10-15"
):
    """Generates simulated forecast data for a grid over India."""
    print(f"\n--- Generating SIMULATED Forecast Grid Data for {date} ---")
    lats = np.arange(lat_range[0], lat_range[1] + grid_res, grid_res)
    lons = np.arange(lon_range[0], lon_range[1] + grid_res, grid_res)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    data = []
    for lat, lon in zip(lat_grid.ravel(), lon_grid.ravel()):
        # Simulate slightly plausible values (vary with latitude, add noise)
        base_temp = 35 - (lat - lat_range[0]) * 0.8 # Cooler further north
        temp_max = base_temp + np.random.uniform(-2, 2) + np.sin(lon/10) * 1.5
        temp_min = temp_max - np.random.uniform(5, 12)
        precip = max(0, np.random.normal(loc=2, scale=5) + np.cos(lat/5)*3) # Patchy rain
        wind = max(0, np.random.normal(loc=10, scale=4))
        radiation = max(10, np.random.normal(loc=180, scale=40) - (lat - lat_range[0]) * 2)

        data.append({
            "date": date,
            "latitude": lat,
            "longitude": lon,
            "temperature_2m_max": round(temp_max, 1),
            "temperature_2m_min": round(temp_min, 1),
            "precipitation_sum": round(precip, 1),
            "wind_speed_10m_max": round(wind, 1),
            "shortwave_radiation_sum": round(radiation, 1)
        })
    print(f"Generated {len(data)} simulated grid points.")
    return pd.DataFrame(data)


def simulate_forecast_timeseries(lat, lon, start_date_str, num_days=30):
    """Generates simulated daily forecast time series for a single location."""
    print(f"\n--- Generating SIMULATED Forecast Time Series for ({lat},{lon}) ---")
    start_date = pd.to_datetime(start_date_str)
    dates = pd.date_range(start_date, periods=num_days, freq='D')

    data = []
    # Base values (can be adjusted based on lat/lon or season if desired)
    base_temp = 30
    temp_amplitude = 8
    precip_chance = 0.15
    precip_amount_mean = 5

    for i, date in enumerate(dates):
        # Simulate daily cycle and gradual change/noise
        day_of_year_effect = np.sin((date.dayofyear / 365.0) * 2 * np.pi - np.pi/2) # Simple seasonal trend
        temp_max = base_temp + temp_amplitude * day_of_year_effect + np.random.uniform(-1.5, 1.5) + np.sin(i/5)*1.0
        temp_min = temp_max - np.random.uniform(6, 10) - np.cos(i/3)*0.5
        precip = 0.0
        if np.random.rand() < precip_chance + np.sin(i/7)*0.05: # Varying chance
             precip = max(0, np.random.normal(loc=precip_amount_mean, scale=4))
        wind = max(1, np.random.normal(loc=8, scale=3) + np.sin(i/10)*2)
        radiation = max(20, np.random.normal(loc=200, scale=30) * (1 - precip/30)) # Less radiation if raining

        data.append({
            "date": date.strftime('%Y-%m-%d'),
            "latitude": lat,
            "longitude": lon,
            "temperature_2m_max": round(temp_max, 1),
            "temperature_2m_min": round(temp_min, 1),
            "precipitation_sum": round(precip, 1),
            "wind_speed_10m_max": round(wind, 1),
            "shortwave_radiation_sum": round(radiation, 1)
        })
    print(f"Generated {len(data)} simulated daily records.")
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']) # Convert date column to datetime objects
    df = df.set_index('date') # Set date as index for easy time series plotting
    return df


# --- Plotting Functions ---

def plot_forecast_map(df_grid, variable, cmap='viridis', title_extra="", save_path=None):
    """Plots simulated grid data on a map of India."""
    if df_grid is None or df_grid.empty or variable not in df_grid.columns:
        print(f"Cannot plot map for '{variable}'. Data is missing or invalid.")
        return

    print(f"\n--- Plotting SIMULATED Map for: {variable} {title_extra} ---")
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree()) # Standard projection

    # Set extent roughly around India
    ax.set_extent([67, 99, 7, 38], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')

    # Create the scatter plot
    sc = ax.scatter(df_grid['longitude'], df_grid['latitude'],
                    c=df_grid[variable],
                    cmap=cmap,
                    s=50, # Adjust size of points
                    transform=ccrs.PlateCarree(), # Data coordinates are lat/lon
                    alpha=0.8)

    # Add color bar
    plt.colorbar(sc, ax=ax, orientation='vertical', label=f'{variable} ({df_grid["date"].iloc[0]})', shrink=0.7)

    # Add gridlines and title
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.title(f'Simulated Forecast: {variable.replace("_", " ").title()} {title_extra}')

    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Map saved to: {save_path}")
        except Exception as e:
            print(f"Error saving map plot: {e}")
            plt.show() # Show if saving failed
    else:
        plt.show()
    plt.close() # Close the figure


def animate_forecast_timeseries(df_timeseries, save_path="forecast_animation.gif"):
    """Creates an animation of the simulated time series forecast."""
    if df_timeseries is None or df_timeseries.empty:
        print("Cannot create animation. Time series data is missing.")
        return

    print(f"\n--- Creating SIMULATED Forecast Animation ---")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Simulated Daily Forecast for ({df_timeseries['latitude'].iloc[0]}, {df_timeseries['longitude'].iloc[0]})")

    # --- Plotting elements setup (lines and bars) ---
    line_temp_max, = axes[0].plot([], [], 'r-', label='Max Temp (°C)')
    line_temp_min, = axes[0].plot([], [], 'b-', label='Min Temp (°C)')
    # Note: bar_precip_container is not used directly anymore, but setup is okay
    # bar_precip_container = axes[1].bar([], [], color='lightblue', label='Precipitation (mm)')
    line_wind, = axes[2].plot([], [], 'g-', label='Max Wind Speed (m/s)')

    # --- Axis limits and labels (Set initial properties) ---
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    min_temp = df_timeseries['temperature_2m_min'].min() - 2
    max_temp = df_timeseries['temperature_2m_max'].max() + 2
    axes[0].set_ylim(min_temp, max_temp)

    # Setup static properties for axes[1] here (they will be reset in update)
    axes[1].set_ylabel("Precipitation (mm)")
    axes[1].grid(True)
    axes[1].set_ylim(0, df_timeseries['precipitation_sum'].max() * 1.1 + 1) # Use full range
    axes[1].legend(loc='upper right') # Setup legend initially

    axes[2].set_ylabel("Wind Speed (m/s)")
    axes[2].legend(loc='upper right')
    axes[2].grid(True)
    axes[2].set_ylim(0, df_timeseries['wind_speed_10m_max'].max() * 1.1)
    axes[2].set_xlabel("Date")

    # Set shared x-axis limits based on the full dataset
    axes[2].set_xlim(df_timeseries.index.min(), df_timeseries.index.max())
    fig.autofmt_xdate() # Improve date formatting

    # --- Animation update function ---
    def update(frame):
        current_date = df_timeseries.index[frame]
        data_so_far = df_timeseries.iloc[:frame+1]

        # --- Update Temperature and Wind Lines ---
        line_temp_max.set_data(data_so_far.index, data_so_far['temperature_2m_max'])
        line_temp_min.set_data(data_so_far.index, data_so_far['temperature_2m_min'])
        line_wind.set_data(data_so_far.index, data_so_far['wind_speed_10m_max'])

        # --- Update Precipitation Bars (Using cla() method) ---
        axes[1].cla() # Clear the entire precipitation axes

        # Re-apply static properties that cla() removes
        axes[1].set_ylabel("Precipitation (mm)")
        axes[1].grid(True)
        # Make sure axis limits cover the whole dataset, not just data_so_far
        axes[1].set_ylim(0, df_timeseries['precipitation_sum'].max() * 1.1 + 1)
        # Ensure x-limits are consistent across frames
        axes[1].set_xlim(df_timeseries.index.min(), df_timeseries.index.max())

        # Re-draw the bars only for the data up to the current frame
        # The label is used for the legend
        current_bars = axes[1].bar(data_so_far.index, data_so_far['precipitation_sum'], color='lightblue', label='Precipitation (mm)')
        # Re-apply legend INSIDE update because cla() removes it
        axes[1].legend(loc='upper right')

        # --- Update title ---
        axes[0].set_title(f"Date: {current_date.strftime('%Y-%m-%d')}")

        # --- Return modified artists ---
        # With blit=False, returning is less critical, but good practice
        # Return the objects that were directly modified or the axes that were cleared/redrawn
        return line_temp_max, line_temp_min, axes[1], line_wind

    # --- Create and save animation ---
    # blit=False is safer and recommended when using cla()
    ani = animation.FuncAnimation(fig, update, frames=len(df_timeseries),
                                  interval=100, blit=False, repeat=False) # interval in ms

    if save_path:
        try:
            print(f"Saving animation to {save_path} (this may take a while)...")
            # Use imageio writer for better GIF control
            writer = imageio.get_writer(save_path, mode='I', duration=0.1) # duration per frame
            for i in range(len(df_timeseries)):
                # Update the plot to the state of frame 'i' before drawing
                update(i)
                fig.canvas.draw() # Draw the canvas

                # --- FIX: Use tostring_argb and reshape for 4 channels ---
                buf = fig.canvas.tostring_argb() # Get buffer in Alpha, Red, Green, Blue format
                image = np.frombuffer(buf, dtype='uint8')
                # Reshape based on canvas dimensions and 4 channels (ARGB)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # imageio can often handle ARGB/RGBA directly for GIF.
                # If conversion is needed, uncomment one of the lines below:
                # image = image[:, :, 1:4] # Simple slice (might get BGR order, test!)
                # pil_image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'ARGB'); image = np.array(pil_image.convert('RGB')) # Robust conversion

                writer.append_data(image) # Add the frame to the GIF
                print(f"  Processed frame {i+1}/{len(df_timeseries)}", end='\r') # Progress indicator
            writer.close()
            print(f"\nAnimation saved successfully to: {save_path}")

        except Exception as e:
            print(f"\nError saving animation: {e}")
            print("Ensure 'imageio[ffmpeg]' or 'ffmpeg'/'imagemagick' is installed and accessible.")
            print("Displaying animation instead (if possible in environment).")
            plt.show() # Show if saving failed
    else:
        print("Displaying animation interactively...")
        plt.show()
    plt.close(fig) # Close the animation figure


# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get weather forecast (real or simulated) and create plots/animations.")
    parser.add_argument("--lat", type=float, default=19.0760, help="Latitude for forecast (default: Mumbai)")
    parser.add_argument("--lon", type=float, default=72.8777, help="Longitude for forecast (default: Mumbai)")
    parser.add_argument("--start_date", default="2025-05-01", help="Start date for API call (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="2025-12-18", help="End date for API call (YYYY-MM-DD)")
    parser.add_argument("--timezone", default="Asia/Kolkata", help="Timezone for API call (e.g., 'auto', 'Asia/Kolkata')")
    parser.add_argument("--sim_days", type=int, default=60, help="Number of days for simulated time series")
    parser.add_argument("--map_var", default="temperature_2m_max", help="Variable to plot on the map")
    parser.add_argument("--map_output", default="forecast_map.png", help="Output filename for the map plot (e.g., map.png)")
    parser.add_argument("--anim_output", default="forecast_animation.gif", help="Output filename for the animation (e.g., anim.gif)")
    parser.add_argument("--skip_real_api", action="store_true", help="Skip the real API call attempt")
    parser.add_argument("--skip_map", action="store_true", help="Skip generating the map plot")
    parser.add_argument("--skip_anim", action="store_true", help="Skip generating the animation")

    args = parser.parse_args()

    # --- 1. Attempt Real API Call (Optional) ---
    real_forecast_data = None
    if not args.skip_real_api:
        real_forecast_data = get_enhanced_forecast_on_end_date(
            args.lat, args.lon, args.start_date, args.end_date, args.timezone
        )
        if real_forecast_data:
            print("\n--- Real Forecast Data Retrieved ---")
            print(real_forecast_data)
            print("----------------------------------")
        else:
            print("\n--- Real API call failed or returned no data (as expected/possible). Proceeding with simulations. ---")
    else:
        print("\n--- Skipping real API call attempt as requested. ---")


    # --- 2. Generate Simulated Data ---
    # Grid data for map
    simulated_grid_df = None
    if not args.skip_map:
        simulated_grid_df = simulate_forecast_grid(date=args.end_date) # Use end_date for consistency demo

    # Time series data for animation (using the specified lat/lon)
    simulated_ts_df = None
    if not args.skip_anim:
        simulated_ts_df = simulate_forecast_timeseries(args.lat, args.lon, args.start_date, args.sim_days)


    # --- 3. Create Plots/Animations from Simulated Data ---
    print("\n--- Generating Plots from SIMULATED Data ---")

    # Map Plot
    if not args.skip_map and simulated_grid_df is not None:
        plot_forecast_map(
            simulated_grid_df,
            variable=args.map_var,
            cmap='coolwarm', # Example colormap
            title_extra=f"on {args.end_date}",
            save_path=args.map_output
        )
    elif not args.skip_map:
         print("Skipping map plot generation as simulated grid data is not available.")


    # Animation
    if not args.skip_anim and simulated_ts_df is not None:
        animate_forecast_timeseries(
            simulated_ts_df,
            save_path=args.anim_output
        )
    elif not args.skip_anim:
        print("Skipping animation generation as simulated time series data is not available.")

    print("\n--- Script Finished ---")