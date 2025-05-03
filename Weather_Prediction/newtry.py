import cdsapi
import xarray as xr # For reading the downloaded NetCDF file
import datetime

# --- Configuration ---
# IMPORTANT: Browse the CDS website for the correct dataset name and parameters!
# This is just an EXAMPLE based on potential datasets like SEAS5 or its successor.
dataset_name = 'seasonal-monthly-single-levels' # Example dataset name - VERIFY ON CDS WEBSITE
output_file = 'seasonal_forecast.nc' # Name for the downloaded file (usually NetCDF)

# Get current year and month
now = datetime.datetime.now()
current_year = str(now.year)
# Usually, seasonal forecasts are issued for months ahead based on the *previous* month's run
# Let's request forecasts for the next 3 months starting from the next month
start_month = str(now.month + 1 if now.month < 12 else 1)
start_year = current_year if now.month < 12 else str(now.year + 1)

# Define target months (e.g., next 3 months) - CDS expects month numbers
target_months = []
month = int(start_month)
year = int(start_year)
for _ in range(3):
    target_months.append(str(month))
    month += 1
    if month > 12:
        month = 1
        # Note: Most seasonal datasets might not cross calendar years easily in one request
        # You might need separate requests if your range crosses Dec/Jan.
        # This example simplifies and assumes it doesn't for clarity.

print(f"Requesting forecast data for months: {target_months} of year {year}")

# Parameters for the CDS request - **THESE MUST BE CHECKED/ADAPTED** based on the specific dataset page on CDS
request_params = {
    'originating_centre': 'ecmwf', # Or other centers if available
    'system': '51', # Example: SEAS51 system number - CHECK for current system on CDS
    'variable': '2m_temperature_anomaly', # Example: Requesting temperature anomaly
    # Other potential variables: 'total_precipitation_anomaly', etc.
    'product_type': 'monthly_mean', # Or 'hindcast_climate_mean', 'forecast_anomaly_probability' etc.
    'year': str(year),
    'month': target_months,
    'leadtime_month': [ # Lead time often starts from 1 (forecast for the next month)
        '1', '2', '3', # Requesting forecasts 1, 2, and 3 months ahead
        # Check dataset documentation for how lead time is defined!
    ],
    'format': 'netcdf', # Common format, easier to handle with xarray
    # You might need to specify specific geographical area (lat/lon bounds)
    # 'area': [north, west, south, east], # Example: [60, -10, 30, 30] for Europe
    # Check dataset details for required parameters!
}

print("Submitting request to CDS API...")
print("Request Parameters:", request_params) # Good practice to print what you're asking for

try:
    c = cdsapi.Client() # Reads credentials from ~/.cdsapirc

    # Submit the request
    c.retrieve(
        dataset_name,
        request_params,
        output_file
    )

    print(f"Data downloaded successfully to: {output_file}")

    # --- Basic Data Inspection using xarray ---
    print("\nAttempting to open and inspect the downloaded NetCDF file...")
    try:
        # Lazy loading by default, doesn't load all data into memory
        ds = xr.open_dataset(output_file)
        print("\n--- Dataset Structure ---")
        print(ds)

        # Example: Accessing the temperature anomaly variable (name might differ)
        # Variable names can be complex, inspect `print(ds)` output carefully
        if 't2a' in ds.variables: # 't2a' is a common short name for 2m temp anomaly
            temp_anomaly = ds['t2a']
            print("\n--- Temperature Anomaly Variable ---")
            print(temp_anomaly)

            # Further processing would go here (e.g., selecting specific location, time)
            # Example: Select data for a specific latitude/longitude point (if available)
            # lat_point = 20.59
            # lon_point = 78.96
            # point_data = temp_anomaly.sel(latitude=lat_point, longitude=lon_point, method='nearest')
            # print(f"\n--- Data for nearest point to Lat={lat_point}, Lon={lon_point} ---")
            # print(point_data)

        else:
             print("\nCould not find expected temperature anomaly variable ('t2a'). Check dataset structure above.")

        ds.close() # Close the file handle

    except Exception as e:
        print(f"\nError reading or processing NetCDF file with xarray: {e}")
        print("You might need to install 'netcdf4' and 'cfgrib' libraries: pip install xarray netcdf4 cfgrib")

except Exception as e:
    print(f"An error occurred during the CDS API request or processing: {e}")
    print("Possible issues:")
    print("- Check your internet connection.")
    print("- Verify your UID/API key in the .cdsapirc file.")
    print("- Ensure the dataset name and request parameters are correct for the CDS dataset.")
    print("- Check the CDS website for any service notifications or quota limits.")
    print("- The specific request might be too large or invalid.")