import pandas as pd
import os

def get_original_names_map(original_data_path):
    """
    Reads the original data and creates a mapping from the 'safe' version of a name
    to the 'original' version with special characters.
    """
    try:
        df_original = pd.read_csv(original_data_path)
    except FileNotFoundError:
        print(f"Error: The original data file was not found at '{original_data_path}'.")
        print("Please make sure 'all_commodities_data.csv' is in the same directory.")
        return None

    # Get all unique combinations from the original data
    combinations = df_original[['State', 'District', 'Market', 'Commodity']].drop_duplicates()
    
    name_map = {}
    for index, row in combinations.iterrows():
        # Create the original, "messy" name
        original_name = f"{row['State']}_{row['District']}_{row['Market']}_{row['Commodity']}"
        
        # Create the 'safe' version, exactly as the main script does
        safe_name = original_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        
        # Store the mapping: {safe_name: original_name}
        name_map[safe_name] = original_name
        
    return name_map

def rename_forecast_columns(forecast_path, name_map):
    """
    Loads the forecast CSV, renames the columns using the provided map,
    and saves a new file with the original names.
    """
    try:
        df_forecast = pd.read_csv(forecast_path, index_col='Date')
    except FileNotFoundError:
        print(f"Error: The forecast file was not found at '{forecast_path}'.")
        print("Please run the main 'price.py' script first to generate it.")
        return

    # Create a dictionary for renaming, adding the '_price' suffix
    # e.g., {'Haryana_Gurgaon_Farukh_Nagar_Onion_price': 'Haryana_Gurgaon_Farukh Nagar_Onion_price'}
    rename_dict = {
        f"{safe_name}_price": f"{original_name}_price"
        for safe_name, original_name in name_map.items()
    }

    # Rename the columns in the forecast DataFrame
    df_forecast.rename(columns=rename_dict, inplace=True)
    
    # Define the new filename
    output_dir = os.path.dirname(forecast_path)
    new_filename = os.path.join(output_dir, 'all_combinations_daily_forecast_ORIGINAL_NAMES.csv')
    
    # Save the newly renamed DataFrame to a new CSV file
    df_forecast.to_csv(new_filename)
    
    print("Success! Columns have been renamed.")
    print(f"New file saved as: {new_filename}")


# --- Main execution block ---
if __name__ == "__main__":
    # Define the paths to your files
    ORIGINAL_DATA_FILE = 'all_commodities_data.csv'
    FORECAST_FILE_PATH = os.path.join('results_all_combinations', 'all_combinations_daily_forecast_9_months.csv')

    print("--- Starting Column Renaming Utility ---")
    
    # Step 1: Get the mapping from safe names to original names
    original_names_map = get_original_names_map(ORIGINAL_DATA_FILE)
    
    if original_names_map:
        # Step 2: Use the map to rename columns in the forecast file
        rename_forecast_columns(FORECAST_FILE_PATH, original_names_map)