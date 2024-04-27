import pandas as pd
import numpy as np
import os

# Original File and Modified File
input_csv_path = 'docs/Peru_2019_AudioMoth_Data_Full.csv'
output_csv_path = 'docs/Peru_2019_Stratified_Sample.csv'

# Create 'docs' directory if it doesn't exist
output_directory = os.path.dirname(output_csv_path)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def stratified_sample(input_csv_path, output_csv_path):
    try:
        # Load the CSV file into DataFrame with dtype specified to avoid mixed types warning
        df_original = pd.read_csv(input_csv_path, dtype={'AudioMothID': str})
        df = df_original.copy()  # Make copy so we leave original csv file untouched

        # Convert 'StartDateTime' to datetime to extract hour
        # This is part of the second strata layer creation
        df['StartDateTime'] = pd.to_datetime(df['StartDateTime'], format='%d.%m.%Y %H:%M')
        df['hour'] = df['StartDateTime'].dt.hour
        
        # Convert file size from bytes to megabytes for accurate comparison
        df['FileSizeMB'] = df['FileSize'] / (1024 * 1024)
        
        # Filter for recordings that are 60 seconds long and approximately 46.1 MB in size
        # Filter for megabyte allowing +- 0.1mb tolerance
        df = df[(df['Duration'] == 60) & (np.isclose(df['FileSizeMB'], 46.1, atol=0.1))]
        
        # Exclude problematic Audiomoth devices. This filters out devices that had problems, as per the first strata layer.
        problematic_devices = ['21', '19', '8', '28']  # Adjust w/ csv formatting
        df = df[~df['AudioMothID'].isin(problematic_devices)]
        
        # Initialize an empty DataFrame to store the sampled data
        sampled_df = pd.DataFrame()
        
        # Loop through each unique Audiomoth device ID, as per the first strata layer.
        for device in df['AudioMothID'].unique():
            # For each device, get the subset of the data.
            device_data = df[df['AudioMothID'] == device]
            
            # Check if the device has enough clips (at least 24, one for each hour of the day)
            # This part makes sure each device in the first strata layer has enough clips.
            if device_data['hour'].nunique() == 24:
                # For each hour, get a random sample (one clip per hour)
                for hour in range(24):
                    hour_data = device_data[device_data['hour'] == hour]
                    # If there are clips for this hour, sample one
                    if not hour_data.empty:
                        sample = hour_data.sample(n=1)  # Random selection
                        sampled_df = pd.concat([sampled_df, sample], ignore_index=True)
        
        # Save the sampled data to a new CSV file. This is the output of the new stratified CSV file.
        sampled_df.to_csv(output_csv_path, index=False)
        return True  # Return True if the process was successful.
    except Exception as e:
        print(f"An error occurred: {e}")
        return False  # Return False if there was an error in the process.

# Execute the function with the defined paths and print the result.
result = stratified_sample(input_csv_path, output_csv_path)
print("Sampling successful:", result)