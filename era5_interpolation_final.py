#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

import pyproj
import cmocean
import cartopy.crs as ccrs
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from shapely.geometry import Polygon, Point
import statsmodels.api as sm
from datetime import timedelta


# ### - Maximum Distance of 1 Degree

# In[2]:


loc_file_path = "/Users/skylarpark/ros/NSIDC_Data/metadata/aross.asos_stations.metadata.csv"

loc_data = pd.read_csv(loc_file_path)


# In[3]:


directory = "/Users/skylarpark/ros/NSIDC_Data/events/"
os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

station_data = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Use the first four letters of the filename as the key
        key = filename[:4]
        # Store the DataFrame in the dictionary
        station_data[key] = df

# print(station_data)


# In[4]:


directory2 = "/Users/skylarpark/ros/michelle/DATA/"
os.makedirs(directory2, exist_ok=True)  # Create the directory if it doesn't exist

era5_data = {}

# Loop through all files in the directory
for filename in os.listdir(directory2):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(directory2, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Use the first four letters of the filename as the key
        key = filename[47:51]
        # Store the DataFrame in the dictionary
        era5_data[key] = df

# print(era5_data)


# In[15]:


era5_data


# ### Filter for stations in Eurasia

# In[9]:


# Filter loc_data to include only stations in Eurasia
# eurasia_filtered = loc_data[(loc_data['longitude'] >= -30) & (loc_data['longitude'] <= 180) & 
#                             (loc_data['latitude'] >= 0) & (loc_data['latitude'] <= 90)]

eurasia_filtered = loc_data[loc_data['stid'].str.startswith(('E', 'U')) | loc_data['stid'].isin(['BIAR', 'BIEG', 'BIKF', 'BIRK'])]

print(eurasia_filtered)

# Extract the list of station IDs from eurasia_filtered
eurasia_station_ids = eurasia_filtered['stid'].tolist()

# Filter station_data to include only those stations whose IDs are in eurasia_station_ids
station_data_filtered = {station_id: data for station_id, data in station_data.items() if station_id in eurasia_station_ids}

# Display the filtered data
# print(station_data_filtered)


# ### Add 'date', 'latitude' and 'longitude' to station_data_filtered

# In[10]:


# Add 'date' column to each DataFrame in station_data_filtered
for station_id, df in station_data_filtered.items():
    df['date'] = pd.to_datetime(df['start']).dt.date

# Convert eurasia_filtered to a dictionary mapping stid to latitude and longitude
location_dict = eurasia_filtered.set_index('stid')[['latitude', 'longitude']].to_dict('index')

# station_data_filtered is your dictionary of DataFrames for each station
# Example station_data_filtered = {'EFTU': pd.DataFrame(...), 'EGWZ': pd.DataFrame(...)}

# Add latitude and longitude to each DataFrame
for stid, df in station_data_filtered.items():
    # Extract latitude and longitude from the dictionary using the station ID
    if stid in location_dict:
        df['latitude'] = location_dict[stid]['latitude']
        df['longitude'] = location_dict[stid]['longitude']
    else:
        # Handle cases where no location info is available
        df['latitude'] = None
        df['longitude'] = None

station_data_filtered['ENBV']


# ### Add 'date' to era5_data

# In[ ]:


# Construct 'date' column in era5_data if 'Year', 'Month', and 'Day' columns are present
for year, df in era5_data.items():
    if all(col in df.columns for col in ['Year', 'Month', 'Day']):
        df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    else:
        print(f"Skipping year {year} due to missing columns")

era5_data


# ### Interpolation

# ##### Cleaning Lat and Lon in era5_data (for ranges)

# In[ ]:


import ast

# Function to convert range strings to float midpoints
# def convert_range_to_float(value):
#     value = str(value)  # Convert to string first
#     if ':' in value:
#         start, end = map(float, value.split(':'))
#         return (start + end) / 2
#     return float(value)

def convert_string_to_mean(value):
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        # Convert the string to a list using ast.literal_eval
        value = ast.literal_eval(value)
        if isinstance(value, list):
            return np.mean(value)  # Calculate the mean of the list
    return float(value)  # Convert to float if it's not a list

# Clean latitude and longitude columns
# for year, df in era5_data.items():
    # if 'Lat' in df.columns and 'Lon' in df.columns:
    # if 'Latitudes' in df.columns and 'Longitudes' in df.columns:
#         df['Latitudes'] = df['Latitudes'].apply(convert_range_to_float)
#         df['Longitudes'] = df['Longitudes'].apply(convert_range_to_float)
#     else:
#         print(f"Skipping year {year} due to missing 'Lat' or 'Lon' columns")

# Clean latitude and longitude columns
for year, df in era5_data.items():
    if 'Latitudes' in df.columns and 'Longitudes' in df.columns:
        df['Latitudes'] = df['Latitudes'].apply(convert_string_to_mean)
        df['Longitudes'] = df['Longitudes'].apply(convert_string_to_mean)
    else:
        print(f"Skipping year {year} due to missing 'Lat' or 'Lon' columns")


# In[ ]:


directory3 = "/Users/skylarpark/ros/michelle/DATA/edit"

os.makedirs(directory3, exist_ok=True)

# Loop through each year's DataFrame and save it as a CSV
for year, df in era5_data.items():
    output_path = os.path.join(directory3, f"era5_data_{year}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {year} data to {output_path}")


# In[7]:


directory3 = "/Users/skylarpark/ros/michelle/DATA/edit"

# Load the CSV files back into the era5_data dictionary
loaded_era5_data = {}
for year in os.listdir(directory3):
    if year.endswith(".csv"):
        year_key = year.replace("era5_data_", "").replace(".csv", "")
        loaded_era5_data[year_key] = pd.read_csv(os.path.join(directory3, year))

era5_data = {}
era5_data = loaded_era5_data


# In[8]:


era5_data


# #### Interpolation with a distance limit

# In[ ]:


import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Function to perform IDW interpolation with dynamic neighbors and a distance limit
def idw_interpolation(x, y, z, xi, yi, power=2, distance_limit=1.0):
    """Perform Inverse Distance Weighting interpolation with a distance limit."""
    tree = cKDTree(np.c_[x, y])
    k = min(4, len(x))  # Use up to 4 neighbors or the number of available points
    distances, indices = tree.query(np.c_[xi, yi], k=k)
    
    # Check the distance limit
    if np.any(distances > distance_limit):
        # If any of the closest points are further than the limit, return NaN
        return np.full(xi.shape, np.nan)
    
    weights = 1 / distances**power
    weights /= weights.sum(axis=1)[:, None]
    zi = np.sum(weights * z[indices], axis=1)
    return zi


# In[17]:


# Function to perform IDW interpolation with dynamic neighbors and a distance limit
def idw_interpolation(x, y, z, xi, yi, power=2, distance_limit=1.0):
    tree = cKDTree(np.c_[x, y])
    k = min(4, len(x))  # Use up to 4 neighbors or the number of available points
    distances, indices = tree.query(np.c_[xi, yi], k=k)
    
    # Apply distance limit
    mask = distances <= distance_limit
    if not np.any(mask):
        return np.full(xi.shape, np.nan)
    
    # Only use points within the distance limit
    distances = distances[:, mask]
    indices = indices[:, mask]
    
    weights = 1 / distances**power
    weights /= weights.sum(axis=1)[:, None]
    zi = np.sum(weights * z[indices], axis=1)
    return zi


# In[13]:


# Interpolate ERA5 rainfall to station locations
for station_id, df in station_data_filtered.items():
    print(f"Processing station: {station_id}")
    df['interpolated_rainfall'] = np.nan  # Initialize the column with NaN values
    for date in df['date'].unique():
        # print(f"  Processing date: {date}")
        era5_year = str(pd.to_datetime(date).year)
        if era5_year in era5_data:
            era5_subset = era5_data[era5_year]
            if 'date' in era5_subset.columns:
                era5_on_date = era5_subset[era5_subset['date'] == pd.to_datetime(date)]
                if not era5_on_date.empty:
                    # Coordinates of ERA5 data points
                    era5_x = era5_on_date['Longitudes'].values
                    era5_y = era5_on_date['Latitudes'].values
                    era5_z = era5_on_date['Rain'].values

                    # Coordinates of the station
                    station_x = df[df['date'] == date]['longitude'].values
                    station_y = df[df['date'] == date]['latitude'].values

                    # Interpolated rainfall
                    interpolated_rainfall = idw_interpolation(era5_x, era5_y, era5_z, station_x, station_y)

                    # Add interpolated rainfall to the DataFrame
                    df.loc[df['date'] == date, 'interpolated_rainfall'] = interpolated_rainfall
                else:
                    print(f"    No data for date {date} in ERA5 subset for year {era5_year}")
            else:
                print(f"    'date' column missing in ERA5 data for year {era5_year}")
        else:
            print(f"    No ERA5 data for year {era5_year}")

# Display the filtered data with interpolated rainfall
print("Filtered station_data with interpolated rainfall:")
for station_id, df in station_data_filtered.items():
    print(f"Station: {station_id}")
    print(df)


# In[18]:


# Interpolate ERA5 rainfall to station locations
for station_id, df in station_data_filtered.items():
    print(f"Processing station: {station_id}")
    df['interpolated_rainfall'] = np.nan  # Initialize the column with NaN values
    for date in df['date'].unique():
        era5_year = str(pd.to_datetime(date).year)
        if era5_year in era5_data:
            era5_subset = era5_data[era5_year]
            if 'date' in era5_subset.columns:
                era5_on_date = era5_subset[era5_subset['date'] == pd.to_datetime(date)]
                if not era5_on_date.empty:
                    # Coordinates of ERA5 data points
                    era5_x = era5_on_date['Longitudes'].values
                    era5_y = era5_on_date['Latitudes'].values
                    era5_z = era5_on_date['Rain'].values

                    # Coordinates of the station
                    station_x = df[df['date'] == date]['longitude'].values
                    station_y = df[df['date'] == date]['latitude'].values

                    # Interpolated rainfall
                    interpolated_rainfall = idw_interpolation(era5_x, era5_y, era5_z, station_x, station_y, distance_limit=1.0)

                    # Add interpolated rainfall to the DataFrame
                    df.loc[df['date'] == date, 'interpolated_rainfall'] = interpolated_rainfall
                else:
                    print(f"    No data for date {date} in ERA5 subset for year {era5_year}")
            else:
                print(f"    'date' column missing in ERA5 data for year {era5_year}")
        else:
            print(f"    No ERA5 data for year {era5_year}")


# In[19]:


# Display the filtered data with interpolated rainfall
print("Filtered station_data with interpolated rainfall:")
for station_id, df in station_data_filtered.items():
    print(f"Station: {station_id}")
    print(df[['date', 'RA', 'interpolated_rainfall']])  # Display relevant columns including the interpolated rainfall


# #### Save the data

# In[20]:


directory3 = '/Users/skylarpark/ros/intern/ERA5_INTERP_v3'
# Check if the directory exists, if not, create it
if not os.path.exists(directory3):
    os.makedirs(directory3)

# Assuming station_data_filtered is your dictionary of DataFrames
for stid, df in station_data_filtered.items():
    # Include the directory in the filename
    filename = os.path.join(directory3, f"{stid}_era5_interp_v3.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {stid} data to {filename}")


# In[21]:


directory3 = '/Users/skylarpark/ros/intern/ERA5_INTERP_v3'  # The directory where the files are stored
data_files = os.listdir(directory3)     # List all files in the directory

station_data_interp = {}  # Dictionary to hold the data

# Loop through each file in the directory
for file in data_files:
    if file.endswith('_era5_interp_v3.csv'):  # Check for your specific file pattern
        stid = file.split('_')[0]           # Extract station ID from the filename
        file_path = os.path.join(directory3, file)
        station_data_interp[stid] = pd.read_csv(file_path)  # Load the file and add to dict

print(station_data_interp)  # Now you have your dictionary of DataFrames


# In[ ]:




