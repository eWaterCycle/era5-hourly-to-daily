import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from esmvalcore.cmor.table import CMOR_TABLES

def cmorize_data(file_to_be_cmorized):
    cmip6_table = CMOR_TABLES['CMIP6']
    new = xr.open_dataset(file_to_be_cmorized)

    # Filter variables that exist in dataset
    var_names = list(new.data_vars)
    rename_dict = {
        "t2m": "tas",            # main variables
        "tp": "pr",
        "pev": "evspsblpot",
        "ssrd": "rsds",
        "e": "evspsbl"
    }
    
    new_variables = {           # always include these
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon"
    }

    home_dir = Path.home()

    correct_file_no_leap_dict = {
        "tas": home_dir / "correct_data/OBS6_ERA5_reanaly_1_day_tas_2015-2015.nc",            # main variables
        "pr": home_dir / "correct_data/OBS6_ERA5_reanaly_1_day_pr_2015-2015.nc",
        "evspsblpot": home_dir / "correct_data/OBS6_ERA5_reanaly_1_Eday_evspsblpot_2015-2015.nc",
        "rsds": home_dir / "correct_data/OBS6_ERA5_reanaly_1_day_rsds_2015-2015.nc",
        "evspsbl": home_dir / "correct_data/OBS6_ERA5_reanaly_1_Eday_evspsbl_1994-1994.nc"
    }
    correct_file_leap_dict = {
        "tas": home_dir / "correct_data/OBS6_ERA5_reanaly_1_day_tas_2016-2016.nc",            # main variables
        "pr": home_dir / "correct_data/OBS6_ERA5_reanaly_1_day_pr_2016-2016.nc",
        "evspsblpot": home_dir / "correct_data/OBS6_ERA5_reanaly_1_Eday_evspsblpot_2016-2016.nc",
        "rsds": home_dir / "correct_data/OBS6_ERA5_reanaly_1_day_rsds_2016-2016.nc",
        "evspsbl": home_dir / "correct_data/OBS6_ERA5_reanaly_1_Eday_evspsbl_1996-1996.nc"
    }
    
    conversion_dict = {
        'rsds': 1/3600,
        'pr': 1/3.6,
        'tas': 1,
        'evspsblpot': 1/3.6,
        'evspsbl': 1/3.6,
    }
    unit_dict = {
        'rsds': cmip6_table.get_variable('day', 'rsds'),
        'pr': cmip6_table.get_variable('day', 'pr'),
        'tas': cmip6_table.get_variable('day', 'tas'),
        'evspsblpot': cmip6_table.get_variable('Eday', 'evspsblpot'),
        'evspsbl': cmip6_table.get_variable('Eday', 'evspsbl'),
    }

    '''We first start with renaming the variables from ERA5 to something readable for ESMValTool'''
    # Get the correct key: value that is in your dataset and have it sit in the first index
    filtered_dict = {k: v for k, v in rename_dict.items() if k in var_names}

    # Add the time and lat/lon
    filtered_dict.update(new_variables)  # ensure coords are always renamed

    # Convert valid_time, which is hourly to daily time index
    daily = new.resample(valid_time="1D").mean()
    
    # Apply renaming
    daily = daily.rename(filtered_dict)

    # Get the correct variable name (first!! from filtered_dict that’s a data_var)
    variable_candidates = [v for k, v in filtered_dict.items() if k in var_names]
    if not variable_candidates:
        raise ValueError(f"No matching main variable found in {file_to_be_cmorized}")
    variable = variable_candidates[0]

    # Converting the data so that it is correct for us
    daily[variable] = daily[variable] * conversion_dict[variable]
    print(f"[INFO] Converted {variable} using factor {1/conversion_dict[variable]:.1f}")

    # Build time bounds
    time = daily.time.values
    start = pd.to_datetime(time)
    if variable == 'tas':
        mid_time = start + pd.Timedelta(hours=11, minutes=30)
        end_bounds = start + pd.Timedelta(hours=23)
    else:
        mid_time = start + pd.Timedelta(hours=12)
        end_bounds = start + pd.Timedelta(hours=24)
    
    # time_bnds = np.column_stack([start, end_bounds]).astype("datetime64[ns]")
    
    if start.hour.all() == 0:
        print(f"[INFO] Resetting time bounds for {file_to_be_cmorized}")
        daily = daily.assign_coords(time=("time", mid_time))  # keep time as coordinate
        # daily["time_bnds"] = (("time", "bnds"), time_bnds)  # assign as variable
    else:
        print(f"OOPS there might be a time bounds error in {file_to_be_cmorized}")
    
    # Drop "number" coordinate if it exists
    if "number" in daily.coords:
        daily = daily.drop_vars("number")

    # Promote 'height' to coordinate if present
    if 'height' in daily.data_vars:
        daily = daily.set_coords('height')
    
    # Sort my lat for monotonic ascending
    # --- Latitude ---
    if "lat" in daily.coords:
        lat = daily["lat"].values
        if np.any(np.diff(lat) < 0):
            print("[FIX] Reversing latitude to be ascending (-90 → +90).")
            daily = daily.sortby("lat")  # reverses data automatically

    # Sort by time just to be sure
    daily = daily.sortby("time")
    
    # Extract year for filename and the leap year
    year = start.year[0]

    # Check if it is a leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        correct_file_dict = correct_file_leap_dict
        correct_year = 2016
        print(f"{year = }: leap year!")
    else:
        print(f"{year = }: no leap")
        correct_file_dict = correct_file_no_leap_dict
        correct_year = 2015

    '''Now we get a file of which we know is correct, to compare'''
    correct_file = correct_file_dict[variable]
    historically_correct_ds = xr.open_dataset(correct_file)
    
    # Variable attributes
    daily[variable].attrs = historically_correct_ds[variable].attrs

    # Time encoding
    daily["time"].encoding = historically_correct_ds["time"].encoding
    
    # Global attributes
    daily.attrs = historically_correct_ds.attrs

    daily[variable].attrs['units'] = str(unit_dict[variable].units)
    print(f"[INFO] Unit {str(unit_dict[variable].units)} added to {variable}")

    # Copy bounds if available
    for var in ["time_bnds", "lat_bnds", "lon_bnds"]:
        if var in historically_correct_ds:
            daily[var] = historically_correct_ds[var]

    correct_bounds = historically_correct_ds.data_vars["time_bnds"].values


    year_change = year-correct_year
    
    # Extract components
    years = correct_bounds.astype('datetime64[Y]').astype(int) + 1970
    months = correct_bounds.astype('datetime64[M]').astype(int) % 12 + 1
    days = (correct_bounds - correct_bounds.astype('datetime64[M]')).astype('timedelta64[D]').astype(int) + 1
    hours = (correct_bounds - correct_bounds.astype('datetime64[D]')).astype('timedelta64[h]').astype(int)
    minutes = (correct_bounds - correct_bounds.astype('datetime64[h]')).astype('timedelta64[m]').astype(int) % 60
    seconds = (correct_bounds - correct_bounds.astype('datetime64[m]')).astype('timedelta64[s]').astype(int) % 60
    nanoseconds = (correct_bounds - correct_bounds.astype('datetime64[s]')).astype('timedelta64[ns]').astype(int)
    years = years + year_change

    # Rebuild datetime64[ns] with new year
    new_bounds = np.array([
        np.datetime64(f'{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}.{ns:09d}')
        for y, m, d, h, mi, s, ns in zip(
           years.flatten(), months.flatten(), days.flatten(), hours.flatten(), minutes.flatten(), seconds.flatten(), nanoseconds.flatten()
        )
    ]).reshape(correct_bounds.shape)

    daily.data_vars["time_bnds"].values = new_bounds

    # 🧹 Cleanup temporary variables
    print("Cleaning temp variables")
    del years, months, days, hours, minutes, seconds, nanoseconds, new_bounds, filtered_dict, historically_correct_ds
    print("Done cleaning temp variables")
    
    if variable in ("evspsblpot", "evspsbl"):
        file_name = f"OBS6_ERA5_reanaly_1_Eday_{variable}_{year}-{year}.nc"
    else:
        file_name = f"OBS6_ERA5_reanaly_1_day_{variable}_{year}-{year}.nc"
        
    return daily, file_name

if __name__ == "__main__":
    home = Path.home() 
    input_folder = home / "data_to_be_cmorized"                  # folder with raw ERA5 files
    output_folder = home / "cmorized_output"           # where to save results
    
    os.makedirs(output_folder, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_folder, "*.nc")))
    
    for f in files:
        try:
            daily, file_name = cmorize_data(f)
            out_path = os.path.join(output_folder, file_name)
            daily.to_netcdf(out_path, format="NETCDF4")
            print(f"[OK] Saved: {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed on {f}: {e}")