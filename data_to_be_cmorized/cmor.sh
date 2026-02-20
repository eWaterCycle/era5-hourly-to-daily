#!/bin/bash

# Download ERA5 hourly ta, evspsblpot, pr, rsds data for years xxxx–yyyy
# Using 2 threads and merging output

for year in {1940..1949}; do
    echo "Processing year $year..."
    era5cli hourly \
        --variables 2m_temperature potential_evaporation total_precipitation surface_solar_radiation_downwards\
        --startyear $year \
        --threads 2 \
        --merge &
    wait
done

# Wait for all background jobs to finish
wait
echo "All downloads completed."