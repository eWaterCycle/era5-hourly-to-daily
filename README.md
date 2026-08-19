# era5-hourly-to-daily

A repository storing our code to go from era5 hourly data, taken with era5cli, to daily data that is usable by ESMValTool.

### Important notes

- This code is run from a linux environment, so the `path.home()` variable might differ from yours.
- You do need correct data from a leap and non-leap year
- This code works, but it is slow and there are improvements to be made for sure.


## How to Use

1. use the [era5cli](https://era5cli.readthedocs.io/en/stable/) bash script in `era5-hourly-to-daily/data_to_be_cmorized/cmor.sh` and get the correct years and variables
2. make sure you have data that is already correct, 2 datasets per variable: 1 non-leap year and 1 leap year.
3. go into: `py_cmor.py` then from the top line to the bottom:
   - make sure the `rename_dict` is correct and has the variables you want, in 'era5': 'cmip variable name'
   - check `home_dir`
   - check the `conversion_dict`, if you are not sure what it is for your variable, you can always set it to 1 and then change accordingly when testing your results
   - In this section: `# Build time bounds` we set the time bounds for 'tas' differently this is because of [this reason](https://docs.esmvaltool.org/en/latest/recipes/recipe_hydrology.html#wflow-sbm-and-wflow-topoflex)
   - In the final 'if-block' of the function you need to check if you want the file name to be of 'Eday' or 'day' or maybe even 'CFDay'
   - then in the final block `if __name__ == "__main__":` check your home folder again
4. from the command line in `era5-hourly-to-daily/` run `py_cmor.py` with: `python py_cmor.sh &` to make sure it becomes a background task.
5. The data should then appear in the `era5-hourly-to-daily/cmorized_output` folder
6. You can check the `testing_values.ipynb` notebook to visually check your data, it also has some other built in checks
   - here you need to check the home folders again
   - and the `correct_data_dict`
   - you can change the `test_variables` beware that the plot that is generated is hardcoded to be 2x2, so 4 variables.

## The zarr version (`zarr_version/zarr_era5.py`)

`py_cmor.py` needs you to download netCDF files first. `zarr_version/zarr_era5.py` skips
that step and streams the hourly data straight out of the analysis-ready (ARCO) zarr
stores that the CDS publishes, then writes the same ESMValTool/OBS6 files. It can produce
**daily** output (like `py_cmor.py`) and **hourly** output.

Supported collections:

| flag | dataset | resolution | covers | filenames |
|---|---|---|---|---|
| `--collection era5-land` (default) | [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data) | 0.1° | 1950-01-02 → | `OBS6_ERA5-Land_reanaly_1_...` |
| `--collection era5` | [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview) | 0.25° | 1940 → | `OBS6_ERA5_reanaly_1_...` |

The dataset name in the filename and in the `dataset_id` global attribute matches
ESMValCore's native6 dataset names, so ESMValTool picks the right fixes up.

### Setup

```bash
pip install -r zarr_version/requirements.txt
cp zarr_version/.env.example zarr_version/.env   # then paste your CDS token into it
```

The token is your **CDS personal access token** from <https://cds.climate.copernicus.eu/profile>.

### Usage

```bash
# always look before you leap - prints the output sizes, downloads nothing
python zarr_version/zarr_era5.py --years 2015 --dry-run

# daily only, ERA5-Land, three variables (the default set)
python zarr_version/zarr_era5.py --years 2015 --frequency day

# daily + hourly, ERA5 single levels, hourly split into monthly files
python zarr_version/zarr_era5.py --collection era5 --years 1990-1995 \
    --frequency both --hourly-split month --output-dir ~/cmorized_output
```

Useful flags: `--variables`, `--reference-dir`, `--overwrite`, `--complevel`, `--progress`,
`--time-chunk`/`--space-chunk` (dask chunking), `--chunking time|geo` (which ARCO store to
read). Run with `--help` for the full list.

### Watching a long run

Every run plans all its files up front and then reports per-file timing, cumulative
progress and an ETA:

```
[PLAN] 6 file(s) to write, reading 52560 hourly time steps in total
[1/6] OBS6_ERA5-Land_reanaly_1_day_tas_2010-2010.nc: {'lat': 1801, 'lon': 3600, 'time': 365} -> ~8.8 GB uncompressed
[1/6] [OK] OBS6_ERA5-Land_reanaly_1_day_tas_2010-2010.nc in 15m 46s (1.6 GB on disk, 9.5 MB/s)
[1/6]      elapsed 1h 12m 30s |  16.7% | ETA 6h 02m 30s
...
[DONE] 6/6 files, 19.1 GB written in 7h 15m 03s (0 failures)
```

The percentage and ETA are weighted by *hours of source data read*, not by file count,
because the ARCO download dominates the runtime — a daily file costs the same to read as
the hourly file for the same period. It's an estimate: writing hourly output adds some
compression time on top, so hourly files land slightly behind the projection.

The per-file timer and ETA are always printed — there is no flag to enable them. dask's
live progress bar is also on by default; pass `--no-progress` to suppress it. It only
works on a terminal, so if you redirect to a log it disables itself automatically (it
redraws with `\r`, which would turn a log file into thousands of lines) and you keep the
per-file timer and ETA.

Output goes to `./cmorized_output` relative to wherever you run the script from, and the
resolved absolute path is printed as the first line of every run. Use `--output-dir` to
put it elsewhere.

### When the network drops

Reads from the ARCO store time out fairly regularly on long jobs. Each file is retried
`--retries` times (default 3, with 10s/20s/40s backoff) before it is given up on, and
`--http-timeout` (default 300s) sets how long a single read may stall first.

Each file is written to `<name>.nc.partial` and renamed only once the write succeeds, so
a failed or killed run can never leave a truncated file under the real name — which
matters, because the resume logic treats an existing file as complete. Any `.partial`
left over from a killed run is swept at the start of the next one.

For a long run, detach it and tail the log:

```bash
nohup python zarr_version/zarr_era5.py --years 2010 --frequency day > era5land_2010.log 2>&1 &
tail -f era5land_2010.log
```

The script switches stdout to line buffering itself, so the log fills in as it goes — you
don't need `python -u`. If a run dies partway, just rerun the same command: finished files
are skipped unless you pass `--overwrite`.

### Important notes

- **The script works on the full global grid.** One year of global *hourly* ERA5-Land is
  ~212 GB per variable (~9 GB daily); ERA5 single levels is ~34 GB hourly and ~1.4 GB
  daily. Use `--dry-run` first, and `--hourly-split month` to keep individual files
  manageable. Existing output files are skipped unless you pass `--overwrite`, so an
  interrupted run resumes.
- **Only `tas`, `pr` and `rsds` are available.** `evspsblpot` and `evspsbl` are *not*
  published as ARCO zarr by the CDS, in either collection — keep using `py_cmor.py` with
  an `era5cli`/CDS download for those two.
- Accumulated fields (`tp`, `ssrd`) are stored as **hourly** accumulations in both ARCO
  collections — ERA5-Land is *not* stored with the usual "accumulated since 00 UTC"
  convention here. So the conversion factors from `py_cmor.py` carry over unchanged and a
  plain daily mean is correct.
- Time stamps and bounds follow `py_cmor.py`: `tas` is stamped at 11:30 with bounds
  `[00:00, 23:00]`, the accumulated variables at 12:00 with bounds `[00:00, 24:00]`
  ([why](https://docs.esmvaltool.org/en/latest/recipes/recipe_hydrology.html#wflow-sbm-and-wflow-topoflex)).
- `--reference-dir` is optional. Without it the CMOR metadata is generated from the CMIP6
  tables (via ESMValCore if installed, otherwise a built-in fallback). With it, daily
  output inherits attributes and time bounds from your known-good files in
  `correct_data/`, exactly like `py_cmor.py` does.
- To add a variable, add a `VarSpec` to `VARIABLES` in `zarr_era5.py` (and a `CMOR_FALLBACK`
  entry if you don't have ESMValCore). The ARCO stores also hold `d2m`, `sp`, `strd`,
  `skt`, `u10`, `v10`, soil temperature/water, and snow.

For questions or if you need some data to test it on, please reach out to the [eWaterCycle team](https://www.ewatercycle.org/contact/)!

