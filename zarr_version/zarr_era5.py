"""CMORize ERA5 / ERA5-Land straight out of the ECMWF ARCO Zarr datastores.

This is the Zarr counterpart of ``py_cmor.py``.  Instead of reading local
netCDF files that were downloaded with ``era5cli``, it streams the hourly data
from the analysis-ready (ARCO) Zarr stores that the CDS publishes:

* ERA5-Land -- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data
* ERA5 single levels -- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=overview

It writes one file per variable per period, in the same ESMValTool/OBS6 layout
that ``py_cmor.py`` produces:

    OBS6_ERA5-Land_reanaly_1_day_tas_2015-2015.nc       (daily)
    OBS6_ERA5-Land_reanaly_1_E1hr_tas_2015-2015.nc      (hourly, --hourly-split year)
    OBS6_ERA5-Land_reanaly_1_E1hr_tas_201501-201501.nc  (hourly, --hourly-split month)

The dataset name in the filename follows the collection: ``ERA5-Land`` for
``--collection era5-land`` and ``ERA5`` for ``--collection era5``, matching
ESMValCore's native6 dataset names.

Notes on the ARCO stores (checked against the live stores, August 2026)
----------------------------------------------------------------------
* Accumulated fields (``tp``, ``ssrd``, ...) are stored as *hourly*
  accumulations in both collections -- ERA5-Land is **not** stored with the
  usual "accumulated since 00 UTC" convention.  That means the conversion
  factors from ``py_cmor.py`` (1/3600 for J m-2 -> W m-2, 1/3.6 for m -> kg
  m-2 s-1) apply unchanged, and a plain daily mean is the right daily
  aggregation.
* Coordinates are already named ``time`` / ``latitude`` / ``longitude`` and
  latitude is already ascending, so several of the fixups in ``py_cmor.py``
  (``valid_time`` renaming, ``number`` dropping, latitude flipping) are
  no-ops here.  They are kept as defensive checks.
* Evaporation is **not** published as ARCO Zarr, so ``pev``/``evspsblpot`` and
  ``e``/``evspsbl`` are fetched from the classic CDS request API instead
  (needs ``cdsapi``).  Via the CDS, ERA5-Land accumulates from 00 UTC, so only
  the 00:00 field of each day is downloaded -- that value is already the
  previous day's total, which is 24x less data than the hourly series.  These
  two are daily-only.  ERA5-Land reports evaporation as a positive magnitude
  (unlike ERA5's downward-positive convention, where evaporation is already
  negative), so the sign needed to make it negative is detected from the
  fetched data itself rather than hardcoded -- see
  ``_resolve_evaporation_sign``.

Adding a variable
-----------------
Add a ``VarSpec`` to ``VARIABLES`` and, if it lives in a store that is not
listed yet, add that store to the relevant ``Collection``.  If ESMValCore is
not installed, also add an entry to ``CMOR_FALLBACK``.

Size warning
------------
This script works on the **full global grid**.  One year of global hourly
ERA5-Land for a single variable is ~212 GB (1801 x 3600 x 8760 x 4 bytes);
ERA5 single levels is ~34 GB.  Daily output is 24x smaller.  Use ``--dry-run``
to print the estimate before committing to a job.
"""

from __future__ import annotations

import argparse
import calendar
import contextlib
import os
import sys
import time
import traceback
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import find_dotenv, load_dotenv

# ---------------------------------------------------------------------------
# ARCO store registry
# ---------------------------------------------------------------------------

ARCO_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-{chunking}-{bucket}"
    "/arco/{root}/{store}/{chunking}Chunked.zarr"
)


@dataclass(frozen=True)
class Store:
    """One ARCO Zarr store and the ERA5 short names it holds."""

    bucket: str  # numeric suffix of the cadl-arco-<chunking>-<bucket> bucket
    store: str  # path segment, e.g. "sfc-2m-temperature"
    variables: tuple[str, ...]


@dataclass(frozen=True)
class Collection:
    """A family of ARCO stores belonging to one CDS dataset."""

    root: str  # path segment, e.g. "reanalysis_era5_land"
    dataset: str  # ESMValTool dataset name, used in the output filename
    cds_dataset: str  # CDS request-API collection id, for the non-ARCO variables
    stores: tuple[Store, ...]

    def store_for(self, era5_name: str) -> Store:
        for store in self.stores:
            if era5_name in store.variables:
                return store
        raise KeyError(
            f"{era5_name!r} is not available as ARCO Zarr in {self.root!r}. "
            f"Available: {sorted(v for s in self.stores for v in s.variables)}"
        )


COLLECTIONS: dict[str, Collection] = {
    "era5-land": Collection(
        root="reanalysis_era5_land",
        # Matches ESMValCore's native6 dataset name (esmvalcore/cmor/_fixes/
        # native6/era5_land.py), so ESMValTool recognises the files.
        dataset="ERA5-Land",
        cds_dataset="reanalysis-era5-land",
        stores=(
            Store("005", "sfc-soil-water", ("swvl1", "swvl2", "swvl3", "swvl4")),
            Store("006", "sfc-soil-temperature", ("stl1", "stl2", "stl3", "stl4")),
            Store("007", "sfc-2m-temperature", ("d2m", "t2m")),
            Store("008", "sfc-wind", ("u10", "v10")),
            Store("009", "sfc-pressure-precipitation", ("sp", "tp")),
            Store("010", "sfc-radiation-heat", ("ssrd", "strd")),
            Store("030", "sfc-snow", ("sde", "snowc")),
            Store("043", "sfc-skin-temperature", ("skt",)),
        ),
    ),
    "era5": Collection(
        root="reanalysis_era5_single_levels",
        dataset="ERA5",
        cds_dataset="reanalysis-era5-single-levels",
        stores=(
            Store(
                "002",
                "sfc",
                (
                    "blh", "cbh", "d2m", "fdir", "fg10", "msl", "skt", "slhf",
                    "sp", "sshf", "ssrd", "sst", "strd", "t2m", "tcc", "tp",
                    "u10", "u100", "v10", "v100",
                ),
            ),
            Store("003", "wav", ()),  # ocean waves; no CMOR mapping defined here
        ),
    ),
}


# ---------------------------------------------------------------------------
# Variable registry -- the Zarr equivalent of py_cmor.py's rename_dict,
# conversion_dict and unit_dict rolled into one.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarSpec:
    era5_name: str  # short name in the ARCO store
    cmor_name: str  # CMIP6 / ESMValTool short name
    day_table: str  # CMOR table used for the daily files
    hour_table: str  # CMOR table used for the hourly files
    factor: float  # multiply the raw ERA5 values by this
    instantaneous: bool  # True for state variables, False for accumulations
    cds_name: str | None = None  # CDS request-API name, for fields absent from ARCO
    sign: float = 1.0  # ESMValCore flips evaporation to the CMOR sign convention


VARIABLES: dict[str, VarSpec] = {
    "tas": VarSpec("t2m", "tas", "day", "E1hr", 1.0, instantaneous=True),
    "pr": VarSpec("tp", "pr", "day", "E1hr", 1 / 3.6, instantaneous=False),
    "rsds": VarSpec("ssrd", "rsds", "day", "E1hr", 1 / 3600, instantaneous=False),
    # Not in ARCO: fetched from the CDS request API instead. Because ERA5-Land
    # accumulates from 00 UTC, we fetch only the 00:00 field, which is already the
    # daily total, so the factor converts m/day (not m/hour) to kg m-2 s-1:
    # 1000 kg m-3 / 86400 s = 1/86.4. sign=-1.0 is only the default expectation
    # (ERA5-Land reports evaporation as a positive magnitude, unlike ERA5's
    # downward-positive convention where it's already negative); the sign
    # actually applied is verified against the fetched data at runtime, see
    # _resolve_evaporation_sign.
    "evspsblpot": VarSpec("pev", "evspsblpot", "Eday", "E1hr", 1 / 86.4, instantaneous=False,
                          cds_name="potential_evaporation", sign=-1.0),
    "evspsbl": VarSpec("e", "evspsbl", "Eday", "E1hr", 1 / 86.4, instantaneous=False,
                       cds_name="total_evaporation", sign=-1.0),
}

# Used when ESMValCore is not installed, and for the fields ESMValCore does not
# expose (cell_methods).  Values follow the CMIP6 tables.
CMOR_FALLBACK: dict[str, dict[str, str | None]] = {
    "tas": {
        "units": "K",
        "standard_name": "air_temperature",
        "long_name": "Near-Surface Air Temperature",
        "cell_methods_day": "area: time: mean",
        "cell_methods_hour": "area: mean time: point",
        "positive": None,
    },
    "pr": {
        "units": "kg m-2 s-1",
        "standard_name": "precipitation_flux",
        "long_name": "Precipitation",
        "cell_methods_day": "area: time: mean",
        "cell_methods_hour": "area: time: mean",
        "positive": None,
    },
    "rsds": {
        "units": "W m-2",
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "long_name": "Surface Downwelling Shortwave Radiation",
        "cell_methods_day": "area: time: mean",
        "cell_methods_hour": "area: time: mean",
        "positive": "down",
    },
    "evspsblpot": {
        "units": "kg m-2 s-1",
        "standard_name": "water_potential_evaporation_flux",
        "long_name": "Potential Evapotranspiration",
        "cell_methods_day": "area: mean where land time: mean",
        "cell_methods_hour": "area: mean where land time: mean",
        "positive": None,
    },
    "evspsbl": {
        "units": "kg m-2 s-1",
        "standard_name": "water_evapotranspiration_flux",
        "long_name": "Evaporation Including Sublimation and Transpiration",
        "cell_methods_day": "area: mean where land time: mean",
        "cell_methods_hour": "area: mean where land time: mean",
        "positive": None,
    },
}

TIME_UNITS_DAY = "days since 1850-01-01 00:00:00"
TIME_UNITS_HOUR = "hours since 1850-01-01 00:00:00"
TIME_CALENDAR = "standard"


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def cmor_metadata(spec: VarSpec, daily: bool) -> dict[str, str | None]:
    """Return CMOR attributes, preferring ESMValCore's tables when available."""
    fallback = CMOR_FALLBACK[spec.cmor_name]
    meta = {
        "units": fallback["units"],
        "standard_name": fallback["standard_name"],
        "long_name": fallback["long_name"],
        "positive": fallback["positive"],
        "cell_methods": fallback["cell_methods_day" if daily else "cell_methods_hour"],
    }

    table = spec.day_table if daily else spec.hour_table
    try:
        from esmvalcore.cmor.table import CMOR_TABLES
    except ImportError:
        return meta

    entry = CMOR_TABLES["CMIP6"].get_variable(table, spec.cmor_name)
    if entry is None and not daily:
        # E1hr does not carry every variable; the daily table has the same
        # units/standard_name, which is all we take from it.
        entry = CMOR_TABLES["CMIP6"].get_variable(spec.day_table, spec.cmor_name)
    if entry is None:
        warnings.warn(
            f"ESMValCore has no CMOR entry for {spec.cmor_name} in {table}; "
            "falling back to the built-in table.",
            stacklevel=2,
        )
        return meta

    for key in ("units", "standard_name", "long_name", "positive"):
        value = getattr(entry, key, None)
        if value:
            meta[key] = str(value)
    return meta


def global_attributes(spec: VarSpec, dataset: str, source: str, table: str, frequency: str) -> dict[str, str]:
    return {
        "Conventions": "CF-1.7",
        "title": f"{dataset} data reformatted for ESMValTool",
        "source": source,
        "project_id": "OBS6",
        "dataset_id": dataset,
        "type": "reanaly",
        "version": "1",
        "tier": "3",
        "mip": table,
        "frequency": frequency,
        "modeling_realm": "atmos",
        "reference": "era5",
        "comment": _provenance_comment(spec, dataset, source),
    }


# ERA5 and ERA5-Land define potential evaporation differently, and the two must
# never be presented as interchangeable. Record which one this file holds.
PET_DEFINITION = {
    "ERA5-Land": "open-water (pan) evaporation",
    "ERA5": "evaporation over well-watered agricultural land",
}


def _provenance_comment(spec: VarSpec, dataset: str, source: str) -> str:
    """State honestly where the numbers came from and what they mean."""
    if "arco.datastores" in source:
        origin = f"Hourly {dataset} ARCO Zarr data"
    else:
        origin = (
            f"{dataset} daily accumulations, taken from the 00:00 UTC field of the "
            "following day via the CDS request API"
        )
    comment = f"{origin}, CMORized to {spec.cmor_name} by zarr_era5.py."
    if spec.cmor_name == "evspsblpot" and dataset in PET_DEFINITION:
        comment += (
            f" Note: {dataset} potential evaporation is {PET_DEFINITION[dataset]}; "
            "the ERA5 and ERA5-Land definitions differ and are not interchangeable."
        )
    return comment


def _cell_bounds(values: np.ndarray) -> np.ndarray:
    """Cell edges from cell centres, assuming a regular-ish axis."""
    centres = np.asarray(values, dtype="float64")
    if centres.size < 2:
        raise ValueError("cannot derive bounds from a single-point axis")
    edges = np.empty(centres.size + 1, dtype="float64")
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return np.column_stack([edges[:-1], edges[1:]])


def add_spatial_bounds(ds: xr.Dataset) -> xr.Dataset:
    """Attach CMIP6-style ``lat_bnds`` / ``lon_bnds``."""
    lat_bnds = np.clip(_cell_bounds(ds["lat"].values), -90.0, 90.0)
    ds["lat_bnds"] = (("lat", "bnds"), lat_bnds)
    ds["lon_bnds"] = (("lon", "bnds"), _cell_bounds(ds["lon"].values))
    ds["lat"].attrs.update(
        {
            "units": "degrees_north",
            "standard_name": "latitude",
            "long_name": "Latitude",
            "axis": "Y",
            "bounds": "lat_bnds",
        }
    )
    ds["lon"].attrs.update(
        {
            "units": "degrees_east",
            "standard_name": "longitude",
            "long_name": "Longitude",
            "axis": "X",
            "bounds": "lon_bnds",
        }
    )
    return ds


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    env_path = find_dotenv(usecwd=True) or str(Path(__file__).with_name(".env"))
    load_dotenv(env_path)
    import os

    key = os.getenv("CDS_API")
    if not key:
        raise SystemExit(
            "No CDS_API key found. Put `CDS_API=\"<your-cds-personal-access-token>\"` "
            f"in {Path(__file__).with_name('.env')} or in the environment."
        )
    return key.strip().strip('"').strip("'")


def store_url(collection: Collection, spec: VarSpec, chunking: str) -> str:
    store = collection.store_for(spec.era5_name)
    return ARCO_URL.format(chunking=chunking, bucket=store.bucket, root=collection.root, store=store.store)


def open_hourly(
    collection: Collection,
    spec: VarSpec,
    api_key: str,
    chunking: str,
    time_chunk: int,
    space_chunk: int,
    http_timeout: float = 300.0,
) -> tuple[xr.DataArray, str]:
    """Open one variable from its ARCO store as a lazy, dask-backed array."""
    url = store_url(collection, spec, chunking)
    storage_options: dict = {"headers": {"Authorization": f"Bearer {api_key}"}}
    try:
        # fsspec's default read timeout is short enough that long ARCO reads trip
        # it; a blown timeout surfaces as a message-less asyncio.TimeoutError.
        import aiohttp

        storage_options["client_kwargs"] = {
            "timeout": aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=http_timeout)
        }
    except ImportError:
        pass

    ds = xr.open_zarr(
        url,
        consolidated=True,
        storage_options=storage_options,
        chunks={"time": time_chunk, "latitude": space_chunk, "longitude": space_chunk},
    )
    if spec.era5_name not in ds:
        raise KeyError(f"{spec.era5_name!r} not found in {url} (has {list(ds.data_vars)})")
    return ds[spec.era5_name], url


def select_period(da: xr.DataArray, start: pd.Timestamp, end: pd.Timestamp, label: str) -> xr.DataArray:
    """Slice ``[start, end)`` and complain when the store does not cover it."""
    sliced = da.sel(time=slice(start, end - pd.Timedelta(hours=1)))
    if sliced.sizes["time"] == 0:
        raise ValueError(f"the store holds no data for {label} (covers {da.time.values[0]} .. {da.time.values[-1]})")
    expected = int((end - start).total_seconds() // 3600)
    if sliced.sizes["time"] != expected:
        warnings.warn(
            f"{label}: got {sliced.sizes['time']} hours, expected {expected}. "
            "The period is only partly covered by the store.",
            stacklevel=2,
        )
    return sliced


# ---------------------------------------------------------------------------
# CDS request-API source, for the variables the ARCO stores do not publish
# ---------------------------------------------------------------------------

CDS_API_URL = "https://cds.climate.copernicus.eu/api"


def _cds_client(api_key: str):
    try:
        import cdsapi
    except ImportError as err:  # pragma: no cover
        raise SystemExit(
            "The evaporation variables need the CDS request API. "
            "Install it with `pip install cdsapi`."
        ) from err
    return cdsapi.Client(url=CDS_API_URL, key=api_key, quiet=True, progress=False)


def _unzip_netcdf(archive: Path, target_dir: Path) -> list[Path]:
    """CDS returns netCDF wrapped in a zip; unpack and return the members."""
    if not zipfile.is_zipfile(archive):
        return [archive]
    with zipfile.ZipFile(archive) as zf:
        members = [n for n in zf.namelist() if n.endswith(".nc")]
        zf.extractall(target_dir)
    return [target_dir / name for name in members]


def fetch_cds_midnight_month(
    collection: Collection,
    spec: VarSpec,
    year: int,
    month: int,
    cache_dir: Path,
    api_key: str,
    area: list[float] | None = None,
) -> list[Path]:
    """Download the 00:00 UTC field for every day of one month.

    ERA5-Land accumulates from 00 UTC, so the 00:00 value of day D+1 already is
    the complete daily total for day D (verified against the hourly series).
    Fetching only that hour is 24x less data than the full hourly series, which
    is what makes a global 0.1 degree year practical at all.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "global" if area is None else "sub"
    stem = f"{collection.cds_dataset}_{spec.cds_name}_{tag}_{year}{month:02d}"
    extracted = sorted(cache_dir.glob(f"{stem}__*.nc"))
    if extracted:
        return extracted

    archive = cache_dir / f"{stem}.download"
    request = {
        "variable": [spec.cds_name],
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)],
        "time": ["00:00"],
        "data_format": "netcdf",
    }
    if area is not None:
        request["area"] = area

    print(f"[CDS]  requesting {spec.cds_name} {year}-{month:02d} ({len(request['day'])} fields)")
    started = time.monotonic()
    _cds_client(api_key).retrieve(collection.cds_dataset, request, str(archive))
    members = _unzip_netcdf(archive, cache_dir)

    renamed = []
    for index, member in enumerate(members):
        target = cache_dir / f"{stem}__{index}.nc"
        os.replace(member, target)
        renamed.append(target)
    archive.unlink(missing_ok=True)
    print(f"[CDS]  got {year}-{month:02d} in {human_time(time.monotonic() - started)}")
    return renamed


def cds_daily_totals(
    collection: Collection,
    spec: VarSpec,
    year: int,
    cache_dir: Path,
    api_key: str,
    area: list[float] | None = None,
) -> xr.DataArray:
    """Daily accumulation totals for `year`, from the 00:00 UTC fields.

    The 00:00 field of day D+1 holds the total for day D, so January needs the
    following 1 January as well, and every stamp is shifted back one day.
    """
    if collection.cds_dataset != "reanalysis-era5-land":
        raise ValueError(
            f"the 00:00-accumulation shortcut is only valid for ERA5-Land; "
            f"{collection.dataset} stores hourly accumulations, so use py_cmor.py "
            f"with an era5cli download for {spec.cmor_name}"
        )

    paths: list[Path] = []
    for month in range(1, 13):
        paths += fetch_cds_midnight_month(collection, spec, year, month, cache_dir, api_key, area)
    # 1 January of the next year carries 31 December's total.
    paths += fetch_cds_midnight_month(collection, spec, year + 1, 1, cache_dir, api_key, area)

    # Keep each horizontal field whole: the files are chunked 4-deep along
    # longitude, which makes any longitude-wise operation cross chunk
    # boundaries and force an expensive rechunk on every read. (The CDS
    # download is already on [0, 360), so no wrap is needed here.)
    ds = xr.open_mfdataset(
        sorted(paths), combine="by_coords",
        chunks={"longitude": -1},  # time keeps the files' own chunking
    )
    if spec.era5_name not in ds:
        raise KeyError(f"{spec.era5_name!r} not in the CDS download (has {list(ds.data_vars)})")
    da = ds[spec.era5_name]

    for extra in ("number", "expver"):
        if extra in da.coords:
            da = da.drop_vars(extra)
    time_name = "valid_time" if "valid_time" in da.dims else "time"

    # Shift each 00:00 stamp back to the day it actually accumulated over.
    stamps = pd.to_datetime(da[time_name].values) - pd.Timedelta(days=1)
    da = da.assign_coords({time_name: stamps})
    if time_name != "time":
        da = da.rename({time_name: "time"})
    # Only reorder if genuinely needed: sortby on a dask array is a fancy-index
    # shuffle across every file, whereas .sel on a monotonic index is a cheap
    # slice. combine="by_coords" already sorts, and shifting every stamp back by
    # one day preserves that order, so this is normally a no-op.
    stamps_sorted = bool((np.diff(da["time"].values).astype("timedelta64[s]").astype(int) > 0).all())
    if not stamps_sorted:
        print("[FIX] Sorting CDS time steps.")
        da = da.sortby("time")
    da = da.sel(time=str(year))

    expected = 366 if calendar.isleap(year) else 365
    if da.sizes["time"] != expected:
        warnings.warn(
            f"{spec.cmor_name} {year}: got {da.sizes['time']} days, expected {expected}",
            stacklevel=2,
        )
    return da


def _resolve_evaporation_sign(da: xr.DataArray, spec: VarSpec, collection: Collection) -> float:
    """Check the raw CDS sign convention instead of trusting ``VarSpec.sign`` blindly.

    ERA5 stores evaporation downward-positive, so evaporation itself is
    negative -- that is what ``py_cmor.py`` writes out for plain ERA5, with no
    sign flip at all. ERA5-Land, fetched here via the CDS request API, has
    been observed to report it as a positive magnitude instead. Either way,
    the CMORized output must end up negative (evaporation as a loss), which
    is the convention common in earth science and the one ERA5 already
    produces. Rather than hardcode which collection needs the flip, sample a
    few already-downloaded days and derive it from the data itself.
    """
    if spec.sign == 1.0:
        return spec.sign

    sample = da.isel(time=slice(0, min(5, da.sizes["time"]))).isel(
        latitude=slice(None, None, 4), longitude=slice(None, None, 4)
    ).values
    finite = sample[np.isfinite(sample)]
    finite = finite[finite != 0]
    if finite.size == 0:
        warnings.warn(
            f"{spec.cmor_name} ({collection.dataset}): could not sample the raw sign "
            f"convention (no non-zero values found); using the default sign {spec.sign:+.0f}.",
            stacklevel=2,
        )
        return spec.sign

    raw_is_positive = bool(np.median(finite) > 0)
    resolved = -1.0 if raw_is_positive else 1.0
    if resolved != spec.sign:
        warnings.warn(
            f"{spec.cmor_name} ({collection.dataset}): raw CDS values are "
            f"{'positive' if raw_is_positive else 'negative'}, which needs sign {resolved:+.0f} "
            f"to come out negative, not the {spec.sign:+.0f} hardcoded in VarSpec. "
            "Using the detected sign instead.",
            stacklevel=2,
        )
    return resolved


def cds_daily_dataset(collection: Collection, spec: VarSpec, year: int, da: xr.DataArray) -> xr.Dataset:
    """Turn CDS daily totals into the same shape to_daily() produces."""
    sign = _resolve_evaporation_sign(da, spec, collection)
    daily = da * (spec.factor * sign)
    daily.name = spec.cmor_name
    print(
        f"[INFO] Converted {spec.cmor_name} using factor 1/{1 / spec.factor:.1f}"
        f"{' and sign -1' if sign < 0 else ''}"
    )

    ds = daily.to_dataset()
    ds = _tidy_coordinates(ds)

    day_starts = pd.to_datetime(ds["time"].values).normalize()
    ds = ds.assign_coords(time=("time", day_starts + pd.Timedelta(hours=12)))
    ds["time_bnds"] = (
        ("time", "bnds"),
        np.column_stack(
            [day_starts.values, (day_starts + pd.Timedelta(hours=24)).values]
        ).astype("datetime64[ns]"),
    )
    return ds


# ---------------------------------------------------------------------------
# CMORization
# ---------------------------------------------------------------------------


def _tidy_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """The py_cmor.py coordinate fixups, kept as defensive no-ops for ARCO."""
    renames = {old: new for old, new in
               (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon"))
               if old in ds.variables}
    if renames:
        ds = ds.rename(renames)

    if "number" in ds.coords:
        ds = ds.drop_vars("number")
    if "height" in ds.data_vars:
        ds = ds.set_coords("height")

    # Drop the store's own time encoding ("seconds since 1970-01-01"); the CMOR
    # encoding is applied at write time, or taken from a reference file.
    ds["time"].encoding = {}

    if "lat" in ds.coords and np.any(np.diff(ds["lat"].values) < 0):
        print("[FIX] Reversing latitude to be ascending (-90 -> +90).")
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        # The ARCO stores use [-180, 180] and the CDS request API returns
        # [0, 360). Normalise to [0, 360) so every variable in a dataset lands
        # on exactly the same grid regardless of which source it came from.
        lon = ds["lon"].values
        if np.any(lon < 0.0):
            print("[FIX] Wrapping longitude from [-180, 180] to [0, 360).")
            wrapped = np.where(lon < 0.0, lon + 360.0, lon)
            ds = ds.assign_coords(lon=wrapped)
            # Wrapping leaves the axis cyclically rotated rather than randomly
            # ordered, so roll it back into place. sortby() would work too, but
            # it is a fancy-index shuffle: on a store chunked along longitude
            # that becomes an all-to-all rechunk behind HDF5's global read
            # lock, which effectively hangs. roll is a slice+concat.
            shift = int(np.argmin(wrapped))
            if shift:
                ds = ds.roll(lon=-shift, roll_coords=True)
        if np.any(np.diff(ds["lon"].values) < 0):
            print("[FIX] Sorting longitude ascending.")
            ds = ds.sortby("lon")
    return ds.sortby("time")


def to_daily(da: xr.DataArray, spec: VarSpec) -> xr.Dataset:
    """Hourly -> daily mean, with py_cmor.py's time stamps and bounds.

    ``tas`` is stamped at 11:30 with bounds [00:00, 23:00] because the wflow
    recipes expect the mean of the 24 instantaneous hourly samples; the
    accumulated variables are stamped at 12:00 with bounds [00:00, 24:00].
    See https://docs.esmvaltool.org/en/latest/recipes/recipe_hydrology.html
    """
    daily = da.resample(time="1D").mean(keep_attrs=False)
    daily = daily * spec.factor
    daily.name = spec.cmor_name
    print(f"[INFO] Converted {spec.cmor_name} using factor 1/{1 / spec.factor:.1f}")

    ds = daily.to_dataset()
    ds = _tidy_coordinates(ds)

    day_starts = pd.to_datetime(ds["time"].values)
    if not np.all(day_starts.hour == 0):
        raise ValueError("daily resample did not land on midnight; refusing to build time bounds")

    if spec.instantaneous:
        stamps = day_starts + pd.Timedelta(hours=11, minutes=30)
        ends = day_starts + pd.Timedelta(hours=23)
    else:
        stamps = day_starts + pd.Timedelta(hours=12)
        ends = day_starts + pd.Timedelta(hours=24)

    ds = ds.assign_coords(time=("time", stamps))
    ds["time_bnds"] = (
        ("time", "bnds"),
        np.column_stack([day_starts.values, ends.values]).astype("datetime64[ns]"),
    )
    return ds


def to_hourly(da: xr.DataArray, spec: VarSpec) -> xr.Dataset:
    """Hourly ARCO data -> hourly CMORized dataset (unit conversion only)."""
    hourly = da * spec.factor
    hourly.name = spec.cmor_name
    print(f"[INFO] Converted {spec.cmor_name} using factor 1/{1 / spec.factor:.1f}")

    ds = hourly.to_dataset()
    ds = _tidy_coordinates(ds)

    if not spec.instantaneous:
        # Accumulated fields are valid over the hour *ending* at the time stamp.
        stamps = pd.to_datetime(ds["time"].values)
        ds["time_bnds"] = (
            ("time", "bnds"),
            np.column_stack([(stamps - pd.Timedelta(hours=1)).values, stamps.values]).astype("datetime64[ns]"),
        )
    return ds


def add_height_coordinate(ds: xr.Dataset, spec: VarSpec) -> xr.Dataset:
    """Attach the CMIP6 2 m scalar ``height`` coordinate that tas needs.

    ERA5 t2m already carries this when it comes from era5cli/the CDS request
    API (py_cmor.py just promotes it to a coordinate); the ARCO Zarr stores
    used here do not publish it at all, so it has to be added by hand.
    """
    if spec.cmor_name != "tas" or "height" in ds.coords:
        return ds
    ds = ds.assign_coords(height=np.float64(2.0))
    ds["height"].attrs = {
        "long_name": "height",
        "standard_name": "height",
        "units": "m",
        "positive": "up",
        "axis": "Z",
    }
    return ds


def finalise(
    ds: xr.Dataset,
    spec: VarSpec,
    daily: bool,
    dataset: str,
    source: str,
    table: str,
) -> xr.Dataset:
    """Attach CMOR variable attributes, coordinate metadata and global attrs."""
    meta = cmor_metadata(spec, daily=daily)

    attrs = {
        "standard_name": meta["standard_name"],
        "long_name": meta["long_name"],
        "units": meta["units"],
        "cell_methods": meta["cell_methods"],
    }
    if meta["positive"]:
        attrs["positive"] = meta["positive"]
    ds[spec.cmor_name].attrs = attrs
    print(f"[INFO] Unit {meta['units']} added to {spec.cmor_name}")

    ds["time"].attrs = {
        "standard_name": "time",
        "long_name": "Time",
        "axis": "T",
        **({"bounds": "time_bnds"} if "time_bnds" in ds else {}),
    }
    ds = add_height_coordinate(ds, spec)
    ds = add_spatial_bounds(ds)
    ds.attrs = global_attributes(spec, dataset, source, table, "day" if daily else "1hr")
    return ds


def apply_reference_metadata(ds: xr.Dataset, spec: VarSpec, reference: Path, year: int) -> xr.Dataset:
    """Copy attributes/bounds from a known-good ESMValTool file (py_cmor.py style).

    Only meaningful for daily output, and only when the reference year has the
    same leap-ness as ``year`` so that the day-of-year bounds line up.
    """
    ref = xr.open_dataset(reference)
    try:
        ref_year = int(pd.to_datetime(ref["time"].values[0]).year)
        if calendar.isleap(ref_year) != calendar.isleap(year):
            raise ValueError(
                f"reference {reference.name} is year {ref_year} (leap={calendar.isleap(ref_year)}), "
                f"which does not match target year {year} (leap={calendar.isleap(year)})"
            )
        if ref.sizes["time"] != ds.sizes["time"]:
            raise ValueError(
                f"reference {reference.name} has {ref.sizes['time']} time steps, target has {ds.sizes['time']}"
            )

        ds[spec.cmor_name].attrs = dict(ref[spec.cmor_name].attrs)
        ds["time"].encoding = dict(ref["time"].encoding)
        ds.attrs = dict(ref.attrs)
        ds[spec.cmor_name].attrs["units"] = str(cmor_metadata(spec, daily=True)["units"])

        for name in ("time_bnds", "lat_bnds", "lon_bnds"):
            if name in ref:
                ds[name] = ref[name]

        if "time_bnds" in ref:
            ds["time_bnds"].values = _shift_years(ref["time_bnds"].values, year - ref_year)
        print(f"[INFO] Metadata harmonised with reference {reference.name}")
    finally:
        ref.close()
    return ds


def _shift_years(bounds: np.ndarray, offset: int) -> np.ndarray:
    """Move datetime64 bounds by a whole number of years, keeping day-of-year."""
    if offset == 0:
        return bounds
    flat = pd.to_datetime(bounds.ravel())
    shifted = pd.to_datetime(
        [stamp.replace(year=stamp.year + offset) for stamp in flat]
    )
    return shifted.values.astype("datetime64[ns]").reshape(bounds.shape)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def output_name(spec: VarSpec, table: str, period_label: str, dataset: str) -> str:
    """OBS6 layout: {project}_{dataset}_{type}_{version}_{mip}_{short_name}_{years}."""
    return f"OBS6_{dataset}_reanaly_1_{table}_{spec.cmor_name}_{period_label}-{period_label}.nc"


def build_encoding(ds: xr.Dataset, spec: VarSpec, daily: bool, complevel: int) -> dict[str, dict]:
    nlat, nlon = ds.sizes["lat"], ds.sizes["lon"]
    encoding: dict[str, dict] = {
        spec.cmor_name: {
            "dtype": "float32",
            "_FillValue": 1.0e20,
            "zlib": complevel > 0,
            "complevel": complevel,
            "chunksizes": (1, min(nlat, 720), min(nlon, 1440)),
        },
        "time": {
            "dtype": "float64",
            "units": TIME_UNITS_DAY if daily else TIME_UNITS_HOUR,
            "calendar": TIME_CALENDAR,
        },
        "lat": {"dtype": "float64", "_FillValue": None},
        "lon": {"dtype": "float64", "_FillValue": None},
    }
    if "height" in ds.coords:
        encoding["height"] = {"dtype": "float64", "_FillValue": None}
    # apply_reference_metadata() copies the reference file's time encoding onto
    # the dataset; when that happened, it wins over the default above.
    if ds["time"].encoding.get("units"):
        encoding["time"] = {
            "dtype": "float64",
            "units": ds["time"].encoding["units"],
            "calendar": ds["time"].encoding.get("calendar", TIME_CALENDAR),
        }
    for name in ("time_bnds", "lat_bnds", "lon_bnds"):
        if name in ds:
            encoding[name] = {"_FillValue": None}
    if "time_bnds" in ds:
        encoding["time_bnds"].update(
            {"dtype": "float64", "units": encoding["time"]["units"], "calendar": encoding["time"]["calendar"]}
        )
    return encoding


def human_bytes(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def human_time(seconds: float) -> str:
    """1h 04m 12s / 4m 12s / 12.3s"""
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _heartbeat(prefix: str, every: float):
    """A log-friendly progress ticker for when the live bar cannot be used.

    dask's ProgressBar redraws with \\r, which is useless once stdout is a log
    file. This prints one plain line every `every` seconds instead, so a long
    single-file write is not 15 minutes of silence.
    """
    from dask.callbacks import Callback

    class Heartbeat(Callback):
        def _start_state(self, dsk, state):
            self.total = max(len(state["dependencies"]), 1)
            self.done = 0
            self.started = self.last = time.monotonic()

        def _posttask(self, key, result, dsk, state, worker_id):
            self.done += 1
            now = time.monotonic()
            if now - self.last < every:
                return
            self.last = now
            fraction = self.done / self.total
            elapsed = now - self.started
            eta = elapsed / fraction - elapsed if fraction > 0 else 0.0
            print(
                f"{prefix}      writing {fraction * 100:5.1f}% "
                f"({self.done}/{self.total} chunks) "
                f"elapsed {human_time(elapsed)} | ETA {human_time(eta)}"
            )

    return Heartbeat()


def dask_progress(enabled: bool, prefix: str = "", heartbeat: float = 60.0):
    """Live bar on a terminal, periodic heartbeat lines when writing to a log."""
    if enabled and sys.stdout.isatty():
        try:
            from dask.diagnostics import ProgressBar

            return ProgressBar()
        except ImportError:  # pragma: no cover - dask is a hard dependency here
            warnings.warn("dask.diagnostics.ProgressBar unavailable", stacklevel=2)
    if heartbeat > 0:
        try:
            return _heartbeat(prefix, heartbeat)
        except ImportError:  # pragma: no cover
            warnings.warn("dask.callbacks unavailable; no progress output", stacklevel=2)
    return contextlib.nullcontext()


class RunTimer:
    """Elapsed time, throughput and ETA across the whole run.

    Progress is weighted by *hours of source data read* rather than by file
    count, because the ARCO download dominates the runtime and a daily file
    costs exactly as much to read as the hourly file for the same period.
    """

    def __init__(self, total_jobs: int, total_hours: int) -> None:
        self.total_jobs = total_jobs
        self.total_hours = max(total_hours, 1)
        self.done_hours = 0
        self.done_jobs = 0
        self.bytes_written = 0
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def finish(self, hours: int, nbytes: int) -> str:
        self.done_hours += hours
        self.done_jobs += 1
        self.bytes_written += nbytes
        elapsed = self.elapsed
        fraction = self.done_hours / self.total_hours
        eta = elapsed / fraction - elapsed if fraction > 0 else 0.0
        return (
            f"elapsed {human_time(elapsed)} | {fraction * 100:5.1f}% | "
            f"ETA {human_time(eta)}"
        )

    def summary(self, failures: int) -> str:
        return (
            f"[DONE] {self.done_jobs}/{self.total_jobs} files, "
            f"{human_bytes(self.bytes_written)} written in {human_time(self.elapsed)}"
            f" ({failures} failure{'' if failures == 1 else 's'})"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def periods(year: int, split: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(label, start, end) triples; ``end`` is exclusive."""
    if split == "year":
        return [(f"{year}", pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1))]
    out = []
    for month in range(1, 13):
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        out.append((f"{year}{month:02d}", start, pd.Timestamp(end)))
    return out


def find_reference(reference_dir: Path | None, spec: VarSpec, year: int, dataset: str) -> Path | None:
    """Pick a reference file with matching leap-ness, py_cmor.py style."""
    if reference_dir is None:
        return None
    pattern = f"OBS6_{dataset}_reanaly_1_{spec.day_table}_{spec.cmor_name}_*.nc"
    candidates = sorted(reference_dir.glob(pattern))
    for candidate in candidates:
        ref_year = int(candidate.stem.rsplit("_", 1)[-1].split("-")[0])
        if calendar.isleap(ref_year) == calendar.isleap(year):
            return candidate
    if candidates:
        warnings.warn(
            f"no reference file for {spec.cmor_name} with leap={calendar.isleap(year)} in {reference_dir}",
            stacklevel=2,
        )
    return None


@dataclass
class Job:
    """One output file, planned before anything is downloaded."""

    spec: VarSpec
    source: str
    hourly_da: xr.DataArray | None  # None for CDS-sourced variables
    year: int
    daily: bool
    table: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    out_path: Path

    @property
    def hours(self) -> int:
        """Hours of source data this job has to read -- the ETA weight."""
        return int((self.end - self.start).total_seconds() // 3600)


def plan_jobs(args: argparse.Namespace, output_dir: Path) -> tuple[list[Job], int]:
    """Open each store once and work out every file the run will produce."""
    collection = COLLECTIONS[args.collection]
    api_key = load_api_key()
    frequencies = ["day", "hour"] if args.frequency == "both" else [args.frequency]
    jobs: list[Job] = []
    failures = 0

    for cmor_name in args.variables:
        spec = VARIABLES[cmor_name]
        via_cds = spec.cds_name is not None

        if via_cds and collection.cds_dataset != "reanalysis-era5-land":
            print(
                f"[SKIP] {cmor_name}: only supported for --collection era5-land. "
                f"{collection.dataset} stores hourly accumulations, so the 00:00 "
                "shortcut does not apply; use py_cmor.py with an era5cli download."
            )
            failures += 1
            continue

        if via_cds:
            # Not in ARCO; downloaded from the CDS request API instead.
            hourly_da = None
            source = f"{CDS_API_URL} :: {collection.cds_dataset} :: {spec.cds_name}"
            print(f"[INFO] {cmor_name}: via CDS request API ({collection.cds_dataset})")
            if "hour" in frequencies:
                print(
                    f"[SKIP] {cmor_name}: hourly output is not supported for CDS-sourced "
                    "variables (it would need the full 8760-step download)"
                )
        else:
            try:
                hourly_da, source = open_hourly(
                    collection, spec, api_key, args.chunking, args.time_chunk,
                    args.space_chunk, args.http_timeout,
                )
            except (KeyError, OSError, ValueError) as err:
                print(f"[SKIP] {cmor_name}: {err}")
                failures += 1
                continue

            print(f"[INFO] {cmor_name}: {source}")
            print(f"[INFO] store covers {hourly_da.time.values[0]} .. {hourly_da.time.values[-1]}")

        for year in args.years:
            for frequency in frequencies:
                daily = frequency == "day"
                if via_cds and not daily:
                    continue
                table = spec.day_table if daily else spec.hour_table
                split = "year" if daily else args.hourly_split
                for label, start, end in periods(year, split):
                    out_path = output_dir / output_name(spec, table, label, collection.dataset)
                    if out_path.exists() and not args.overwrite:
                        expected = int((end - start).total_seconds() // 3600)
                        if daily:
                            expected //= 24
                        complete, why = existing_is_complete(
                            out_path, spec.cmor_name, expected, args.verify_existing
                        )
                        if complete:
                            print(f"[SKIP] {out_path.name} already done{why}")
                            continue
                        # Present but not trustworthy: rebuild it rather than
                        # silently treat a bad file as finished work.
                        print(f"[REDO] {out_path.name} exists but {why}; rebuilding")
                    jobs.append(
                        Job(spec, source, hourly_da, year, daily, table, label, start, end, out_path)
                    )
    return jobs, failures


def existing_is_complete(
    path: Path, cmor_name: str, expected_steps: int, verify: bool
) -> tuple[bool, str]:
    """Is an already-present output file finished, or only half there?

    Reads the header only, so it costs milliseconds. Guards against the failure
    mode this script actually hits: a run dies mid-write and leaves a file that
    the resume logic would otherwise accept as complete.
    """
    if not verify:
        return True, " (not verified)"
    try:
        with xr.open_dataset(path) as ds:
            if cmor_name not in ds.data_vars:
                return False, f"has no {cmor_name} variable"
            steps = ds.sizes.get("time", 0)
            if steps != expected_steps:
                return False, f"has {steps} time steps, expected {expected_steps}"
    except Exception as err:  # noqa: BLE001 - any failure to read means redo it
        detail = " ".join(str(err).split())[:120]
        return False, f"is unreadable ({type(err).__name__}: {detail})"
    return True, f" ({expected_steps} time steps)"


def _transient_errors() -> tuple[type[BaseException], ...]:
    """Network hiccups worth retrying. ARCO reads time out fairly regularly."""
    errors: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:  # aiohttp is fsspec's async HTTP backend
        from aiohttp import ClientError, ServerTimeoutError

        errors += [ClientError, ServerTimeoutError]
    except ImportError:
        pass
    return tuple(errors)


TRANSIENT_ERRORS = _transient_errors()


def sweep_partials(output_dir: Path) -> None:
    """Remove leftover *.partial files from an earlier interrupted run.

    On Windows a failed netCDF write can survive the cleanup in run_job(): the
    traceback keeps the Dataset alive, and HDF5 recreates the header file when
    it is finally garbage-collected. Sweeping at startup catches those.
    """
    stale = sorted(output_dir.glob("*.partial"))
    for path in stale:
        try:
            path.unlink()
            print(f"[INFO] removed stale partial file {path.name}")
        except OSError as err:
            print(f"[WARN] could not remove stale partial file {path.name}: {err}")


def run_job(
    job: Job, args: argparse.Namespace, reference_dir: Path | None, prefix: str
) -> tuple[int, float]:
    """Build and write one output file. Returns (bytes on disk, seconds)."""
    spec = job.spec
    collection = COLLECTIONS[args.collection]
    dataset = collection.dataset

    if spec.cds_name is not None and args.dry_run:
        # Never fetch under --dry-run: the CDS path downloads before it can know
        # the real shape, so report the plan instead.
        days = 366 if calendar.isleap(job.year) else 365
        print(
            f"{prefix} {job.out_path.name}: {days} days via {collection.cds_dataset}, "
            f"13 CDS requests ({spec.cds_name}, 00:00 fields only)"
        )
        return 0, 0.0

    if spec.cds_name is not None:
        totals = cds_daily_totals(
            collection, spec, job.year,
            Path(args.cds_cache).expanduser().resolve(),
            load_api_key(), args.area,
        )
        ds = cds_daily_dataset(collection, spec, job.year, totals)
    else:
        raw = select_period(job.hourly_da, job.start, job.end, f"{spec.cmor_name} {job.label}")
        ds = to_daily(raw, spec) if job.daily else to_hourly(raw, spec)
    ds = finalise(ds, spec, job.daily, dataset, job.source, job.table)

    if job.daily and reference_dir is not None:
        reference = find_reference(reference_dir, spec, job.year, dataset)
        if reference is not None:
            ds = apply_reference_metadata(ds, spec, reference, job.year)

    size = ds[spec.cmor_name].size * 4
    print(f"{prefix} {job.out_path.name}: {dict(ds.sizes)} -> ~{human_bytes(size)} uncompressed")
    if args.dry_run:
        return 0, 0.0

    # Write to a temporary name and rename only on success. A half-written file
    # must never be left under the final name: the resume logic skips files that
    # already exist, so it would be silently taken as complete.
    partial = job.out_path.with_name(job.out_path.name + ".partial")
    _quiet_unlink(partial)

    job_started = time.monotonic()
    try:
        with dask_progress(args.progress, prefix, args.heartbeat):
            ds.to_netcdf(
                partial,
                format="NETCDF4",
                encoding=build_encoding(ds, spec, job.daily, args.complevel),
            )
    except BaseException:
        ds.close()
        _quiet_unlink(partial)
        raise
    job_elapsed = time.monotonic() - job_started

    # On Windows os.replace fails if anything holds the destination open -- a
    # Jupyter kernel that called xr.open_dataset on it is the usual culprit.
    # The data is already safely written, so wait rather than throw it away.
    for attempt in range(1, 7):
        try:
            os.replace(partial, job.out_path)
            break
        except PermissionError as err:
            if attempt == 6:
                raise PermissionError(
                    f"{job.out_path.name} is open in another program, so the finished "
                    "file could not be moved into place. Close it (e.g. restart the "
                    "Jupyter kernel, or call ds.close()) and rerun -- the completed "
                    f"data is kept as {partial.name}, nothing was lost."
                ) from err
            print(
                f"{prefix} [WAIT] {job.out_path.name} is locked by another process; "
                f"retrying in 10s ({attempt}/5)"
            )
            time.sleep(10)

    on_disk = job.out_path.stat().st_size
    rate = size / job_elapsed if job_elapsed > 0 else 0.0
    print(
        f"{prefix} [OK] {job.out_path.name} in {human_time(job_elapsed)} "
        f"({human_bytes(on_disk)} on disk, {human_bytes(rate)}/s)"
    )
    return on_disk, job_elapsed


def _quiet_unlink(path: Path) -> None:
    """Delete a file without ever masking the error that led us here."""
    try:
        path.unlink(missing_ok=True)
    except OSError as err:
        print(f"[WARN] could not remove {path.name}: {err}")


def process(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] writing to {output_dir}")
    reference_dir = Path(args.reference_dir).expanduser() if args.reference_dir else None

    jobs, failures = plan_jobs(args, output_dir)
    if not jobs:
        print("[DONE] nothing to do")
        return failures

    timer = RunTimer(len(jobs), sum(job.hours for job in jobs))
    print(
        f"[PLAN] {len(jobs)} file(s) to write, reading {timer.total_hours} hourly "
        f"time steps in total"
    )

    sweep_partials(output_dir)

    for index, job in enumerate(jobs, start=1):
        spec = job.spec
        prefix = f"[{index}/{len(jobs)}]"
        on_disk = 0
        try:
            for attempt in range(1, max(args.retries, 1) + 1):
                try:
                    on_disk, job_elapsed = run_job(job, args, reference_dir, prefix)
                    break
                except TRANSIENT_ERRORS as err:
                    if attempt == args.retries:
                        raise
                    backoff = min(120, 10 * 2 ** (attempt - 1))
                    print(
                        f"{prefix} [RETRY] {type(err).__name__} on attempt "
                        f"{attempt}/{args.retries}; retrying in {backoff}s"
                    )
                    time.sleep(backoff)
            if args.dry_run:
                continue
            print(f"{prefix}      {timer.finish(job.hours, on_disk)}")
        except Exception as err:  # noqa: BLE001 - keep going across periods
            failures += 1
            # Some netCDF/HDF5 and fsspec errors carry an empty message
            # (asyncio.TimeoutError is the common one here), so always print the
            # type and the traceback, or the failure is undiagnosable.
            print(f"{prefix} [ERROR] Failed on {spec.cmor_name} {job.label}: "
                  f"{type(err).__name__}: {err}")
            traceback.print_exc(file=sys.stdout)
            timer.finish(job.hours, 0)

    if not args.dry_run:
        print(timer.summary(failures))
    return failures


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-", 1)
            years.extend(range(int(first), int(last) + 1))
        elif part:
            years.append(int(part))
    if not years:
        raise argparse.ArgumentTypeError("no years given")
    return sorted(set(years))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python zarr_era5.py --years 2015 --variables tas --frequency day\n"
            "  python zarr_era5.py --collection era5 --years 1990-1995 --frequency both "
            "--hourly-split month\n"
            "  python zarr_era5.py --years 2015 --dry-run   # print sizes, write nothing\n"
        ),
    )
    parser.add_argument("--collection", choices=sorted(COLLECTIONS), default="era5-land")
    parser.add_argument("--years", type=parse_years, required=True, help="e.g. 2015 or 1990-1999 or 1990,1995")
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=sorted(VARIABLES),
        default=["tas", "pr", "rsds"],
        help="CMOR short names. The default is the three ARCO Zarr variables, which "
             "stream fast. evspsblpot and evspsbl are supported but opt-in: they go "
             "through the CDS request API (~13 queued requests per year, ERA5-Land "
             "daily only), so they are not run unless you ask for them.",
    )
    parser.add_argument("--frequency", choices=("day", "hour", "both"), default="both")
    parser.add_argument(
        "--hourly-split",
        choices=("year", "month"),
        default="year",
        help="one hourly file per year (default) or per month",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "cmorized_output"),
        help="where to write the netCDF files (default: ./cmorized_output, relative to "
             "where you run the script from)",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="optional folder of known-good ESMValTool files; when given, daily "
             "output inherits their attributes and time bounds (py_cmor.py behaviour)",
    )
    parser.add_argument(
        "--chunking",
        choices=("time", "geo"),
        default="time",
        help="'time' = time-chunked store (one time step per chunk, best for whole-globe maps); "
             "'geo' = geo-chunked store (best for long time series at a point)",
    )
    parser.add_argument("--time-chunk", type=int, default=24, help="dask chunk size along time (hours)")
    parser.add_argument("--space-chunk", type=int, default=1024, help="dask chunk size along lat/lon")
    parser.add_argument("--complevel", type=int, default=4, help="netCDF deflate level, 0 disables compression")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="attempts per file before giving up, for transient network errors "
             "(default: 3, with 10s/20s/40s backoff). ARCO reads time out fairly often.",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=300.0,
        help="seconds to wait on a single read from the ARCO store before treating it "
             "as a timeout (default: 300)",
    )
    parser.add_argument(
        "--cds-cache",
        default="cds_cache",
        help="where CDS downloads are kept so reruns do not refetch them "
             "(default: ./cds_cache). Only used for evspsblpot/evspsbl.",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=None,
        help="optional CDS subset box for evspsblpot/evspsbl, e.g. --area 53 4 51 7. "
             "Omit for the whole globe.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild files that already exist instead of skipping them",
    )
    parser.add_argument(
        "--verify-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="before skipping an existing file, check it opens and has the expected "
             "number of time steps; rebuild it if not (default: on). --no-verify-existing "
             "skips on filename alone.",
    )
    parser.add_argument("--dry-run", action="store_true", help="report sizes without downloading or writing")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dask's live progress bar while each file is written (default: on; "
             "--no-progress disables it). Needs a terminal; when stdout is redirected "
             "the --heartbeat lines are used instead.",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=60.0,
        help="seconds between progress lines while a single file is being written "
             "(default: 60, 0 disables). These are plain lines, so they work in a log file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Python block-buffers stdout when it is redirected, so `... > run.log &`
    # would show nothing for hours. Line buffering makes the timer usable there.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)

    via_cds = [v for v in args.variables if VARIABLES[v].cds_name]
    if via_cds:
        print(
            f"[INFO] {', '.join(via_cds)}: not published as ARCO Zarr, so these come from "
            "the CDS request API (daily output only; needs `pip install cdsapi`)."
        )

    if args.frequency in ("hour", "both") and not args.dry_run:
        print(
            "[WARN] Hourly output on the full global grid is large: roughly 212 GB per "
            "variable-year for ERA5-Land and 34 GB for ERA5. Run with --dry-run first."
        )

    return process(args)


if __name__ == "__main__":
    sys.exit(min(main(), 1))