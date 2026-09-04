#!/usr/bin/env python
"""Check -- and optionally repair -- CMORized ERA5/ERA5-Land netCDF files.

Two things went wrong in files written by earlier versions of this repo, and
both are cheap to correct in place instead of re-downloading and re-CMORizing:

1. **Longitude convention.**  Output was normalised to ``[-180, 180]``; it must
   be ``[0, 360)`` and ascending.  Fixing it is a coordinate relabel plus a
   cyclic roll of the data, so the values keep the longitudes they belong to.
   ``lon_bnds`` is recomputed from the corrected axis.
2. **Evaporation sign.**  ``evspsblpot`` / ``evspsbl`` must be negative
   (evaporation as a loss of water).  ERA5-Land reports potential evaporation
   as a positive magnitude, and files written before the sign flip went in
   carry it that way.  Fixing it is a multiplication by -1.
3. **Missing height coordinate.**  ``tas`` must carry the CMIP6 2 m scalar
   ``height`` coordinate.  ERA5's t2m carries it through from era5cli/the CDS
   request API, but the ARCO Zarr stores used by ``zarr_era5.py`` do not
   publish it at all, so files written before that was added are missing it
   entirely.  Fixing it adds the scalar coordinate back (value 2.0, units m).

Usage
-----
Report only (nothing is written)::

    python fix_cmorized.py ../cmorized_output

Repair, writing corrected copies to another folder and leaving the originals
alone::

    python fix_cmorized.py ../cmorized_output --fix --output-dir ../fixed

Repair in place (each file is written to ``*.partial`` first and only swapped
in once the write succeeded)::

    python fix_cmorized.py ../cmorized_output --fix

The checks are read-only and open only what they need, so running without
``--fix`` on a folder of multi-GB files takes seconds.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zarr_era5 import VARIABLES, _cell_bounds, human_bytes  # noqa: E402

# netCDF encoding keys worth carrying over from the source file. Everything
# else xarray puts in .encoding (source, original_shape, preferred_chunks, ...)
# is bookkeeping that the writer rejects or ignores.
ENCODING_KEYS = (
    "dtype", "_FillValue", "zlib", "complevel", "shuffle", "fletcher32",
    "contiguous", "chunksizes", "units", "calendar", "scale_factor",
    "add_offset", "least_significant_digit",
)

# Variables whose CMORized values must come out negative.
NEGATIVE_VARIABLES = {name for name, spec in VARIABLES.items() if spec.sign == -1.0}


def find_variable(ds: xr.Dataset) -> str | None:
    """The CMOR variable in this file, ignoring the bounds variables."""
    for name in ds.data_vars:
        if str(name) in VARIABLES:
            return str(name)
    return None


def check_longitude(ds: xr.Dataset) -> list[str]:
    """Problems with the longitude axis, worst first."""
    problems: list[str] = []
    if "lon" not in ds.coords:
        return ["has no lon coordinate"]
    lon = np.asarray(ds["lon"].values, dtype="float64")
    if lon.min() < 0.0:
        problems.append(
            f"longitude runs [{lon.min():.3f}, {lon.max():.3f}], expected [0, 360)"
        )
    elif np.any(np.diff(lon) < 0):
        problems.append("longitude is not ascending")
    if "lon_bnds" in ds.variables and not problems:
        expected = _cell_bounds(lon)
        if not np.allclose(np.asarray(ds["lon_bnds"].values), expected, atol=1e-6):
            problems.append("lon_bnds do not match the lon axis")
    return problems


def check_sign(ds: xr.Dataset, name: str, stride: int) -> list[str]:
    """Is a variable that must be negative actually stored negative?

    Sampled rather than read whole: a global year is several GB, and the sign
    convention is a property of the whole field, so a few strided days settle
    it. Mirrors _resolve_evaporation_sign() in zarr_era5.py.
    """
    if name not in NEGATIVE_VARIABLES:
        return []
    da = ds[name]
    sample = da.isel(time=slice(0, min(5, da.sizes.get("time", 1))))
    if "lat" in da.dims and "lon" in da.dims:
        sample = sample.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
    values = np.asarray(sample.values)
    finite = values[np.isfinite(values)]
    finite = finite[finite != 0]
    if finite.size == 0:
        return [f"{name}: could not sample the sign (no non-zero values found)"]
    median = float(np.median(finite))
    if median > 0:
        return [f"{name}: values are positive, expected negative (median {median:+.3e})"]
    return []


def check_height(ds: xr.Dataset, name: str) -> list[str]:
    """tas must carry the CMIP6 2 m scalar height coordinate."""
    if name != "tas":
        return []
    if "height" not in ds.coords:
        return ["tas has no height coordinate (ERA5-Land ARCO does not publish one)"]
    value = float(np.asarray(ds["height"].values))
    if not np.isclose(value, 2.0):
        return [f"height coordinate is {value}, expected 2.0"]
    return []


def fix_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Relabel to [0, 360) and roll the data so values keep their longitude."""
    lon = np.asarray(ds["lon"].values, dtype="float64")
    # assign_coords() replaces the variable outright, taking its CF attributes
    # (units, standard_name, bounds, ...) with it, so put them back by hand.
    lon_attrs, lon_encoding = dict(ds["lon"].attrs), dict(ds["lon"].encoding)
    bnds_attrs = dict(ds["lon_bnds"].attrs) if "lon_bnds" in ds.variables else {}

    wrapped = np.where(lon < 0.0, lon + 360.0, lon)
    ds = ds.assign_coords(lon=wrapped)
    ds["lon"].attrs = lon_attrs
    ds["lon"].encoding = lon_encoding
    # Wrapping leaves the axis cyclically rotated, not randomly ordered, so a
    # roll puts it back in order. sortby() would work too but it is a
    # fancy-index shuffle across every chunk; roll is a slice+concat.
    shift = int(np.argmin(wrapped))
    if shift:
        ds = ds.roll(lon=-shift, roll_coords=True)
    if "lon_bnds" in ds.variables:
        # The rolled bounds still carry the old [-180, 180] numbers, so derive
        # them again from the corrected centres.
        ds["lon_bnds"] = (("lon", "bnds"), _cell_bounds(ds["lon"].values))
        ds["lon_bnds"].attrs = bnds_attrs
    return ds


def fix_sign(ds: xr.Dataset, name: str) -> xr.Dataset:
    """Flip a positive-magnitude evaporation field to the negative convention."""
    attrs = dict(ds[name].attrs)
    encoding = dict(ds[name].encoding)
    ds[name] = ds[name] * -1.0
    ds[name].attrs = attrs
    ds[name].encoding = encoding
    return ds


def fix_height(ds: xr.Dataset) -> xr.Dataset:
    """Attach the missing CMIP6 2 m scalar height coordinate to tas."""
    ds = ds.assign_coords(height=np.float64(2.0))
    ds["height"].attrs = {
        "long_name": "height",
        "standard_name": "height",
        "units": "m",
        "positive": "up",
        "axis": "Z",
    }
    return ds


def source_encoding(ds: xr.Dataset) -> dict[str, dict]:
    """Keep the file's own dtypes, compression and time units on rewrite."""
    encoding: dict[str, dict] = {}
    for name in list(ds.data_vars) + list(ds.coords):
        kept = {k: v for k, v in ds[name].encoding.items() if k in ENCODING_KEYS}
        # A float variable with no _FillValue gets one (NaN) invented for it on
        # write unless it is pinned to None. The coordinates and the bounds
        # variables are written without one, and CMOR files must stay that way.
        if "_FillValue" not in kept and "_FillValue" not in ds[name].attrs:
            kept["_FillValue"] = None
        encoding[str(name)] = kept
    return encoding


def note_history(ds: xr.Dataset, notes: list[str]) -> xr.Dataset:
    """Record what was changed, so a corrected file says so about itself."""
    entry = "fix_cmorized.py: " + "; ".join(notes)
    existing = ds.attrs.get("history", "")
    ds.attrs["history"] = f"{existing}\n{entry}".strip()
    return ds


def process_file(path: Path, args: argparse.Namespace) -> tuple[str, list[str]]:
    """Returns (status, problems) where status is ok / fixed / broken / skipped."""
    with xr.open_dataset(path, chunks={"time": args.time_chunk}) as ds:
        name = find_variable(ds)
        if name is None:
            return "skipped", [f"no known CMOR variable (has {list(ds.data_vars)})"]

        lon_problems = check_longitude(ds)
        sign_problems = check_sign(ds, name, args.sample_stride)
        height_problems = check_height(ds, name)
        problems = lon_problems + sign_problems + height_problems
        if not problems:
            return "ok", []
        if not args.fix:
            return "broken", problems
        if "has no lon coordinate" in lon_problems:
            return "broken", problems + ["cannot be fixed automatically"]

        fixed = ds
        notes: list[str] = []
        if lon_problems:
            fixed = fix_longitude(fixed)
            notes.append("longitude rewritten to [0, 360)")
        if any("expected negative" in problem for problem in sign_problems):
            fixed = fix_sign(fixed, name)
            notes.append(f"{name} sign flipped to negative")
        if height_problems:
            fixed = fix_height(fixed)
            notes.append("height coordinate (2 m) added to tas")
        if not notes:
            return "broken", problems + ["nothing could be fixed"]
        fixed = note_history(fixed, notes)

        destination = path if args.output_dir is None else args.output_dir / path.name
        tmp = destination.with_suffix(destination.suffix + ".partial")
        encoding = source_encoding(ds)
        if "height" in fixed.coords and "height" not in encoding:
            encoding["height"] = {"dtype": "float64", "_FillValue": None}
        fixed.to_netcdf(tmp, encoding=encoding, unlimited_dims=[])

    # The source is closed by here, which Windows requires before the swap.
    os.replace(tmp, destination)
    return "fixed", problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fix_cmorized.py ../cmorized_output\n"
            "  python fix_cmorized.py ../cmorized_output --fix --output-dir ../fixed\n"
            "  python fix_cmorized.py ../cmorized_output --fix   # in place\n"
        ),
    )
    parser.add_argument("folder", help="folder of CMORized netCDF files to check")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="rewrite the files that need it; without this the run only reports",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="write corrected copies here instead of overwriting the originals",
    )
    parser.add_argument(
        "--pattern", default="*.nc", help="which files to look at (default: *.nc)"
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="days held in memory at a time while rewriting (default: 1)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=4,
        help="lat/lon stride used when sampling the evaporation sign (default: 4)",
    )
    args = parser.parse_args(argv)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[ERROR] {folder} is not a folder")
        return 1
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir).expanduser().resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.output_dir == folder:
            args.output_dir = None  # same thing as an in-place fix

    paths = sorted(folder.glob(args.pattern))
    if not paths:
        print(f"[DONE] no files matching {args.pattern} in {folder}")
        return 0
    print(f"[INFO] checking {len(paths)} file(s) in {folder}")
    if not args.fix:
        print("[INFO] report only; rerun with --fix to correct them")

    counts = {"ok": 0, "fixed": 0, "broken": 0, "skipped": 0, "error": 0}
    for index, path in enumerate(paths, start=1):
        prefix = f"[{index}/{len(paths)}]"
        try:
            status, problems = process_file(path, args)
        except Exception as err:  # noqa: BLE001 - one bad file must not stop the sweep
            counts["error"] += 1
            print(f"{prefix} [ERROR] {path.name}: {type(err).__name__}: {err}")
            traceback.print_exc(file=sys.stdout)
            continue
        counts[status] += 1
        if status == "ok":
            print(f"{prefix} [OK]   {path.name}")
        else:
            tag = {"fixed": "[FIX]", "broken": "[BAD]", "skipped": "[SKIP]"}[status]
            print(f"{prefix} {tag}  {path.name}")
            for problem in problems:
                print(f"           - {problem}")
            if status == "fixed":
                target = path if args.output_dir is None else args.output_dir / path.name
                print(f"           -> wrote {target} ({human_bytes(target.stat().st_size)})")

    print(
        f"[DONE] {counts['ok']} already correct, {counts['fixed']} fixed, "
        f"{counts['broken']} still wrong, {counts['skipped']} skipped, "
        f"{counts['error']} errored"
    )
    return counts["broken"] + counts["error"]


if __name__ == "__main__":
    sys.exit(min(main(), 1))
