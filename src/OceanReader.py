from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import glob
import os
import numpy as np
import xarray as xr


Source = Literal["ACCESS-OM2", "EN4", "ORAS5"]


@dataclass(frozen=True)
class OceanReader:
    """
    Read ocean T/S (and other) data from supported sources and return an xarray DataArray.

    Notes
    -----
    - Repository can be called `obs-seaice-analysis`, but the importable package must be
      `obs_seaice_analysis`.
    - ACCESS-OM2 reading assumes the `intake` + `access-nri` catalogue is available.
    """

    base_dir: str = "/g/data/gv90/wrh581"

    def read(
        self,
        *,
        src: Source,
        expt: str = "obs",
        var: str,
        start_year: int,
        end_year: int,
        latmin: float = -90.0,
        latmax: float = 90.0,
        zmin: float = 0.0,
        zmax: float = 6000.0,
        freq: str = "1mon",
        chunks: dict | int | str | None = None,
        parallel: bool = True,
        decode_timedelta: bool = False,
    ) -> xr.DataArray:
        """
        Parameters
        ----------
        src
            Data source: "ACCESS-OM2", "EN4", or "ORAS5".
        expt
            Intake experiment key for ACCESS-OM2 catalog (ignored for EN4/ORAS5).
        var
            Variable name in files.
        start_year, end_year
            Inclusive year range.
        latmin, latmax
            Latitude selection.
        zmin, zmax
            Vertical selection (units depend on source, typically meters).
        freq
            ACCESS-OM2 intake frequency (e.g., '1mon', 'fx').
        chunks
            Passed to xarray open_mfdataset/open_dataset. Use None for eager arrays.
            Common choices: "auto", {}, {"time": 12}.
        parallel
            Passed to open_mfdataset.
        decode_timedelta
            Passed to xarray open_* calls.

        Returns
        -------
        xarray.DataArray
        """
        years = [str(y) for y in range(int(start_year), int(end_year) + 1)]
        fpaths = self._get_filepaths(src=src, expt=expt, var=var, years=years, freq=freq)

        if not fpaths:
            raise FileNotFoundError(
                f"No files found for src={src}, var={var}, years={start_year}-{end_year}, freq={freq}"
            )

        # Time-fixed (fx) variables: open single dataset and slice.
        if src == "ACCESS-OM2" and freq == "fx":
            with xr.open_dataset(fpaths[0], decode_timedelta=decode_timedelta, chunks=chunks) as ds:
                da = ds[var]
            return self._slice_generic(da, latmin=latmin, latmax=latmax, zmin=zmin, zmax=zmax)

        # ORAS5 needs special handling
        if src == "ORAS5":
            preprocess = self._preprocess_oras5(var=var, latmin=latmin, latmax=latmax, zmin=zmin, zmax=zmax)
            ds = xr.open_mfdataset(
                fpaths,
                preprocess=preprocess,
                decode_timedelta=decode_timedelta,
                chunks=chunks,
                parallel=parallel,
                combine="by_coords",
            )
            return ds[var]

        # Generic multi-file case (EN4, ACCESS-OM2 non-fx, etc.)
        dims = self._infer_var_dims(fpaths[0], var=var, decode_timedelta=decode_timedelta)
        preprocess = self._preprocess_generic(
            var=var,
            dims=dims,
            latmin=latmin,
            latmax=latmax,
            zmin=zmin,
            zmax=zmax,
        )
        ds = xr.open_mfdataset(
            fpaths,
            preprocess=preprocess,
            decode_timedelta=decode_timedelta,
            chunks=chunks,
            parallel=parallel,
            combine="by_coords",
        )
        return ds[var]

    # -----------------------
    # internal helpers
    # -----------------------

    def _get_filepaths(self, *, src: Source, expt: str, var: str, years: list[str], freq: str) -> list[str]:
        if src == "ACCESS-OM2":
            try:
                import intake  # noqa: F401
            except ImportError as e:
                raise ImportError("ACCESS-OM2 requires `intake` and the access-nri catalog.") from e

            catalog = intake.cat.access_nri.search(model=src, variable=var, frequency=freq)
            pattern = catalog[expt].search(variable=var).df["path"].tolist()

            if freq == "fx":
                return [pattern[0]]

            return sorted([p for p in pattern if any(y in p for y in years)])

        # Non-intake sources: local filesystem patterns
        base = Path(self.base_dir) / src

        if src == "EN4":
            pattern = str(base / "EN.4.2.2.?.analysis.l09.*.nc")
        elif src == "ORAS5":
            # Your original layout: /g/data/gv90/wrh581/ORAS5/<var>/ORAS5_<var>_monthly_SOcean_*.nc
            pattern = str((base / var) / f"ORAS5_{var}_monthly_SOcean_*.nc")
        else:
            raise ValueError(f"Unsupported src={src}")

        return sorted([f for f in glob.glob(pattern) if any(y in f for y in years)])

    def _infer_var_dims(self, path: str, *, var: str, decode_timedelta: bool) -> tuple[str, ...]:
        with xr.open_dataset(path, decode_timedelta=decode_timedelta) as ds0:
            if var not in ds0:
                raise KeyError(f"Variable {var!r} not found in {path}")
            return ds0[var].dims

    def _preprocess_generic(
        self,
        *,
        var: str,
        dims: tuple[str, ...],
        latmin: float,
        latmax: float,
        zmin: float,
        zmax: float,
    ) -> Callable[[xr.Dataset], xr.Dataset]:
        # Uses positional assumptions like your notebook, but avoids re-opening files repeatedly.
        # dims could be e.g. (time, depth, lat, lon) or (time, lat, lon) etc.
        if len(dims) == 4:
            zdim, latdim = dims[1], dims[2]
            space_range = {zdim: slice(zmin, zmax), latdim: slice(latmin, latmax)}
        elif len(dims) == 3:
            latdim = dims[1]
            space_range = {latdim: slice(latmin, latmax)}
        else:
            # Fallback: do not slice if unexpected
            space_range = {}

        def _sel(ds: xr.Dataset) -> xr.Dataset:
            if space_range:
                ds = ds.sel(**space_range)
            return ds

        return _sel

    def _preprocess_oras5(
        self,
        *,
        var: str,
        latmin: float,
        latmax: float,
        zmin: float,
        zmax: float,
    ) -> Callable[[xr.Dataset], xr.Dataset]:
        def _sel(ds: xr.Dataset) -> xr.Dataset:
            if var not in ds:
                raise KeyError(f"Variable {var!r} not found in ORAS5 dataset.")

            da = ds[var].isel(x=slice(0, 1440))  # trim wraparound points (your original intent)

            # vertical selection (ORAS5 typically uses 'deptht')
            if "deptht" in da.dims:
                da = da.sel(deptht=slice(zmin, zmax))

            # meridional selection: ORAS has 2D nav_lat/nav_lon coords
            if "nav_lat" in da.coords:
                lat = da["nav_lat"]
                da = da.where((lat >= latmin) & (lat <= latmax), drop=True)

            return da.to_dataset(name=var)

        return _sel

    def _slice_generic(
        self,
        da: xr.DataArray,
        *,
        latmin: float,
        latmax: float,
        zmin: float,
        zmax: float,
    ) -> xr.DataArray:
        # For fx or already-opened arrays: attempt a simple slice using first dims.
        dims = da.dims
        if len(dims) == 3:
            return da.sel({dims[0]: slice(zmin, zmax), dims[1]: slice(latmin, latmax)})
        if len(dims) == 2:
            return da.sel({dims[0]: slice(latmin, latmax)})
        return da
