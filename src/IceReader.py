from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import re
import glob
import os
import numpy as np
import xarray as xr


Source = Literal["ACCESS-OM2", "EN4", "ORAS5"]


@dataclass(frozen=True)
class IceReader:
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
            ysl = self._oras5_yslice(fpaths[0], latmin, latmax)

            def preprocess(ds):
                da = ds[var].isel(x=slice(0, 1440), y=ysl)
                if "deptht" in da.dims:
                    da = da.sel(deptht=slice(zmin, zmax))
                return da.to_dataset(name=var)

            ds = xr.open_mfdataset(
                fpaths,
                preprocess=preprocess,
                decode_timedelta=decode_timedelta,
                chunks=chunks,
                parallel=parallel,
                combine="by_coords",
                engine="netcdf4",
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
    
    def _oras5_yslice(self, sample_path: str, latmin: float, latmax: float) -> slice:
        import numpy as np
        import xarray as xr

        with xr.open_dataset(sample_path, decode_timedelta=False, engine="netcdf4") as ds0:
            # nav_lat is (y, x) -> make a 1D representative transect
            lat1d = ds0["nav_lat"].isel(x=0).values  # shape (y,)
        jj = np.where((lat1d >= latmin) & (lat1d <= latmax))[0]
        if jj.size == 0:
            raise ValueError(f"No ORAS5 y indices found for lat range [{latmin}, {latmax}]")
        return slice(int(jj.min()), int(jj.max()) + 1)

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

    # ---------- AWI L2/L3 helpers ----------

    @staticmethod
    def _awi_release_dir(level: int) -> str:
        if level == 3:
            return "l3cp_release"
        if level == 2:
            return "l2p_release"
        raise ValueError("level must be 2 or 3")

    @staticmethod
    def _awi_datekey_from_fname(fname: str) -> str:
        """
        Extract YYYYMM or YYYYMMDD token from filenames like:
        ...-201011-fv4p0.nc  or similar.
        Returns a sortable string; falls back to fname if not found.
        """
        m = re.search(r"-(\d{6,8})-", Path(fname).name)
        return m.group(1) if m else Path(fname).name

    def _get_awi_filepaths(
        self,
        *,
        awi_root: str,
        level: int,
        hemisphere: str,
        platform: str,
        start_year: int,
        end_year: int,
        pattern: str = "*.nc",
    ) -> list[str]:
        base = Path(awi_root) / self._awi_release_dir(level) / hemisphere / platform
        fpaths: list[str] = []
        for yr in range(int(start_year), int(end_year) + 1):
            ydir = base / str(yr)
            fpaths.extend(sorted(map(str, ydir.glob(pattern))))
        # sort by date token in filename (YYYYMM / YYYYMMDD)
        fpaths = sorted(fpaths, key=self._awi_datekey_from_fname)
        return fpaths

    # ---------- Public method ----------

    def read_awi(
        self,
        *,
        var: str | Sequence[str] = "sea_ice_thickness",
        start_year: int,
        end_year: int,
        level: int = 3,
        hemisphere: str = "sh",
        platform: str = "cryosat2",
        awi_root: str = "/g/data/gv90/da1339/SeaIce/AWI",
        chunks: dict | str | None = None,
        parallel: bool = False,
        engine: str = "netcdf4",
        decode_timedelta: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """
        Read AWI ESA CCI sea-ice thickness products (default: L3CP).

        Parameters
        ----------
        var
            A variable name (returns DataArray) or list/tuple of names (returns Dataset).
            Common: "sea_ice_thickness", "snow_depth", "sea_ice_freeboard", etc.
        start_year, end_year
            Inclusive year range.
        level
            3 (default) for l3cp_release, or 2 for l2p_release.
        hemisphere
            "sh" or "nh".
        platform
            e.g. "cryosat2", "envisat", "sentinel3a", "sentinel3b".
        awi_root
            Root directory containing l2p_release/ and l3cp_release/.
        chunks
            Use None for eager open; "auto" or dict for dask arrays.
        parallel
            Keep False by default on Gadi to reduce netCDF/HDF contention.
        engine
            "netcdf4" or "h5netcdf" if you hit HDF errors.
        """

        fpaths = self._get_awi_filepaths(
            awi_root=awi_root,
            level=level,
            hemisphere=hemisphere,
            platform=platform,
            start_year=start_year,
            end_year=end_year,
        )
        if not fpaths:
            raise FileNotFoundError(
                f"No AWI files found: root={awi_root}, level={level}, hemi={hemisphere}, "
                f"platform={platform}, years={start_year}-{end_year}"
            )

        # normalise var argument
        if isinstance(var, str):
            want_vars = [var]
            return_dataarray = True
        else:
            want_vars = list(var)
            return_dataarray = False

        def _pre(ds: xr.Dataset) -> xr.Dataset:
            # ensure lat/lon are treated as coordinates for convenience
            for c in ("lat", "lon", "xc", "yc"):
                if c in ds:
                    ds = ds.set_coords(c)

            # keep only requested vars (plus coords that may be stored as variables)
            keep = [v for v in want_vars if v in ds.data_vars]
            if not keep:
                raise KeyError(f"Requested var(s) {want_vars} not found. Available: {list(ds.data_vars)}")

            return ds[keep]

        ds = xr.open_mfdataset(
            fpaths,
            preprocess=_pre,
            combine="by_coords",         # each file has time=1; by_coords is fine
            decode_timedelta=decode_timedelta,
            chunks=chunks,
            parallel=parallel,
            engine=engine,
        )

        if return_dataarray:
            return ds[want_vars[0]]
        return ds
