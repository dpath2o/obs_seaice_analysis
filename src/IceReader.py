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

    def list_awi_platforms(
            self,
            *,
            hemisphere: str = "sh",
            level: int = 3,
            awi_root: str = "/g/data/gv90/da1339/SeaIce/AWI",
        ) -> list[str]:
            """Return platforms available on disk for a given hemisphere + level."""
            base = Path(awi_root) / self._awi_release_dir(level) / hemisphere
            if not base.exists():
                return []
            return sorted([p.name for p in base.iterdir() if p.is_dir()])

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
        platform: str | None = "cryosat2",
        platforms: Sequence[str] | None = None,
        awi_root: str = "/g/data/gv90/da1339/SeaIce/AWI",
        chunks: dict | str | None = None,
        parallel: bool = False,
        engine: str = "netcdf4",
        decode_timedelta: bool = False,
        stack_platform: bool | None = None,
        strict_platforms: bool = False,
        collapse_platforms: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """
        Platform-agnostic AWI reader.

        Key behaviours
        --------------
        - platforms=None and platform="cryosat2" => single platform (backwards compatible).
        - platform="all" (or platform=None) => auto-discover all platforms on disk.
        - platforms=[...] => explicit multi-platform request.
        - If multiple platforms:
            * default is to stack along a new 'platform' dimension (platform, time, yc, xc)
            * optionally collapse into one product via combine_first (priority order)
        """

        # ---- normalise requested platform list ----
        if platforms is not None:
            plat_list = list(platforms)
        else:
            if platform is None or str(platform).lower() == "all":
                plat_list = self.list_awi_platforms(hemisphere=hemisphere, level=level, awi_root=awi_root)
            else:
                plat_list = [platform]

        if not plat_list:
            raise FileNotFoundError(
                f"No AWI platforms found: root={awi_root}, level={level}, hemisphere={hemisphere}"
            )

        # If more than one platform and user didn't specify, default to stacking
        if stack_platform is None:
            stack_platform = (len(plat_list) > 1)

        # ---- normalise var argument ----
        if isinstance(var, str):
            want_vars = [var]
            return_dataarray = True
        else:
            want_vars = list(var)
            return_dataarray = False

        def _pre(ds: xr.Dataset) -> xr.Dataset:
            # Promote common coords for convenience
            for c in ("lat", "lon", "xc", "yc"):
                if c in ds:
                    ds = ds.set_coords(c)

            keep = [v for v in want_vars if v in ds.data_vars]
            if not keep:
                raise KeyError(f"Requested var(s) {want_vars} not found. Available: {list(ds.data_vars)}")

            return ds[keep]

        # ---- open per-platform to keep bookkeeping clean ----
        per_platform: list[xr.Dataset] = []
        missing: list[str] = []

        for plat in plat_list:
            fpaths = self._get_awi_filepaths(
                awi_root=awi_root,
                level=level,
                hemisphere=hemisphere,
                platform=plat,
                start_year=start_year,
                end_year=end_year,
            )
            if not fpaths:
                missing.append(plat)
                continue

            ds_plat = xr.open_mfdataset(
                fpaths,
                preprocess=_pre,
                combine="by_coords",
                decode_timedelta=decode_timedelta,
                chunks=chunks,
                parallel=parallel,
                engine=engine,
            )

            # annotate platform (only if stacking/collapsing makes sense)
            ds_plat = ds_plat.assign_coords(platform=plat).expand_dims(platform=[plat])
            per_platform.append(ds_plat)

        if missing and strict_platforms:
            raise FileNotFoundError(
                f"No files found for platforms={missing} (strict_platforms=True). "
                f"Requested platforms={plat_list}, years={start_year}-{end_year}."
            )

        if not per_platform:
            raise FileNotFoundError(
                f"No AWI files found for any requested platform. platforms={plat_list}, years={start_year}-{end_year}"
            )

        # ---- combine platforms ----
        if len(per_platform) == 1:
            ds = per_platform[0]
            # drop platform dim for backwards compatibility if user did not ask for stacking
            if not stack_platform:
                ds = ds.isel(platform=0, drop=True)
        else:
            if collapse_platforms:
                # Priority order is plat_list (the order user provided or discovered)
                ds = per_platform[0].isel(platform=0, drop=True)
                for nxt in per_platform[1:]:
                    ds = ds.combine_first(nxt.isel(platform=0, drop=True))
            else:
                # Stack: output dims include platform and preserve overlaps
                ds = xr.concat(per_platform, dim="platform", join="outer")

        # ---- return type ----
        if return_dataarray:
            return ds[want_vars[0]]
        return ds

