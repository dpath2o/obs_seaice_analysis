from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import re
import glob
import os
import numpy as np
import xarray as xr

Source = Literal["ACCESS-OM2", "ACCESS-OM3", "EN4", "ORAS5"]

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

    def read_model(self, *,
                   src              : Source,
                   expt             : str = "ORAS5",
                   var              : str,
                   start_year       : int,
                   end_year         : int,
                   latmin           : float = -90.0,
                   latmax           : float = 90.0,
                   zmin             : float = 0.0,
                   zmax             : float = 6000.0,
                   freq             : str = "1mon",
                   chunks           : dict | int | str | None = None,
                   parallel         : bool = True,
                   decode_timedelta : bool = False) -> xr.DataArray:
        """
        Open and subset gridded ocean/reanalysis/model fields (e.g., ORAS5, EN4, ACCESS-OM2)
        and return a single variable as an :class:`xarray.DataArray`.

        This method standardises:
        (i) file discovery for supported sources,
        (ii) efficient opening of multi-file NetCDF collections (optionally Dask-chunked),
        and (iii) light-weight spatial subsetting prior to analysis/plotting.

        Parameters
        ----------
        src : {"ACCESS-OM2", "ACCESS-OM3", "EN4", "ORAS5"}
            Data source identifier.

            - **ACCESS-OM2**: paths are discovered via the `access-nri` intake catalog.
            - **ORAS5**: paths are discovered from a local filesystem layout under
              ``{base_dir}/ORAS5/<var>/ORAS5_<var>_monthly_SOcean_*.nc``.
            - **EN4**: paths are discovered from a local filesystem layout under
              ``{base_dir}/EN4/EN.4.2.2.?.analysis.l09.*.nc``.
            - **ACCESS-OM3**: reserved for future implementation (include only when supported).

        expt : str, default="ORAS5"
            Experiment/collection key. Used only for ACCESS-OM2 (intake catalog lookup).
            Ignored for EN4/ORAS5 filesystem-based reading.

            Examples (ACCESS-OM2): ``expt="01deg_jra55_iaf"`` or similar, depending on catalog.

        var : str
            Variable name to extract from the datasets.

            Examples:
            - ORAS5: ``"thetao"``, ``"vosaline"``
            - EN4: typical temperature/salinity variables depending on EN4 file conventions
            - ACCESS-OM2: depends on catalog variable naming

        start_year, end_year : int
            Inclusive year range. Files are filtered by matching the year token in the path/filename.

        latmin, latmax : float, default=(-90.0, 90.0)
            Latitude bounds for subsetting.

            Notes:
            - For **ORAS5**, latitude is a 2D curvilinear coordinate (``nav_lat``). To remain Dask-safe,
              the method uses an **index-based y-slice** computed from a representative transect.
            - For **regular 1D latitude grids**, the method uses ``.sel(lat=slice(...))``-style slicing
              based on inferred dimension order.

        zmin, zmax : float, default=(0.0, 6000.0)
            Vertical bounds (typically meters). The relevant depth dimension depends on source.

            Notes:
            - ORAS5 commonly uses ``deptht``.
            - Other sources may use different dimension names; this method uses positional
              assumptions for generic cases.

        freq : str, default="1mon"
            Frequency specifier for ACCESS-OM2 intake search (e.g., ``"1mon"``, ``"day"``, ``"fx"``).

            Special case:
            - If ``src="ACCESS-OM2"`` and ``freq="fx"``, a single time-invariant file is opened with
              :func:`xarray.open_dataset` and then subset.

        chunks : dict | int | str | None, default=None
            Chunking argument passed to xarray open routines. Use to enable Dask-backed arrays.

            Typical values:
            - ``None``: eager arrays (simplest; best for small subsets)
            - ``"auto"``: let xarray/dask choose chunks
            - ``{ "time": 12 }``: explicit chunking

            Practical guidance (Gadi/shared filesystems):
            - Use ``chunks="auto"`` for month-by-month workflows and plotting.
            - Consider ``parallel=False`` when opening many NetCDF files to reduce HDF5/netCDF contention.

        parallel : bool, default=True
            Passed to :func:`xarray.open_mfdataset`.

            Practical guidance:
            - On shared HPC filesystems, parallel I/O can trigger intermittent netCDF/HDF errors.
              If you see ``RuntimeError: NetCDF: HDF error``, retry with ``parallel=False`` and/or
              ``chunks=None`` (eager open).

        decode_timedelta : bool, default=False
            Passed to xarray open routines. Useful when datasets contain timedelta-like variables
            that can be problematic to decode.

        Returns
        -------
        xarray.DataArray
            The requested variable as a DataArray, typically with dimensions including
            ``time`` and spatial dimensions (plus depth where applicable).

        Raises
        ------
        FileNotFoundError
            If no files matching the requested source/years/variable are found.
        KeyError
            If the requested variable is not present in the dataset(s).
        ImportError
            If ``src="ACCESS-OM2"`` but `intake`/catalog dependencies are not available.
        ValueError
            For invalid source identifiers or invalid latitude selection (e.g., ORAS5 y-slice not found).

        Notes
        -----
        ORAS5-specific behaviour
        - Files may include wraparound longitudes; this reader trims the x-dimension to 1440 points.
        - Latitude is curvilinear (2D). To remain Dask-safe (avoid boolean indexing with unknown shape),
          the reader computes a fixed **y index slice** once, then applies ``isel(y=slice(...))`` to every file.

        Examples
        --------
        ORAS5 monthly salinity (Southern Ocean subset):

        >>> r = IceReader(base_dir="/g/data/gv90/wrh581")
        >>> da = r.read_model(
        ...     src="ORAS5",
        ...     var="vosaline",
        ...     start_year=2006,
        ...     end_year=2007,
        ...     latmin=-80,
        ...     latmax=-45,
        ...     zmin=0,
        ...     zmax=1000,
        ...     chunks="auto",
        ...     parallel=False,
        ... )
        >>> da

        ACCESS-OM2 fixed field (example pattern; catalog-dependent):

        >>> da = r.read_model(
        ...     src="ACCESS-OM2",
        ...     expt="01deg_jra55_iaf",
        ...     var="geolon_t",
        ...     start_year=2000,
        ...     end_year=2000,
        ...     freq="fx",
        ... )
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
    # internal helpers for read_model()
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

    def _preprocess_generic(self, *,
                            var   : str,
                            dims  : tuple[str, ...],
                            latmin: float,
                            latmax: float,
                            zmin  : float,
                            zmax  : float,) -> Callable[[xr.Dataset], xr.Dataset]:
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

    def _preprocess_oras5(self, *,
                          var   : str,
                          latmin: float,
                          latmax: float,
                          zmin  : float,
                          zmax  : float) -> Callable[[xr.Dataset], xr.Dataset]:
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

    def _slice_generic(self,
                       da: xr.DataArray,
                       *,
                       latmin: float,
                       latmax: float,
                       zmin  : float,
                       zmax  : float) -> xr.DataArray:
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

    def _get_awi_filepaths(self, *,
                           awi_root   : str,
                           level      : int,
                           hemisphere : str,
                           platform   : str,
                           start_year : int,
                           end_year   : int,
                           pattern    : str = "*.nc") -> list[str]:
        base = Path(awi_root) / self._awi_release_dir(level) / hemisphere / platform
        fpaths: list[str] = []
        for yr in range(int(start_year), int(end_year) + 1):
            ydir = base / str(yr)
            fpaths.extend(sorted(map(str, ydir.glob(pattern))))
        # sort by date token in filename (YYYYMM / YYYYMMDD)
        fpaths = sorted(fpaths, key=self._awi_datekey_from_fname)
        return fpaths

    def list_awi_platforms(self, *,
                           hemisphere : str = "sh",
                           level      : int = 3,
                           awi_root   : str = "/g/data/gv90/da1339/SeaIce/AWI") -> list[str]:
            """Return platforms available on disk for a given hemisphere + level."""
            base = Path(awi_root) / self._awi_release_dir(level) / hemisphere
            if not base.exists():
                return []
            return sorted([p.name for p in base.iterdir() if p.is_dir()])

    # ---------- Public method ----------
    def read_awi(self, *,
                 var                : str | Sequence[str] = "sea_ice_thickness",
                 start_year         : int,
                 end_year           : int,
                 level              : int = 3,
                 hemisphere         : str = "sh",
                 platform           : str | None = "cryosat2",
                 platforms          : Sequence[str] | None = None,
                 awi_root           : str = "/g/data/gv90/da1339/SeaIce/AWI",
                 chunks             : dict | str | None = None,
                 parallel           : bool = False,
                 engine             : str = "netcdf4",
                 decode_timedelta   : bool = False,
                 stack_platform     : bool | None = None,
                 strict_platforms   : bool = False,
                 collapse_platforms : bool = False) -> xr.DataArray | xr.Dataset:
        """
        Read AWI ESA CCI sea-ice thickness products (L3CP by default; L2P supported by layout).

        The AWI products are stored in a structured directory tree by hemisphere, platform,
        and year. This method provides:

        - file discovery by year range (inclusive),
        - platform-specific reading (single platform),
        - platform-agnostic reading (auto-discover all platforms or user-provided list),
        - optional stacking across platforms (preserve overlaps) or collapsing to a best-available product.

        Parameters
        ----------
        var : str or Sequence[str], default="sea_ice_thickness"
            Variable(s) to load.

            - If a single string is provided, returns an :class:`xarray.DataArray`.
            - If a list/tuple is provided, returns an :class:`xarray.Dataset` containing those variables.

            Common L3CP variables include:
            ``sea_ice_thickness``, ``sea_ice_thickness_uncertainty``,
            ``snow_depth``, ``sea_ice_freeboard``, ``radar_freeboard``,
            ``status_flag``, ``quality_flag``, ``region_code``.

        start_year, end_year : int
            Inclusive year range. Files are pulled from yearly subdirectories and then sorted by
            a YYYYMM / YYYYMMDD token extracted from filenames.

        level : int, default=3
            Product level selector.
            - ``3``: uses ``l3cp_release`` (default; gridded monthly products)
            - ``2``: uses ``l2p_release`` (present on disk; variable naming/time cadence may differ)

        hemisphere : {"sh", "nh"}, default="sh"
            Hemisphere selector for directory traversal.

        platform : str or None, default="cryosat2"
            Single-platform request. Typical values on disk include:
            ``"cryosat2"``, ``"envisat"``, ``"sentinel3a"``, ``"sentinel3b"``.

            Special behaviour:
            - If ``platform="all"`` (case-insensitive) or ``platform=None`` and ``platforms`` is not set,
              the reader will auto-discover all platform subdirectories under the requested
              ``(awi_root, level, hemisphere)``.

        platforms : Sequence[str] or None, default=None
            Explicit multi-platform request (overrides `platform` if provided). The order of this list
            becomes the platform priority order if ``collapse_platforms=True``.

        awi_root : str, default="/g/data/gv90/da1339/SeaIce/AWI"
            Root directory containing ``l2p_release/`` and ``l3cp_release/``.

            Expected L3CP layout:

            ``{awi_root}/l3cp_release/{hemisphere}/{platform}/{YYYY}/*.nc``

        chunks : dict | str | None, default=None
            Chunking argument passed to :func:`xarray.open_mfdataset`.

            Practical guidance:
            - For monthly products, ``chunks="auto"`` typically yields one time-step per chunk.
            - For stability during initial development/testing, use ``chunks=None`` (eager open).

        parallel : bool, default=False
            Passed to :func:`xarray.open_mfdataset`. Default is conservative to reduce I/O instability
            on shared filesystems.

        engine : {"netcdf4", "h5netcdf"}, default="netcdf4"
            Backend engine for reading NetCDF files.

            Practical guidance:
            - If you encounter ``RuntimeError: NetCDF: HDF error``, retry with:
              ``engine="h5netcdf"`` and/or ``parallel=False`` and/or ``chunks=None``.

        decode_timedelta : bool, default=False
            Passed to xarray open routines.

        stack_platform : bool or None, default=None
            Controls whether multiple platforms are preserved as a separate dimension.

            - If ``None`` (default): automatically stacks when more than one platform is requested.
            - If ``True``: ensures output includes dimension ``platform``.
            - If ``False``: for single-platform reads, drops the singleton platform dimension.

        strict_platforms : bool, default=False
            If True, raises an error when any requested platform has no matching files for the year range.
            If False, missing platforms are silently skipped.

        collapse_platforms : bool, default=False
            If True and multiple platforms are opened, collapses them into a single Dataset/DataArray by
            filling missing values in priority order (first platform wins where it has data; gaps filled
            by subsequent platforms). Implementation uses :meth:`xarray.Dataset.combine_first`.

            If False (default), platforms are kept separate via concatenation along ``platform``.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            - DataArray if `var` is a string
            - Dataset if `var` is a sequence of strings

            Typical dimensions for L3CP:
            - Stacked: ``(platform, time, yc, xc)``
            - Collapsed: ``(time, yc, xc)``

            Coordinates:
            The reader promotes common variables to coordinates where present, e.g. ``lat``, ``lon``,
            ``xc``, ``yc``.

        Raises
        ------
        FileNotFoundError
            If no platforms are found (platform="all") or if no files are found for the selected
            platform(s) and year range.
        KeyError
            If the requested variable(s) are not present in the files.
        ValueError
            For invalid `level` values (must be 2 or 3).

        Notes
        -----
        Platform-agnostic workflows
        - Use ``platform="all"`` to take all platforms present on disk for a given hemisphere/level.
        - Use ``platforms=[...]`` for explicit platform choice and ordering.
        - Use ``collapse_platforms=True`` when you want a single “best-available” time series for plotting.

        Quality control
        - For meeting-ready plots, it is common to apply conservative filters using ``status_flag`` and
          ``quality_flag`` after loading (these fields are included in the product files).

        Examples
        --------
        Single platform SIT:

        >>> r = IceReader()
        >>> sit = r.read_awi(
        ...     var="sea_ice_thickness",
        ...     start_year=2010,
        ...     end_year=2012,
        ...     hemisphere="sh",
        ...     platform="cryosat2",
        ...     chunks="auto",
        ... )
        >>> sit

        Load all platforms and keep them separate:

        >>> sit_all = r.read_awi(
        ...     var="sea_ice_thickness",
        ...     start_year=2010,
        ...     end_year=2012,
        ...     hemisphere="sh",
        ...     platform="all",
        ...     chunks="auto",
        ... )
        >>> sit_all.dims
        ('platform', 'time', 'yc', 'xc')

        Best-available collapsed product (priority order):

        >>> sit_best = r.read_awi(
        ...     var="sea_ice_thickness",
        ...     start_year=2010,
        ...     end_year=2012,
        ...     hemisphere="sh",
        ...     platforms=["cryosat2", "envisat", "sentinel3a", "sentinel3b"],
        ...     collapse_platforms=True,
        ...     chunks="auto",
        ... )
        >>> sit_best.dims
        ('time', 'yc', 'xc')

        Load SIT + QA fields and apply a conservative mask:

        >>> ds = r.read_awi(
        ...     var=["sea_ice_thickness", "status_flag", "quality_flag"],
        ...     start_year=2010,
        ...     end_year=2012,
        ...     hemisphere="sh",
        ...     platform="all",
        ...     chunks="auto",
        ... )
        >>> sit_qc = ds["sea_ice_thickness"].where(ds["status_flag"] == 0)
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

