# obs_seaice_analysis

A lightweight Python module for reading, analysing, and presenting **sea-ice observational products** (and closely related ocean/reanalysis fields) in a format suitable for **monthly technical meetings** and discussion among sea-ice scientists.

This repository is being built by consolidating an existing, working collection of analysis scripts (see [https://github.com/willrhobbs/Obs-seaice-analysis]) into a maintainable Python package. The initial backbone is the `IceReader` class, which standardises file discovery and loading across multiple products and sources.

## Philosophy

- **Meeting-ready by default**: quick loading of common products and rapid generation of monthly figures/tables.
- **Reproducible and modular**: scripts become importable functions/classes with stable APIs.
- **Pragmatic HPC-first**     : designed to work on NCI/Gadi-style filesystems (large NetCDF collections, Dask chunking, conservative parallel I/O).

## Current status

Implemented (and used in notebooks):
- `IceReader.read_model(...)` for model/reanalysis-style NetCDF collections (e.g., ORAS5; optional ACCESS-OM2 via intake).
- `IceReader.read_awi(...)` for AWI ESA CCI sea-ice thickness products:
  - **L3CP** (default) and scaffolding for **L2P**
  - platform-aware and platform-agnostic access (`platform="all"` / `platforms=[...]`)

In progress:
- Porting the legacy scripts/notebooks into cohesive submodules (readers, diagnostics, plotting, workflow scripts).

## Repository layout

obs_seaice_analysis/
├── init.py # exports IceReader
├── src/
│ ├── init.py
│ └── IceReader.py # core reader class (model + satellite products)
└── notebooks/ # working notebooks / examples (WIP)

## Getting started

### 1. Clone the repository

From a terminal:

```bash
git clone https://github.com/dpath2o/obs_seaice_analysis.git
cd obs_seaice_analysis
```

#### 1a. If you plan to contribute: create a branch

If you are working in the same remote repository:
```bash
git checkout -b feature/<short-description>
```

#### 1b. If you are contributing via a fork, fork on GitHub first, then:
```bash
git clone <your-fork-url>
cd obs_seaice_analysis
git remote add upstream https://github.com/dpath2o/obs_seaice_analysis.git
git fetch upstream
git checkout -b feature/<short-description>
```

### 2. Use the package in a notebook workspace

This repository is currently used as an importable source tree (no packaging metadata required).

In your notebook (or any Python session), add the directory containing `obs_seaice_analysis/` to  `sys.path`
```python
import sys
from pathlib import Path

# Example: if you cloned into /home/581/<user>/AFIM/src/obs_seaice_analysis
repo_parent = Path("/home/581/da1339/AFIM/src")  # adjust for your location
sys.path.insert(0, str(repo_parent))

from obs_seaice_analysis import IceReader
print(obs_seaice_analysis.__file__)
```
### 3. Load data with IceReader

#### 3.1 ORAS5 example (monthly Southern Ocean salinity)
```python
r = IceReader(base_dir="/g/data/gv90/wrh581")
vosaline = r.read_model(src        = "ORAS5",
                        var        = "vosaline",
                        start_year = 2006,
                        end_year   = 2007,
                        latmin     = -80,
                        latmax     = -45,
                        zmin       = 0,
                        zmax       = 1000,
                        chunks     = "auto",
                        parallel   = False)   # recommended on shared filesystems
```
Notes:
+ ORAS5 uses 2D nav_lat/nav_lon. The reader uses an index-based y slice to avoid boolean indexing with Dask.
+ If you encounter NetCDF: HDF error, try parallel=False, chunks=None, or (in code) switching the xarray engine to h5netcdf.

#### 3.2 AWI ESA CCI sea-ice thickness (L3CP) examples

##### Single platform (default):
```python
sit = r.read_awi(var        = "sea_ice_thickness",
                 start_year = 2010,
                 end_year   = 2012,
                 hemisphere = "sh",
                 platform   = "cryosat2",
                 chunks     = "auto")
```

##### Platform-agnostic (load all available platforms on disk and keep them separate):
```python
sit = r.read_awi(var        = "sea_ice_thickness",
                 start_year = 2010,
                 end_year   = 2012,
                 hemisphere = "sh",
                 platform   = "all",
                 chunks     = "auto")
```

##### Platform-agnostic with collapsing (best-available, priority order):
```python
sit = r.read_awi(var        = "sea_ice_thickness",
                 start_year = 2010,
                 end_year   = 2012,
                 hemisphere = "sh",
                 platforms  = ["cryosat2", "envisat", "sentinel3a", "sentinel3b"],
                 collapse_platforms = True,   # fill missing values by priority order
                 chunks     = "auto")
```

##### Load multiple variables across multiple platforms (SIT + QA fields):
```python 
sit = r.read_awi(var        = ["sea_ice_thickness", "status_flag", "quality_flag"],
                 start_year = 2010,
                 end_year   = 2012,
                 hemisphere = "sh",
                 platform   = "all",
                 chunks     = "auto")
```

### Monthly SIT maps (PyGMT)

AWI L3CP SIT is provided on a projected grid with coordinates xc/yc (km). A reliable PyGMT approach is to plot in Cartesian space using -JX:
```python
import numpy as np
from pathlib import Path
import pygmt

def plot_sit_monthly(sit, D_out,
                     platform_name = None,
                     region = None,
                     fig_size = "20c"
                     vmin   = 0.0,
                     vmax   = 5.0, 
                     cmap   = "cmocean/matter"):
    '''
    sit must be a 3D array (i.e. time, x, y) and be stripped of the platform dimension before providing to this function
    '''
    D_out  = Path(D_out); D_out.mkdir(parents=True, exist_ok=True)
    sit    = sit.sortby("yc")  # GMT expects ascending y
    region = region if region is not None else [float(sit.xc.min()), float(sit.xc.max()), float(sit.yc.min()), float(sit.yc.max())]
    for t in sit.time.values:
        tstr = np.datetime_as_string(t, unit="M")
        grid = sit.sel(time=t).load()  # compute one month at a time
        fig  = pygmt.Figure()
        pygmt.makecpt(cmap = cmap, series=[vmin, vmax])
        fig.basemap(region     = region,
                    projection = f"X{fig_size}/0",
                    frame      = [f'+tAWI SIT {platform_name} {tstr}', "xaf", "yaf"])
        fig.grdimage(grid = grid, cmap = True, nan_transparent = True)
        fig.colorbar(frame = ['x+lSea-ice thickness (m)'])
        fig.savefig(D_out / f"awi_sit_{platform_name}_{tstr}.png", dpi=300)
```

## Legacy scripts to be incorporated

The OBS-SEAICE-SCRIPTS.zip archive is treated as the authoritative reference for the workflows that will be migrated into this module. Contents currently include:

- NCL plotting/analysis scripts (*.ncl)
- Python readers/workflows (read_functions.py, read_ocean_data.py, SIT_obs_analysis.py)
- Jupyter notebooks (*.ipynb)

### Archive file list:
+ ARGO_ocean_lontime_hovmuller.ncl
+ ERA5_sfcflux_map.ncl
+ ncl_funcs.ncl
+ NSIDC_ice_area_write.ncl
+ NSIDC_ice_atmo_streamline_summary_map.ncl
+ NSIDC_ice_atmo_summary_map.ncl
+ NSIDC_ice_sst_summary_map.ncl
+ NSIDC_SIA_cycle_tplot.ncl
+ NSIDC_SIA_sector_tplot.ncl
+ NSIDC_sic_clim_map.ncl
+ NSIDC_sic_map_generic.ncl
+ NSIDC_totalSIA_anoms_byyear.ncl
+ NSIDC_totalSIA_SH-NH_compare.ncl
+ NSIDC_totalSIA_tplot.ncl
+ NSIDC_totalSIA_write.ncl
+ OBS_ocean_depthtime_hovmuller.ncl
+ plot_funcs.ncl
+ Reanalysis_sfcflux_map.ncl
+ SIA_anoms_tplot_bymonth.ncl
+ SST_arealave_tplot.ncl
+ SST_ice_timeseries.ncl
+ read_functions.py
+ read_ocean_data.py
+ SIT_obs_analysis.py
+ merid-wind-and-SIE_monthly.ipynb
+ merid-wind-and-SIE.ipynb
+ NSIDC_SIE_max_vs_maxdate.ipynb
+ Obs_depth_lat_trend.ipynb
+ OBS_ocean_depthtime_hovmuller.ipynb
+ OISST_global_anoms.ipynb
+ OSISAF_totalSIA_plot.ipynb

## Contributing

Open an issue describing the dataset/workflow you want to add (inputs, outputs, expected figures).

Implement as an importable function/class under obs_seaice_analysis/.

Add a small notebook in notebooks/ demonstrating usage with a minimal working example.

## Disclaimer

This repository is an evolving research toolset. Interfaces may change while the legacy scripts are being migrated into stable APIs.