# pysamosa

A Python toolkit for processing, merging, and modeling multi-source air quality datasets — primarily low-cost sensor networks (PurpleAir), regulatory monitors (CPCB), and satellite retrievals (TROPOMI, Martin AOD-PM), with a spatiotemporal decomposition layer.

## Package Structure

```
pysamosa/
├── data_source/          # Format raw data from each source into xarray Datasets
│   ├── bam.py            # MetOne BAM-1022 (reference PM2.5)
│   ├── era.py            # ERA5 reanalysis (wind, temperature, RH)
│   ├── gee.py            # Google Earth Engine satellite products
│   ├── ghsl.py           # GHSL population density
│   ├── martin.py         # WUSTL Randall Martin AOD-PM inversion
│   ├── pa.py             # PurpleAir (Earthmetry platform)
│   ├── pr.py             # PurpleAir (PurpleAir platform)
│   ├── quantaq.py        # QuantAQ sensors
│   ├── reg.py            # CPCB regulatory sites
│   ├── rwi.py            # Relative Wealth Index
│   └── tropomi.py        # TROPOMI NO2 column density
├── data_calculations/    # QA/QC, calibration, merging, and statistical utilities
│   ├── bootstrap.py      # Bootstrap sampling (Numba JIT)
│   ├── calibrate.py      # PurpleAir sensor calibration models
│   ├── flag.py           # Quality assurance flags per data source
│   ├── india.py          # India-specific helpers (AQI, seasons, wind)
│   ├── merge.py          # Spatial/temporal dataset merging
│   ├── metrics.py        # Evaluation metrics (RMSE, MAE, MBE, R²)
│   ├── peak_factor.py    # Baseline/peak decomposition
│   └── trim.py           # Time trimming and geospatial clipping
└── spatial_temporal_model/   # Spatiotemporal decomposition and reconstruction
    ├── car_bayes.py      # Conditional autoregressive Bayesian model
    ├── gpr.py            # Gaussian process regression
    ├── memd.py           # Multivariate empirical mode decomposition
    ├── model.py          # SpatialTemporalModel class
    ├── pod.py            # Proper orthogonal decomposition (GappyPOD, SpatialPOD)
    ├── qr_pivot.py       # QR pivot decomposition
    ├── reprojection.py   # Spatial reprojection matrix computation
    └── sampling.py       # Sampling matrix construction
```

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

```python
from pysamosa.data_source.bam import format_bam
from pysamosa.data_source.pa import format_pa
from pysamosa.data_calculations.flag import qa_bam, qa_pa
from pysamosa.data_calculations.merge import merge_reference

# Format raw data
ds_bam = format_bam("/path/to/bam/data")
ds_pa = format_pa("/path/to/pa/data")

# Quality-assurance
ds_bam = qa_bam(ds_bam)
ds_pa = qa_pa(ds_pa)

# Merge into a single reference dataset
ds_ref = merge_reference("/path/to/flagged/data")
```

## License

MIT
