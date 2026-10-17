# Konya air quality forecasting (konya_aqf)

Code and processed data for the manuscript "A Horizon-Aware and Leakage-Safe Machine Learning
Framework for Operational Air Quality Forecasting: The Case of Konya".
Archived release: https://doi.org/10.5281/zenodo.23426748

## Contents

- `src/`, `config.py`, `run.py`: forecasting framework (model pool, features, targets, cross-validated selection).
- `eswa_revision/`: scripts and result tables for the fold-wise preprocessing protocol reported in the manuscript.
- `data/konya_clean_2stations.csv`: processed hourly series of the two anchor stations (Karkent and Meram), 2017-01-01 to 2026-04-29 (81,721 rows). Pollutant values are in ug/m3; SO2 and CO of Meram were converted from ppb with ug/m3 = ppb x molar weight / 24.45 (25 C, 1 atm; molar weights 64.066 for SO2 and 28.010 for CO).
- `data/05_air_quality_7node_long.csv`: earlier long-format export of a seven-node subset, not used for the reported results.
- `outputs/`: cached meteorological features (ERA5 and archived day-ahead forecasts) and earlier result tables.

The raw hourly measurements belong to the national air quality monitoring network of the Republic of Turkiye Ministry of Environment, Urbanization and Climate Change and can be requested from that institution.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Do not install LightGBM or XGBoost. The code adds them to the model pool when they are present, and the reported results were obtained with scikit-learn HistGradientBoosting only.

## Reproducing the reported tables

Run every command from the repository root. Each experiment takes from a few seconds (weekly and monthly horizons) to several minutes (hourly and daily horizons) per target-horizon cell on one CPU core; the full baseline sweep takes roughly one to two hours on two cores.

```bash
export OMP_NUM_THREADS=1 PYTHONPATH=.:eswa_revision

# Main results without meteorology, all targets and horizons (writes outputs/proto2_baseline.csv)
python eswa_revision/proto2.py --scen baseline --workers 2

# Persistence, seasonal-naive and climatology references for all targets and horizons
python eswa_revision/refs2.py

# Meteorological comparison: ERA5 and operational day-ahead forecasts
python eswa_revision/proto2.py --scen era5 --workers 2
python eswa_revision/proto2.py --scen operational --workers 2

# Purge sensitivity: same protocol without purging (writes outputs/nopurge_*.csv)
python eswa_revision/proto2.py --scen baseline --workers 2 --no-purge

# Window-based baselines, anomalous-year and difference-parameterization ablation
python eswa_revision/baselines2.py
python eswa_revision/ablate2.py
```

Under this protocol, anomalous-year detection, clipping thresholds and gap-filling statistics are learned within the training period of each cross-validation fold (2021 to 2022, 2022 to 2023, 2023 to 2024) and, for the final model, from the years up to 2024; training rows whose forward-looking target window crosses the training boundary are purged. The test period is 2025 onward.

The stored processed series was clipped before it was deposited. The anomalous-year check can therefore be rerun, but the separate effect of the clipping threshold on an anomalous year cannot be reproduced from this file.

## Original pipeline

`python run.py` runs the earlier single-split pipeline (validation year 2024, test from 2025) and requires the raw xlsx file given by `DATA_PATH` in `config.py`.

## Final protocol and raw-target evaluation (eswa_revision)

The scripts in `eswa_revision/` reproduce the tables and figures of the submitted manuscript. Place the raw workbook `Konya_İstasyonları_Saatlik_Veriler__2017-2025_.xlsx` in `data/` so that the evaluation targets are the unclipped observations; models are fitted on series clipped at the training 99.5th percentile. Without the workbook, `proto2.py` falls back to `data/konya_clean_2stations.csv`, whose values are already clipped, and prints a warning.

Run order (see `eswa_revision/runall2.sh`): `proto2.py --scen baseline|era5|operational`, `refs2.py`, `alerts2.py`, `baselines2.py`, `ablate2.py`, `fill_audit.py`, `proto2.py --scen baseline --no-purge`, `nowcast2.py`, and finally `make_figs_final.py` for Figures 4 to 7. The result tables are stored as CSV files next to the scripts. The LSTM check reported in the manuscript was run under an earlier protocol and its code is not included.
