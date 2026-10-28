# Data

## `processed/flights_2022_sample.parquet` (committed, 3 MB)

A stratified sample of 200,000 completed US domestic flights from January–July
2022 (21.65% delayed), with the 16 pre-departure features used by the models,
plus `FlightDate`, `Month`, `DistanceGroup`, `Quarter` and the target `ArrDel15`.
It is everything `scripts/03_eda.py` to `scripts/07_compare_splits.py` and the
models need, so the project can be run without downloading the raw data.

It was produced by `scripts/02_prepare_data.py --year 2022` (seed 42) from the
raw file below.

## Raw data (not committed, ~150 MB per year)

Kaggle: [Flight Delay Dataset 2018–2022](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)
(file `Combined_Flights_2022.parquet`). Needed only for `scripts/01_inspect_dataset.py`,
`scripts/02_prepare_data.py` and Phases 1–2 of the notebook. Download it from the
Kaggle page, or with a Kaggle API token:

```bash
scripts/download_data.sh 2022
```

and either keep it in `data/raw/` or point `FLIGHT_DATA_DIR` at its folder.

## Source and licence

The records come from the US Bureau of Transportation Statistics
[Reporting Carrier On-Time Performance](https://www.transtats.bts.gov) data,
a work of the US Government in the public domain, as compiled on Kaggle by
Rob Mulla.
