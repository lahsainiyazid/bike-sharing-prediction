# Bike Sharing Demand Prediction

Predicting hourly bike rental demand using the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), with a focus on proper time-series feature engineering and evaluation.

## Overview

This project builds a regression model to predict `cnt` — the total number of hourly bike rentals — using weather, calendar, and time-based features. The dataset is a real-world hourly time series (2011–2012), which means special care is needed to avoid data leakage and to evaluate the model the way it would actually be used in production: predicting the future from the past.

## Dataset

- **Source**: UCI Bike Sharing Dataset (hourly resolution, `hour.csv`)
- **Size**: 17,379 hourly records over two years
- **Target**: `cnt` (total rentals per hour), log-transformed (`log1p`) to correct right-skew
- **Raw features**: season, year, month, hour, holiday, weekday, working day, weather situation, temperature, "feels-like" temperature, humidity, windspeed

## Feature Engineering

- **Cyclical encoding**: hour, month, and weekday encoded as sine/cosine pairs to preserve their circular nature (e.g. hour 23 is close to hour 0)
- **Calendar flags**: rush hour (working day + commute hours), off-peak hours
- **Lag features**: previous hour's count, previous day's (same hour) count
- **Rolling statistics**: 3-hour and 24-hour rolling mean/std of past demand
- **Weather interactions**: temp × humidity, temp × windspeed, delta between temp and "feels-like" temp, bad-weather flag

All lag and rolling features use only past values (`shift`/`rolling`, never centered or future-looking), so they reflect information genuinely available at prediction time.

## Methodology

- **Train/test split**: chronological (80/20), *not* randomly shuffled — the model is trained on earlier data and evaluated on later data, matching real-world deployment
- **Leakage avoidance**: `casual` and `registered` (which sum to the target `cnt`) are excluded from the feature set, since they wouldn't be known ahead of time in a real forecasting scenario
- **Target transform**: predictions are made on `log1p(cnt)` and converted back with `expm1` before computing evaluation metrics, so reported errors are in original rental-count units

## Models

| Version | Model | Notes | RMSE |
|---------|-------|-------|------|
| v0 | Random Forest | Default hyperparameters, baseline | — |
| v1 | XGBoost | Tuned via `RandomizedSearchCV` with `TimeSeriesSplit` | 56.5 |
| v2 | XGBoost | Refined feature set (dropped low-value flags) | 47.0 |

RMSE is computed on the held-out chronological test set, in original rental-count units.

## Results

The final model achieves an **RMSE of ~47 rentals/hour** on unseen future data. For context, hourly rental counts in this dataset range from 1 to just under 1,000, with a median in the double digits to low hundreds depending on time of day — so this represents a substantial improvement over a naive baseline.

## Project Structure

```
.
├── bikes.ipynb        # Main notebook: EDA, feature engineering, modeling, evaluation
├── data/
│   └── hour.csv        # UCI hourly dataset
└── README.md
```

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
xgboost
```

## Possible Next Steps

- Model `casual` and `registered` ridership separately (they have distinct usage patterns) and sum predictions
- Add longer lag features (e.g. same hour one week prior) to better capture weekly seasonality
- Expand hyperparameter search space (regularization terms, tree depth)
- Add prediction interval / uncertainty estimates
