# Car Price Prediction

Scrape vehicle listings from **ikman.lk**, preprocess the dataset, and prepare data for training a machine learning model to predict car prices in Sri Lankan Rupees (LKR).

## Project Structure

- `ikman_full_dataset.py` — Web scraper for collecting vehicle listings from ikman.lk
- `preprocessing.py` — Data preprocessing pipeline (type conversion, missing values, feature engineering)
- `preprocessing.ipynb` — Notebook version of preprocessing
- `ikman_cars_model_ready.csv` — Final cleaned dataset ready for modeling

## Getting Started

### 1) Scrape data

```bash
python ikman_full_dataset.py
```

Output: `ikman_cars_dataset.csv`

### 2) Preprocess data (script)

```bash
python preprocessing.py
```

Output: `ikman_cars_model_ready.csv`

### 3) Preprocess data (notebook)

Open `preprocessing.ipynb` and run the cells in order.

## Data Pipeline

1. Load raw CSV
2. Inspect columns / schema
3. Handle missing values (fill/remove)
4. Convert data types
5. Feature engineering (e.g., `vehicle_age`, `mileage_per_year`)
6. Handle outliers (clip to 1st–99th percentile)
7. Encode categorical variables (one-hot encoding)
8. Export final modeling dataset

## Dataset Features

- `price_lkr` (target) — Vehicle price in Sri Lankan Rupees
- `brand`, `model` — Make and model
- `year`, `mileage_km` — Vehicle year and odometer reading
- `fuel_type`, `transmission` — Fuel and transmission type
- `engine_capacity_cc` — Engine displacement
- `location` — Sale location
- `vehicle_age`, `mileage_per_year` — Engineered features

## Next Steps

- Train regression models (e.g., Linear Regression, Random Forest, XGBoost)
- Evaluate on a holdout test set (MAE, RMSE, R²)
- Deploy a simple API for price prediction