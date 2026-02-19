# Car Price Prediction Model

This project scrapes vehicle data from ikman.lk and builds a machine learning model to predict car prices based on specifications.

## Files

- **ikman_full_dataset.py** – Web scraper for collecting vehicle listings from ikman.lk
- **preprocessing.py** – Data preprocessing pipeline (convert types, handle missing values, feature engineering)
- **preprocessing.ipynb** – Interactive Jupyter notebook version of preprocessing
- **ikman_cars_model_ready.csv** – Final cleaned dataset ready for modeling

## Usage

### 1. Scrape Data
```bash
python ikman_full_dataset.py
```
Outputs: `ikman_cars_dataset.csv`

### 2. Preprocess Data (Script)
```bash
python preprocessing.py
```
Outputs: `ikman_cars_model_ready.csv`

### 3. Preprocess Data (Notebook)
Open and run `preprocessing.ipynb` cell-by-cell in Jupyter.

## Data Pipeline

1. **Load** – Read raw CSV
2. **Inspect structure** – Check columns and schema
3. **Handle missing values** – Fill/remove incomplete records
4. **Convert data types** – Numeric/categorical normalization
5. **Feature engineering** – vehicle_age, mileage_per_year
6. **Handle outliers** – Clip to 1st–99th percentile
7. **Encode categorical variables** – One-hot encoding
8. **Prepare final modeling dataset** – Export cleaned data

## Features

- **price_lkr** (target) – Vehicle price in Sri Lankan Rupees
- **brand, model** – Make and model
- **year, mileage_km** – Age and odometer reading
- **fuel_type, transmission** – Petrol, Diesel, Hybrid, Electric, etc.
- **engine_capacity_cc** – Engine displacement
- **location** – Sale location
- **vehicle_age, mileage_per_year** – Engineered features

## Next Steps

- Train regression model (Random Forest, XGBoost, Linear Regression)
- Evaluate on test set (MAE, RMSE, R²)
- Deploy API for price predictions
