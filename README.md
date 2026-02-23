# Vehicle Price Prediction (Sri Lanka)

This project trains a car price prediction model and serves predictions through a Streamlit web app.

## Project Files

- `app.py` – Streamlit app for interactive vehicle price prediction
- `model_new.ipynb` – Main notebook for preprocessing, training, evaluation, and model export
- `car_dataset_before_preprocessed.csv` – Reference dataset used by the app for dropdown values
- `car_preprocessed.csv` – Preprocessed dataset
- `cars_dataset_raw.csv` – Raw dataset
- `catboost_model.pkl` – Trained model artifact used by the app
- `feature_columns.pkl` – Saved feature column order for inference alignment

## Setup

1. Create and activate a Python virtual environment.
2. Install required packages:

```bash
pip install streamlit pandas numpy joblib catboost scikit-learn matplotlib seaborn shap
```

## Train / Export Model

Run the notebook:

- Open `model_new.ipynb`
- Execute all cells in order
- Ensure these files are generated in the project root:
	- `catboost_model.pkl`
	- `feature_columns.pkl`

## Run the App

```bash
streamlit run app.py
```

The app:

- Takes vehicle inputs (Brand, Model, Year, Mileage, Fuel type, Engine capacity, Transmission)
- Predicts price in **millions of LKR**
- Shows an estimated LKR value and a ±10% suggested range
- Displays feature importance and CatBoost local explanation (SHAP-style contributions)

## Notes

- `app.py` expects `catboost_model.pkl` and `feature_columns.pkl` in the same folder.
- If `car_dataset_before_preprocessed.csv` is available, the app uses it to populate realistic dropdown values.
