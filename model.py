import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import shap

# Load raw dataset
df = pd.read_csv("car_dataset_before_preprocessed.csv")

# Show first few rows
df.head()

# Check shape and column names
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check missing values
df.isna().sum()

# Clean text columns (remove extra spaces and make consistent text)
text_cols = ["Brand", "Model", "Fuel type", "Transmission"]

for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

# Standardise formatting
df["Brand"] = df["Brand"].str.title()
df["Model"] = df["Model"].str.title()
df["Fuel type"] = df["Fuel type"].str.title()
df["Transmission"] = df["Transmission"].str.title()

# Check unique values (optional)
df[text_cols].nunique()

# Check unique categories after standardising
for col in ["Brand", "Model", "Fuel type", "Transmission"]:
    print(f"\n{col} unique values:")
    print(df[col].unique())

# Quick summary to see min/max and percentiles
df[["Price (M)", "Mileage (km)", "Engine capacity (CC)"]].describe(percentiles=[0.01, 0.05, 0.95, 0.99])

# Work on a copy
df_clean = df.copy()

cols_to_clip = ["Price (M)", "Mileage (km)", "Engine capacity (CC)"]

for col in cols_to_clip:
    low = df_clean[col].quantile(0.01)
    high = df_clean[col].quantile(0.99)
    df_clean[col] = df_clean[col].clip(lower=low, upper=high)

print("Done: Outliers clipped using 1% and 99% quantiles.")
df_clean[cols_to_clip].describe(percentiles=[0.01, 0.99])

# Step 1: Count occurrences of each Brand-Model pair
brand_model_counts = df.groupby(["Brand", "Model"]).size().reset_index(name="Count")

# Step 2: Identify rare pairs (Count < 10)
rare_pairs = brand_model_counts[brand_model_counts["Count"] < 10]

# Step 3: Sort by count (smallest first)
rare_pairs = rare_pairs.sort_values(by="Count")

# Show rare Brand-Model pairs
print("Rare Brand-Model Pairs:")
print(rare_pairs)

# Step 4: Frequency distribution (1,2,3,... counts)
frequency_distribution = rare_pairs["Count"].value_counts().sort_index()

print("\nFrequency Distribution:")
print(frequency_distribution)

# Get Brand-Model pairs with Count <= 3
remove_pairs = rare_pairs[rare_pairs["Count"] <= 3][["Brand", "Model"]]

# Remove them from original dataframe
df_filtered = df.merge(
    remove_pairs,
    on=["Brand", "Model"],
    how="left",
    indicator=True
)

# Keep only rows NOT in remove_pairs
df_filtered = df_filtered[df_filtered["_merge"] == "left_only"].drop(columns=["_merge"])

# Check size
print("Original shape:", df.shape)
print("Filtered shape:", df_filtered.shape)

# Use filtered dataset (important)
df = df_filtered.copy()

# Define inputs (X) and output (y)
X = df.drop(columns=["Price (M)"])
y = df["Price (M)"]

# Identify categorical columns
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Categorical columns:", cat_features)

# Split into 70% train and 30% temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Split temp into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Show shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Base CatBoost model
cat_model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=42,
    verbose=0
)

# Parameter grid (what we want to test)
param_grid = {
    "iterations": [300, 400, 500],
    "depth": [6, 8, 10, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring="r2",
    cv=4,
    n_jobs=-1
)

# Run Grid Search
grid_search.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

# Best model
best_model = grid_search.best_estimator_

# Best parameters
print("Best Parameters:")
print(grid_search.best_params_)

# Best CV score
print("Best CV R2 Score:")
print(grid_search.best_score_)

# Train model
best_cat = CatBoostRegressor(
    iterations=300,
    depth=8,
    learning_rate=0.03,
    loss_function="RMSE",
    random_seed=42,
    verbose=0
)

best_cat.fit(X_train, y_train, cat_features=cat_features)

# Predict
y_train_pred = best_cat.predict(X_train)
y_val_pred = best_cat.predict(X_val)
y_test_pred = best_cat.predict(X_test)

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Normalised metrics
    nmse = mse / np.var(y_true)          # NMSE = MSE / Var(y)
    nmae = mae / np.mean(y_true)         # NMAE = MAE / Mean(y)

    return r2, mse, rmse, mae, nmse, nmae

# Compute metrics
train_r2, train_mse, train_rmse, train_mae, train_nmse, train_nmae = metrics(y_train, y_train_pred)
val_r2, val_mse, val_rmse, val_mae, val_nmse, val_nmae = metrics(y_val, y_val_pred)
test_r2, test_mse, test_rmse, test_mae, test_nmse, test_nmae = metrics(y_test, y_test_pred)

# Print results
print("=== Train ===")
print("R2   :", train_r2)
print("NMSE :", train_nmse)
print("NMAE :", train_nmae)

print("\n=== Validation ===")
print("R2   :", val_r2)
print("NMSE :", val_nmse)
print("NMAE :", val_nmae)

print("\n=== Test ===")
print("R2   :", test_r2)
print("NMSE :", test_nmse)
print("NMAE :", test_nmae)

# SHAP analysis
train_pool = Pool(X_train, y_train, cat_features=cat_features)
shap_values = best_cat.get_feature_importance(train_pool, type="ShapValues")
shap_values = shap_values[:, :-1]  # Remove base value

# Beeswarm plot
shap.summary_plot(shap_values, X_train, show=True)