"""
RandomForest Regressor for Vehicle Price Prediction
Trains and evaluates a RandomForestRegressor with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = Path("ikman_cars_preprocessed.csv")
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading preprocessed data...")
df = pd.read_csv(INPUT_FILE)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Separate target and features
# Assuming 'price_lkr' is the target
y = df["price_lkr"]
X = df.drop(columns=["price_lkr"])

print(f"Target (price_lkr) shape: {y.shape}")
print(f"Features shape: {X.shape}")
print(f"Feature columns: {X.columns.tolist()[:10]}... ({len(X.columns)} total)\n")

# ============================================================================
# 2. TRAIN/VAL/TEST SPLIT (70/15/15)
# ============================================================================
print("Splitting data into train/val/test (70/15/15)...")

# First split: 70% train, 30% temp (for val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE
)

# Second split: divide temp into val and test (50/50 of 30% = 15% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE
)

print(f"Train set: {X_train.shape[0]} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({100*len(X_val)/len(X):.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({100*len(X_test)/len(X):.1f}%)\n")

# ============================================================================
# 3. HYPERPARAMETER TUNING WITH GRIDSEARCHCV (on training set only)
# ============================================================================
print("Tuning hyperparameters with GridSearchCV (5-fold CV on training set)...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, 20],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", "log2"],
}

rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}\n")

# Use the best model
best_model = grid_search.best_estimator_

# ============================================================================
# 4. EVALUATE ON TRAIN, VALIDATION, AND TEST SETS
# ============================================================================
print("Evaluating model on train/val/test sets...")

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return {
        "Dataset": dataset_name,
        "R² Score": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Sample Count": len(y)
    }

results_train = evaluate_model(best_model, X_train, y_train, "Train")
results_val = evaluate_model(best_model, X_val, y_val, "Validation")
results_test = evaluate_model(best_model, X_test, y_test, "Test")

# Create results table
results_df = pd.DataFrame([results_train, results_val, results_test])
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(results_df.to_string(index=False))
print("="*80 + "\n")

# ============================================================================
# 5. PLOT 1: PREDICTED vs ACTUAL (Test Set)
# ============================================================================
print("Generating predicted vs actual scatter plot...")

y_test_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, s=20, color="steelblue")

# Add perfect prediction line
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

plt.xlabel("Actual Price (LKR)", fontsize=12, fontweight="bold")
plt.ylabel("Predicted Price (LKR)", fontsize=12, fontweight="bold")
plt.title("Predicted vs Actual Price - Test Set", fontsize=14, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=300, bbox_inches="tight")
print("Saved: predicted_vs_actual.png\n")
plt.close()

# ============================================================================
# 6. PLOT 2: FEATURE IMPORTANCE (Top 20)
# ============================================================================
print("Generating feature importance plot...")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)

# Top 20 features
top_20 = feature_importance.head(20)

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_20)), top_20["Importance"].values, color="steelblue")
plt.yticks(range(len(top_20)), top_20["Feature"].values, fontsize=10)
plt.xlabel("Importance", fontsize=12, fontweight="bold")
plt.title("Feature Importance - Top 20 Features", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("feature_importance_top20.png", dpi=300, bbox_inches="tight")
print("Saved: feature_importance_top20.png\n")
plt.close()

# ============================================================================
# 7. SUMMARY & BEST HYPERPARAMETERS
# ============================================================================
print("="*80)
print("SELECTED HYPERPARAMETERS (from GridSearchCV)")
print("="*80)
for param, value in grid_search.best_params_.items():
    print(f"{param:.<30} {value}")
print("="*80 + "\n")

print("Model training and evaluation complete!")
print("Outputs:")
print("  1. Results Table (above)")
print("  2. predicted_vs_actual.png")
print("  3. feature_importance_top20.png")
print("\n" + "="*80)
print("SAVING MODEL ARTIFACTS")
print("="*80)

# Save the trained model
joblib.dump(best_model, "rf_model.pkl")
print("✓ Saved: rf_model.pkl")

# Save feature columns to ensure alignment at inference
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("✓ Saved: feature_columns.pkl")

print("="*80)
print("Model artifacts saved successfully!")
print("You can now use app.py for predictions.\n")
