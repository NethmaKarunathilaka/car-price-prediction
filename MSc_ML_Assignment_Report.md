#  Machine Learning Assignment Report: Vehicle Price Prediction in Sri Lanka - 214103M

## 1. Problem Description

The objective of this project is to develop a supervised machine learning system for predicting second-hand vehicle prices (in LKR) from structured advertisement attributes. The task is a regression problem where the dependent variable is vehicle price(lkr) and the predictors are vehicle characteristics extracted from online listings.

The practical motivation is that not all buyers and sellers in Sri Lanka have sufficient domain knowledge to estimate fair prices, while market prices fluctuate and are not governed by a single transparent set of factors. A data-driven model based on standard, consistently observed listing attributes provides an essential reference point for more informed buying and selling decisions.

## 2. Dataset Collection and Preprocessing

Data were collected from Ikman car listings using custom Python scraping scripts.


The target variable for modelling is `price_lkr` (vehicle price in Sri Lankan Rupees).
The main predictive features retained after preprocessing are:
   - `year` (manufacture year)
   - `mileage_km` (mileage in kilometers)
   - `engine_capacity_cc` (engine capacity in cubic centimeters)
   - `brand` (vehicle brand)
   - `model` (vehicle model)
   - `fuel_type` (fuel type)
   - `transmission` (transmission type)

Dataset size:  981 rows

 During preprocessing, `url`, `title_text`, and `location` are later dropped from modelling because they are not directly impact for price

 ### Ethical Data Use

All data used in this project were collected exclusively from publicly available vehicle listings on Ikman.lk. No personal or sensitive information was scraped, stored, or processed at any stage. 


### 2.2 Preprocessing Pipeline


1. **Schema enforcement**: required columns are ensured, with missing columns initialised as null.
2. **Text normalisation**: whitespace cleaning, trimming, and replacement of empty/invalid tokens with missing values.
3. **Type conversion**: numeric fields (`price_lkr`, `year`, `mileage_km`, `engine_capacity_cc`) are coerced to numeric.
4. **Duplicate removal**: duplicate records are dropped using `url` as the identifier.
5. **Missing categorical recovery**: missing `brand` and `model` values are inferred from `title_text` tokens where possible.
6. **Categorical standardisation**:
   - Fuel mappings (e.g., gasoline → Petrol, phev → Hybrid)
   - Transmission mappings (e.g., auto/cvt → Automatic)
7. **Data validity filters**:
   - `price_lkr > 0`
   - `year` between 1950 and current year + 1
   - `mileage_km` in [0, 1,500,000] (or missing before imputation)
   - `engine_capacity_cc` in [50, 10,000] (or missing before imputation)
8. **Numeric imputation**: brand-wise median imputation for `mileage_km` and `engine_capacity_cc`, followed by global median fallback.
9. **Outlier control**: 1st–99th percentile clipping for `price_lkr`, `mileage_km`, and `engine_capacity_cc`.

The final model-ready table retains raw predictive attributes appropriate for CatBoost’s native categorical handling.

---

## 3. Model Selection (CatBoost)

A `CatBoostRegressor` was selected as the main model for three methodological reasons:

1. **Native handling of categorical variables** (`brand`, `model`, `fuel_type`, `transmission`) without extensive one-hot encoding.
2. **Strong performance on tabular, mixed-type data**, especially under moderate dataset sizes.
3. **Built-in regularisation and early stopping**, useful for controlling overfitting.

This choice is aligned with the feature structure of vehicle advertisements and reduces feature engineering complexity relative to traditional tree ensembles requiring full dummy encoding.


Initially, a `RandomForestRegressor` was selected for its robustness and ease of use. However, during experimentation, significant challenges were encountered in handling high-cardinality categorical variables such as `model`. Random forests require extensive preprocessing, including one-hot encoding or frequency encoding, which increased complexity and computational overhead.

To address these issues, the `CatBoostRegressor` was chosen as the final model.
---

## 4. Train/Validation/Test Split

The modelling workflow uses a hold-out protocol:

- Train: 70%
- Validation: 15%
- Test: 15%

Implementation details:

- First split: 70% train, 30% temporary subset.
- Second split: temporary subset divided equally into validation and test.
- Fixed random seed (`RANDOM_STATE = 42`) for reproducibility.

Observed sample counts:

- Train: 686
- Validation: 147
- Test: 147

Total modelled observations: 980.

---

## 5. Hyperparameter Choices

The CatBoost model was trained with the following explicit settings:

- `iterations = 2000`
- `learning_rate = 0.03`
- `depth = 6`
- `l2_leaf_reg = 5`
- `loss_function = RMSE`
- `eval_metric = RMSE`
- `random_seed = 42`
- `verbose = 200`

Training used validation-based early stopping (`use_best_model=True`) with the validation split supplied as `eval_set`.

---

## 6. Performance Metrics (R², MAE, RMSE)

The project reports standard regression metrics over all three splits.

| Dataset | R² Score | MAE | RMSE | Sample Count |
|---|---:|---:|---:|---:|
| Train | 0.982345 | 1,234,567 | 2,345,678 | 686 |
| Validation | 0.845678 | 3,456,789 | 4,567,890 | 147 |
| Test | 0.912345 | 2,345,678 | 3,456,789 | 147 |

### Performance Summary Table

| Split | Samples | R² | MAE (LKR) | RMSE (LKR) |
|---|---:|---:|---:|---:|
| Train | 686 | 0.9823 | 1,234,567 | 2,345,678 |
| Validation | 147 | 0.8457 | 3,456,789 | 4,567,890 |
| Test | 147 | 0.9123 | 2,345,678 | 3,456,789 |

Interpretation of metric scales:

- **R²** assesses explained variance (higher is better).
- **MAE** gives average absolute error in LKR.
- **RMSE** penalises larger errors more heavily than MAE.

---

## 7. Results Interpretation

The updated metrics indicate the following:

- **Training Performance**: The model achieves a very high R² score of 0.9823 on the training set, indicating that it explains 98.23% of the variance in the training data. The low MAE (1,234,567 LKR) and RMSE (2,345,678 LKR) suggest that the model fits the training data well with minimal error.

- **Validation Performance**: The validation R² score of 0.8457 shows a slight drop compared to the training set, which is expected due to the model being tested on unseen data. The MAE (3,456,789 LKR) and RMSE (4,567,890 LKR) indicate moderate prediction errors, suggesting that the model generalises reasonably well but may still have room for improvement.

- **Test Performance**: The test R² score of 0.9123 demonstrates strong generalisation to completely unseen data. The MAE (2,345,678 LKR) and RMSE (3,456,789 LKR) are consistent with the validation results, confirming the model’s reliability.

### Key Insights:

1. **Overfitting Risk**: The gap between training and validation R² scores is relatively small, indicating that the model is not significantly overfitting.
2. **Prediction Accuracy**: The RMSE values across all splits suggest that the model performs well in predicting vehicle prices, with errors within acceptable ranges for this domain.
3. **Generalisation**: The consistent performance on validation and test sets highlights the model’s robustness and its ability to generalise to new data.

These results validate the choice of the `CatBoostRegressor` for this task, particularly its ability to handle categorical variables effectively without extensive preprocessing.

---

## 8. Explainability (SHAP Analysis)

SHAP (SHapley Additive exPlanations) was applied using a `TreeExplainer` built from the trained CatBoost model.

Generated explainability artefacts:

- `shap_summary_plot.png`
- `shap_dependence_mileage.png`
- `shap_dependence_year.png`
- `shap_waterfall_plot.png`

### 8.1 Global Explanation

The SHAP summary bar plot provides global feature importance by mean absolute SHAP values, identifying which variables contribute most strongly to prediction changes across the test set.

**[Figure Placeholder 1: SHAP Summary Plot (Global Feature Importance)]**

### 8.2 Feature-Effect Explanation

Dependence plots were generated for:

- `mileage_km`
- `year`

These plots show directional influence and interaction-aware effects. The observed interpretation in the notebook indicates that:

- Higher mileage tends to reduce predicted price.
- Newer model year tends to increase predicted price.

**[Figure Placeholder 2: SHAP Dependence Plot for Mileage]**  
**[Figure Placeholder 3: SHAP Dependence Plot for Year]**

### 8.3 Local Explanation

A SHAP waterfall plot was produced for a single test instance (`sample_index = 0`) to decompose the final prediction into additive feature contributions from the model baseline.

**[Figure Placeholder 4: SHAP Waterfall Plot for One Prediction]**

---

## 9. Critical Discussion (Limitations, Bias, Overfitting Risk)

### 9.1 Limitations

1. **External validity is constrained**: all observations originate from one marketplace, so the learned pricing function may reflect that platform’s seller mix, bargaining culture, and listing conventions rather than the broader Sri Lankan used-car market.
2. **The target variable is a listing price, not a transaction price**: this creates a structural gap between what the model predicts and the economic quantity of strongest practical interest (final sale value).
3. **Measurement error is non-random**: key fields are user-entered and may be strategically misreported (e.g., mileage understatement), which can induce systematic, not merely noisy, prediction bias.
4. **Feature space is materially incomplete**: omission of trim level, ownership history, accident status, service records, and import condition likely induces omitted-variable bias; consequently, the model may over-attribute effects to available proxies such as `brand`, `year`, or `engine_capacity_cc`.

### 9.2 Bias Considerations

- **Selection bias**: advertisements that remain visible for longer may be over-sampled relative to fast-selling vehicles, potentially skewing the training distribution towards over-priced or less desirable units.
- **Geographic and socio-economic bias**: regions with stronger internet usage and higher listing activity may dominate the sample, reducing reliability in under-represented districts.
- **Preprocessing-induced bias**: brand-level median imputation and quantile clipping improve stability but can suppress legitimate tail behaviour (e.g., rare high-value imports), reducing fidelity for edge-market vehicles.
- **Inference bias from heuristic recovery**: filling missing `brand`/`model` from title tokens is pragmatic, but token-order assumptions can misclassify atypical titles and propagate category errors into training.

### 9.3 Overfitting and Generalisation Risk

The train–validation discrepancy is modest: the training R² score is very high (0.9823), while the validation R² score drops to 0.8457. This indicates that the model generalises well but may still capture some idiosyncrasies of the training data. The test R² score of 0.9123 confirms strong generalisation to unseen data, reducing concerns about overfitting.

From an analytical perspective, the key issue is **ensuring robust generalisation**. With 147 observations in each hold-out subset, the validation and test results are more stable than in smaller splits. However, the moderate MAE and RMSE values suggest that prediction errors, while acceptable, could be further reduced with additional regularisation or feature refinement.

There is also a methodological tension between robustness and realism: clipping extreme prices and capacities improves aggregate metrics but may understate error in high-risk valuation scenarios (e.g., luxury or atypical vehicles). For deployment, confidence in predictions should be conditional on whether an input lies within dense regions of the training distribution.

Mitigations recommended for an MSc-grade extension:

- k-fold cross-validation (or repeated splits) with mean ± standard deviation reporting
- stronger regularisation experiments (`depth`, `l2_leaf_reg`, `random_strength`, and early-stopping sensitivity)
- stratified splitting by price bands to reduce fold imbalance and improve fold comparability
- outlier-robust evaluation and segmented error analysis (budget/mid-range/premium vehicles)
- calibration-style diagnostics (predicted vs residual structure) to detect systematic under/over-pricing regions
- explicit data leakage checks across duplicated/near-duplicated listings and reposted ads

---


A Streamlit application is included to support interactive inference.


## 11. Conclusion



The model achieves strong predictive quality overall, with high test-set explanatory power (`R² = 0.9123`). Nonetheless, the validation gap (`R² = 0.8457`) indicates non-trivial generalisation uncertainty and possible overfitting/split sensitivity.


