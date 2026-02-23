from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

CURRENT_YEAR = datetime.now().year

# ---------------------------
# Model artefacts (rename if needed)
# ---------------------------
MODEL_PATH = "catboost_model.pkl"  
FEATURES_PATH = "feature_columns.pkl"

# Optional: use your dataset to populate dropdowns (recommended)
DATA_PATH = "car_dataset_before_preprocessed.csv"


@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


@st.cache_data
def load_reference_data():
    """
    Used to populate dropdowns with realistic values.
    If file not found, app will fall back to text inputs.
    """
    try:
        df = pd.read_csv(DATA_PATH)

        # Standardise like your notebook
        for col in ["Brand", "Model", "Fuel type", "Transmission"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()

        # Ensure numeric columns exist (if present)
        for col in ["Year", "Mileage (km)", "Engine capacity (CC)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    except Exception:
        return None


def format_millions(value: float) -> str:
    return f"{value:,.2f} M"


def format_lkr(value_millions: float) -> str:
    return f"LKR {(value_millions * 1_000_000):,.0f}"


def build_input_dataframe(
    brand: str,
    model_name: str,
    year: int,
    mileage_km: float,
    fuel_type: str,
    engine_capacity_cc: float,
    transmission: str,
) -> pd.DataFrame:
    row = {
        "Brand": str(brand).strip().title(),
        "Model": str(model_name).strip().title(),
        "Year": int(year),
        "Mileage (km)": float(mileage_km),
        "Fuel type": str(fuel_type).strip().title(),
        "Engine capacity (CC)": float(engine_capacity_cc),
        "Transmission": str(transmission).strip().title(),
    }
    return pd.DataFrame([row])


def out_of_range_warnings(ref_df, year, mileage, engine_cc):
    """
    Friendly warnings if user enters values far from typical training values.
    """
    if ref_df is None:
        return

    if "Year" in ref_df.columns:
        q1, q99 = ref_df["Year"].quantile(0.01), ref_df["Year"].quantile(0.99)
        if year < q1 or year > q99:
            st.warning("Year looks unusual compared to the training data. Prediction may be less reliable.")

    if "Mileage (km)" in ref_df.columns:
        q1, q99 = ref_df["Mileage (km)"].quantile(0.01), ref_df["Mileage (km)"].quantile(0.99)
        if mileage < q1 or mileage > q99:
            st.warning("Mileage looks unusual compared to the training data. Prediction may be less reliable.")

    if "Engine capacity (CC)" in ref_df.columns:
        q1, q99 = ref_df["Engine capacity (CC)"].quantile(0.01), ref_df["Engine capacity (CC)"].quantile(0.99)
        if engine_cc < q1 or engine_cc > q99:
            st.warning("Engine capacity looks unusual compared to the training data. Prediction may be less reliable.")


def show_global_explanations(model, feature_columns):
    st.subheader("Model Insights")
    st.caption("Top features used by the model (global importance).")

    # CatBoost
    if hasattr(model, "get_feature_importance"):
        try:
            imps = model.get_feature_importance()
            imp_df = (
                pd.DataFrame({"Feature": feature_columns, "Importance": imps})
                .sort_values("Importance", ascending=False)
                .head(10)
            )
            st.bar_chart(imp_df.set_index("Feature"))
            return
        except Exception:
            pass

    # sklearn fallback
    if hasattr(model, "feature_importances_"):
        imp_df = (
            pd.DataFrame({"Feature": feature_columns, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(10)
        )
        st.bar_chart(imp_df.set_index("Feature"))
    else:
        st.info("Global feature importance is not available for this model format.")


def show_local_explanations_catboost(model, input_aligned):
    """
    CatBoost local explanation: top positive and negative contributions for the prediction.
    """
    try:
        from catboost import Pool

        cat_features = [
            idx for idx, c in enumerate(input_aligned.columns)
            if input_aligned[c].dtype == "object"
        ]
        pool = Pool(input_aligned, cat_features=cat_features)
        shap_values = model.get_feature_importance(pool, type="ShapValues")

        # last column is expected value (base)
        contrib = shap_values[0, :-1]
        base = shap_values[0, -1]
        pred_explained = float(base + contrib.sum())

        contrib_df = pd.DataFrame(
            {"Feature": input_aligned.columns, "Contribution": contrib}
        ).sort_values("Contribution", ascending=False)

        st.markdown("## 🔍 Why this price?")
        st.caption("Positive values increase the price, negative values decrease it.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Top positive drivers")
            st.dataframe(
                contrib_df.head(5),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.markdown("### Top negative drivers")
            st.dataframe(
                contrib_df.tail(5).sort_values("Contribution"),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"Model baseline (expected value): {format_millions(base)} | "
            f"Explained prediction: {format_millions(pred_explained)}"
        )

    except Exception:
        st.info("Local explanations are available when using a CatBoost model artifact.")


def main():
    st.set_page_config(page_title="Vehicle Price Predictor", layout="wide")
    st.title("Vehicle Price Predictor (Sri Lanka)")
    st.caption("Predict used car price in **Millions (LKR)** and view simple explanations.")

    # Load artefacts
    try:
        model, feature_columns = load_model_and_features()
    except Exception as exc:
        st.error(f"Failed to load model files: {exc}")
        st.info(f"Ensure `{MODEL_PATH}` and `{FEATURES_PATH}` are in the same folder as app.py.")
        return

    # Load reference dataset for dropdowns
    ref_df = load_reference_data()

    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.write("Model: CatBoost Regressor")
        st.write("Output: Price (M) = price in millions of LKR")
        st.markdown("---")
        st.write("Tip: Use dropdowns for realistic values.")

    # ---------------------------
    # 1) Two-column layout only for input + prediction
    # ---------------------------
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Enter vehicle details")

        # Dropdowns if data exists
        if ref_df is not None and "Brand" in ref_df.columns:
            brands = sorted(ref_df["Brand"].dropna().unique().tolist())
            brand = st.selectbox("Brand", brands if brands else ["Toyota"])

            if "Model" in ref_df.columns:
                models = sorted(ref_df.loc[ref_df["Brand"] == brand, "Model"].dropna().unique().tolist())
                model_name = st.selectbox("Model", models if models else ["Axio"])
            else:
                model_name = st.text_input("Model", value="Axio")

            fuels = sorted(ref_df["Fuel type"].dropna().unique().tolist()) if "Fuel type" in ref_df.columns else ["Petrol", "Diesel", "Hybrid", "Electric"]
            fuel_type = st.selectbox("Fuel type", fuels)

            trans = sorted(ref_df["Transmission"].dropna().unique().tolist()) if "Transmission" in ref_df.columns else ["Automatic", "Manual", "Tiptronic"]
            transmission = st.selectbox("Transmission", trans)
        else:
            brand = st.text_input("Brand", value="Toyota")
            model_name = st.text_input("Model", value="Axio")
            fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "Hybrid", "Electric"])
            transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Tiptronic"])

        year = st.number_input("Year", min_value=1950, max_value=CURRENT_YEAR + 1, value=2018, step=1)
        mileage_km = st.number_input("Mileage (km)", min_value=0.0, value=50000.0, step=1000.0)
        engine_capacity_cc = st.number_input("Engine capacity (CC)", min_value=50.0, value=1500.0, step=50.0)

        out_of_range_warnings(ref_df, year, mileage_km, engine_capacity_cc)

        predict_btn = st.button("Predict Price", type="primary")

    # We'll store prediction inputs so we can use them below the columns too
    input_aligned = None
    predicted_value = None

    with right:
        st.subheader("Prediction")

        if not predict_btn:
            st.write("Enter inputs and click **Predict Price**.")
        else:
            try:
                input_df = build_input_dataframe(
                    brand=brand,
                    model_name=model_name,
                    year=year,
                    mileage_km=mileage_km,
                    fuel_type=fuel_type,
                    engine_capacity_cc=engine_capacity_cc,
                    transmission=transmission,
                )

                input_aligned = input_df.reindex(columns=feature_columns)
                predicted_value = float(model.predict(input_aligned)[0])

                st.success(
                    f"Estimated Price: **{format_millions(predicted_value)}**  "
                    f"({format_lkr(predicted_value)})"
                )

                # Simple suggested range
                low = predicted_value * 0.90
                high = predicted_value * 1.10
                st.caption(f"Suggested range (±10%): {format_millions(low)} to {format_millions(high)}")

            except Exception as exc:
                st.error("Prediction failed. Please check inputs and model artefacts.")
                st.exception(exc)
                return

    # ---------------------------
    # 2) Full-width explanations BELOW the two columns
    # ---------------------------
    if predict_btn and input_aligned is not None:
        st.markdown("---")
        show_global_explanations(model, feature_columns)

        st.markdown("---")
        show_local_explanations_catboost(model, input_aligned)


if __name__ == "__main__":
    main()