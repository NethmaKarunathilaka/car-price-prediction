from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

CURRENT_YEAR = datetime.now().year
MODEL_PATH = "catboost_model.pkl"
FEATURES_PATH = "feature_columns.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def format_millions(value: float) -> str:
    return f"{value:,.2f} M"


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
        "Brand": brand.strip().title(),
        "Model": model_name.strip().title(),
        "Year": int(year),
        "Mileage (km)": float(mileage_km),
        "Fuel type": fuel_type.strip().title(),
        "Engine capacity (CC)": float(engine_capacity_cc),
        "Transmission": transmission.strip().title(),
    }
    return pd.DataFrame([row])


def show_global_explanations(model, feature_columns):
    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        st.subheader("Model Explanations")
        st.caption("Top global feature importance values from the trained model.")
        st.bar_chart(importances.head(10).set_index("Feature"))


def show_local_explanations(model, input_aligned):
    try:
        from catboost import Pool

        cat_features = [
            idx
            for idx, column in enumerate(input_aligned.columns)
            if input_aligned[column].dtype == "object"
        ]
        input_pool = Pool(input_aligned, cat_features=cat_features)
        shap_values = model.get_feature_importance(input_pool, type="ShapValues")

        contributions = shap_values[0, :-1]
        contribution_df = pd.DataFrame(
            {
                "Feature": input_aligned.columns,
                "Contribution": contributions,
                "AbsContribution": np.abs(contributions),
            }
        ).sort_values("AbsContribution", ascending=False)

        st.caption("Top local feature contributions for this specific prediction.")
        st.dataframe(
            contribution_df[["Feature", "Contribution"]].head(10),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.info("Local contribution explanations are available for CatBoost-compatible models.")


def main():
    st.set_page_config(page_title="Vehicle Price Predictor", layout="centered")
    st.title("Vehicle Price Predictor")
    st.write("Enter vehicle features, press predict, and view both price and explanations.")

    try:
        model, feature_columns = load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load model files: {exc}")
        st.info("Ensure `catboost_model.pkl` and `feature_columns.pkl` are in the project folder.")
        return

    with st.form("prediction_form"):
        st.subheader("Input Features")

        brand = st.text_input("Brand", value="Toyota")
        model_name = st.text_input("Model", value="Axio")
        year = st.number_input("Year", min_value=1950, max_value=CURRENT_YEAR + 1, value=2018, step=1)
        mileage_km = st.number_input("Mileage (km)", min_value=0.0, value=50000.0, step=1000.0)
        fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "Hybrid", "Electric"])
        engine_capacity_cc = st.number_input("Engine capacity (CC)", min_value=50.0, value=1500.0, step=50.0)
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Tiptronic"])

        submitted = st.form_submit_button("Predict Price")

    if submitted:
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
            prediction = float(model.predict(input_aligned)[0])

            st.success(f"Predicted Price: {format_millions(prediction)}")
            show_global_explanations(model, feature_columns)
            show_local_explanations(model, input_aligned)

        except Exception as exc:
            st.error("Prediction failed. Check your inputs and model artifacts.")
            st.exception(exc)


if __name__ == "__main__":
    main()
