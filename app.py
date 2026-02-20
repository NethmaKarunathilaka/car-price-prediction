import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

CURRENT_YEAR = 2025
MODEL_PATH = "rf_model.pkl"
FEATURES_PATH = "feature_columns.pkl"
FREQ_MAPS_PATH = "model_freq_maps.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    freq_maps = joblib.load(FREQ_MAPS_PATH)
    return model, feature_columns, freq_maps


def format_lkr(value: float) -> str:
    return f"LKR {value:,.0f}"


def build_input_dataframe(
    year: int,
    mileage_km: float,
    engine_capacity_cc: float,
    fuel_type: str,
    transmission: str,
    brand: str,
    model_name: str,
    location: str,
) -> pd.DataFrame:
    vehicle_age = CURRENT_YEAR - year

    row = {
        "year": year,
        "mileage_km": mileage_km,
        "engine_capacity_cc": engine_capacity_cc,
        "vehicle_age": vehicle_age,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "brand": brand.strip(),
        "model": model_name.strip(),
        "location": location.strip(),
    }

    return pd.DataFrame([row])


def main():
    st.set_page_config(page_title="Car Price Predictor", layout="centered")
    st.title("Car Price Prediction")
    st.write("Predict `price_lkr` using your trained RandomForestRegressor model.")

    try:
        rf_model, feature_columns, freq_maps = load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load model files: {exc}")
        st.info("Make sure `rf_model.pkl`, `feature_columns.pkl`, and `model_freq_maps.pkl` exist in the project folder.")
        return

    with st.form("prediction_form"):
        st.subheader("Vehicle Details")

        year = st.number_input("Year", min_value=1950, max_value=CURRENT_YEAR + 1, value=2018, step=1)
        mileage_km = st.number_input("Mileage (km)", min_value=0.0, value=50000.0, step=1000.0)
        engine_capacity_cc = st.number_input("Engine Capacity (cc)", min_value=50.0, value=1500.0, step=50.0)

        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Tiptronic"])

        brand = st.text_input("Brand", value="Toyota")
        model_name = st.text_input("Model", value="Corolla")
        location = st.text_input("Location", value="Colombo")

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        try:
            input_df = build_input_dataframe(
                year=year,
                mileage_km=mileage_km,
                engine_capacity_cc=engine_capacity_cc,
                fuel_type=fuel_type,
                transmission=transmission,
                brand=brand,
                model_name=model_name,
                location=location,
            )

            # Apply frequency encoding to 'model' column using saved maps
            model_map = freq_maps.get("model", {})
            input_df["model"] = input_df["model"].map(model_map).fillna(model_map.get("Unknown", 0.0))
            
            # Apply one-hot encoding to remaining categorical columns
            input_encoded = pd.get_dummies(input_df, columns=["brand", "fuel_type", "transmission", "location"], dtype=int)
            input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)

            prediction = rf_model.predict(input_aligned)[0]

            st.success(f"Predicted Price: {format_lkr(prediction)}")
            st.caption(f"Computed age: {CURRENT_YEAR - int(year)}")

            if hasattr(rf_model, "feature_importances_"):
                importances = pd.DataFrame(
                    {
                        "feature": feature_columns,
                        "importance": rf_model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)

                top10 = importances.head(10).sort_values("importance", ascending=True)

                st.subheader("Top 10 Feature Importances")
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.barh(top10["feature"], top10["importance"])
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                ax.set_title("RandomForest Top 10 Feature Importances")
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as exc:
            st.error("Prediction failed. Please check your inputs and model files.")
            st.exception(exc)


if __name__ == "__main__":
    main()
