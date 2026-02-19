import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


CURRENT_YEAR = pd.Timestamp.today().year
EXPECTED_COLUMNS = [
	"url",
	"title_text",
	"price_lkr",
	"brand",
	"model",
	"year",
	"mileage_km",
	"fuel_type",
	"engine_capacity_cc",
	"transmission",
	"location",
]
NUMERIC_COLUMNS = ["price_lkr", "year", "mileage_km", "engine_capacity_cc"]
CATEGORICAL_COLUMNS = ["brand", "model", "fuel_type", "transmission", "location"]


def load_dataset(file_path: Path) -> pd.DataFrame:
	try:
		return pd.read_csv(file_path, low_memory=False)
	except UnicodeDecodeError:
		return pd.read_csv(file_path, encoding="latin-1", low_memory=False)


def resolve_input_path(input_value: str | None) -> Path:
	if input_value:
		path = Path(input_value)
		if path.exists():
			return path
		raise FileNotFoundError(f"Input dataset not found: {input_value}")

	candidates = [
		"ikman_cars_dataset-Copy.csv",
		"ikman_cars_dataset - Copy.csv",
		"ikman_cars_dataset.csv",
	]
	for candidate in candidates:
		candidate_path = Path(candidate)
		if candidate_path.exists():
			return candidate_path

	raise FileNotFoundError(
		"No input dataset found. Expected one of: " + ", ".join(candidates)
	)


def normalize_text(series: pd.Series) -> pd.Series:
	normalized = (
		series.astype(str)
		.str.replace(r"\s+", " ", regex=True)
		.str.strip()
		.replace({"": np.nan, "nan": np.nan, "None": np.nan})
	)
	return normalized


def extract_location_from_url(url: str | float | None) -> str | None:
	if not isinstance(url, str):
		return None
	url = url.strip()
	if not url:
		return None
	match = re.search(r"for-sale-([a-z\-]+)(?:-\d+)?$", url, flags=re.I)
	if not match:
		return None
	return match.group(1).replace("-", " ").title()


def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
	for column in EXPECTED_COLUMNS:
		if column not in df.columns:
			df[column] = np.nan
	return df[EXPECTED_COLUMNS].copy()


def standardize_fuel_type(series: pd.Series) -> pd.Series:
	mapping = {
		"petrol": "Petrol",
		"gasoline": "Petrol",
		"diesel": "Diesel",
		"hybrid": "Hybrid",
		"electric": "Electric",
		"plugin hybrid": "Hybrid",
		"plug-in hybrid": "Hybrid",
		"phev": "Hybrid",
	}
	normalized = series.str.lower().str.strip().replace(mapping)
	return normalized.str.title().fillna("Unknown")


def standardize_transmission(series: pd.Series) -> pd.Series:
	mapping = {
		"auto": "Automatic",
		"automatic": "Automatic",
		"manual": "Manual",
		"tiptronic": "Tiptronic",
		"cvt": "Automatic",
	}
	normalized = series.str.lower().str.strip().replace(mapping)
	return normalized.str.title().fillna("Unknown")


def fill_brand_model_from_title(df: pd.DataFrame) -> pd.DataFrame:
	split_tokens = df["title_text"].fillna("").str.split()
	missing_brand = df["brand"].isna()
	missing_model = df["model"].isna()

	df.loc[missing_brand, "brand"] = split_tokens[missing_brand].str.get(0)
	df.loc[missing_model, "model"] = split_tokens[missing_model].str.get(1)
	return df


def impute_numeric_with_brand_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
	brand_median = df.groupby("brand")[column].transform("median")
	df[column] = df[column].fillna(brand_median)
	df[column] = df[column].fillna(df[column].median())
	return df


def clip_quantile(series: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
	lower = series.quantile(low)
	upper = series.quantile(high)
	return series.clip(lower=lower, upper=upper)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
	# Inspect structure
	df.columns = [col.strip() for col in df.columns]
	df = ensure_expected_columns(df)

	# Handle missing values
	for column in ["url", "title_text", *CATEGORICAL_COLUMNS]:
		df[column] = normalize_text(df[column])

	# Convert data types
	for column in NUMERIC_COLUMNS:
		df[column] = pd.to_numeric(df[column], errors="coerce")

	df = df.drop_duplicates(subset=["url"], keep="first")
	df = fill_brand_model_from_title(df)
	df["location"] = df["location"].fillna(df["url"].apply(extract_location_from_url))

	df["brand"] = df["brand"].str.title().fillna("Unknown")
	df["model"] = df["model"].str.title().fillna("Unknown")
	df["location"] = df["location"].str.title().fillna("Unknown")
	df["fuel_type"] = standardize_fuel_type(df["fuel_type"])
	df["transmission"] = standardize_transmission(df["transmission"])

	# Handle missing values
	df = df[df["price_lkr"].notna()]
	df = df[df["price_lkr"] > 0]
	df = df[df["year"].between(1950, CURRENT_YEAR + 1, inclusive="both")]
	df = df[df["mileage_km"].isna() | df["mileage_km"].between(0, 1_500_000)]
	df = df[df["engine_capacity_cc"].isna() | df["engine_capacity_cc"].between(50, 10_000)]

	df = impute_numeric_with_brand_median(df, "mileage_km")
	df = impute_numeric_with_brand_median(df, "engine_capacity_cc")

	# Handle outliers
	df["price_lkr"] = clip_quantile(df["price_lkr"])
	df["mileage_km"] = clip_quantile(df["mileage_km"])
	df["engine_capacity_cc"] = clip_quantile(df["engine_capacity_cc"])

	# Feature engineering (age etc.)
	df["vehicle_age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)
	df["mileage_per_year"] = df["mileage_km"] / np.maximum(df["vehicle_age"], 1)

	ordered_columns = [
		"price_lkr",
		"brand",
		"model",
		"year",
		"mileage_km",
		"fuel_type",
		"engine_capacity_cc",
		"transmission",
		"location",
		"vehicle_age",
		"mileage_per_year",
	]
	return df[ordered_columns].reset_index(drop=True)


def create_model_ready_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
	# Encode categorical variables
	model_df = df_clean.drop(columns=["url", "title_text"], errors="ignore")
	model_df = pd.get_dummies(
		model_df,
		columns=["brand", "model", "fuel_type", "transmission", "location"],
		dtype=int,
	)
	# Prepare final modelling dataset
	return model_df


def run(input_path: Path, clean_output_path: Path, model_output_path: Path) -> None:
	# Load dataset
	raw_df = load_dataset(input_path)
	clean_df = preprocess(raw_df)
	model_df = create_model_ready_dataset(clean_df)

	clean_df.to_csv(clean_output_path, index=False)
	model_df.to_csv(model_output_path, index=False)

	print(f"Input rows: {len(raw_df):,}")
	print(f"Cleaned rows: {len(clean_df):,}")
	print(f"Model-ready rows: {len(model_df):,}")
	print(f"Saved cleaned dataset: {clean_output_path}")
	print(f"Saved model-ready dataset: {model_output_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Preprocess vehicle price dataset")
	parser.add_argument(
		"--input",
		default=None,
		help="Path to raw CSV dataset (auto-detects ikman_cars_dataset-Copy.csv / ikman_cars_dataset - Copy.csv / ikman_cars_dataset.csv if omitted)",
	)
	parser.add_argument(
		"--clean-output",
		default="ikman_cars_cleaned.csv",
		help="Path to save cleaned dataset",
	)
	parser.add_argument(
		"--model-output",
		default="ikman_cars_model_ready.csv",
		help="Path to save model-ready dataset",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	input_path = resolve_input_path(args.input)
	run(
		input_path=input_path,
		clean_output_path=Path(args.clean_output),
		model_output_path=Path(args.model_output),
	)
