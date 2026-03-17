# Load raw parquet files for all assets, merge into a single clean dataset, and save to processed directory

import pandas as pd
from pathlib import Path

# Define paths for raw input data and processed output data
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# List of all assets to include in the merged dataset
ASSETS = [
    "SP500", "NASDAQ", "NIFTY50", "VIX",
    "DXY", "US10Y", "GOLD", "OIL", "BTC"
]


def load_asset(asset):
    # Read raw parquet file and keep only Close and Volume columns
    df = pd.read_parquet(RAW_DIR / f"{asset}.parquet")
    df = df[["Close", "Volume"]]

    # Prefix column names with asset name to avoid conflicts when merging
    df.columns = [f"{asset}_close", f"{asset}_volume"]
    return df


def build_dataset():
    dfs = []

    # Load each asset and collect into a list
    for asset in ASSETS:
        df = load_asset(asset)
        dfs.append(df)

    # Merge all assets side by side on their date index
    data = pd.concat(dfs, axis=1)

    # Sort by date, forward-fill missing values, and drop any remaining NaNs
    data = data.sort_index()
    data = data.ffill()
    data = data.dropna()

    return data


def main():
    # Ensure the processed output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Build and save the merged dataset as a single parquet file
    df = build_dataset()
    df.to_parquet(PROCESSED_DIR / "market_data.parquet")

    print("Saved processed dataset")


if __name__ == "__main__":
    main()