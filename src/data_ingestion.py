# Download historical OHLCV data for major market assets from Yahoo Finance and save each as a parquet file

import yfinance as yf
import pandas as pd
from pathlib import Path
import time

# Define directory to store raw parquet files, create it if it doesn't exist
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Map of asset names to their Yahoo Finance ticker symbols
ASSETS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "NIFTY50": "^NSEI",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "BTC": "BTC-USD"
}

# Define the historical date range for data download
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"


def download_asset(symbol, name):
    print(f"Downloading {name}")

    # Fetch OHLCV data from Yahoo Finance without auto-adjusting for splits/dividends
    df = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    # Skip saving if no data was returned for this asset
    if df.empty:
        print(f"{name} returned empty data")
        return

    # Save the downloaded data as a parquet file named after the asset
    df.to_parquet(DATA_DIR / f"{name}.parquet")


def main():
    # Iterate over all assets and download each one sequentially
    for name, symbol in ASSETS.items():
        download_asset(symbol, name)
        time.sleep(1)  # Prevent Yahoo Finance throttling


if __name__ == "__main__":
    main()