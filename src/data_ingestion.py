import yfinance as yf
import pandas as pd
from pathlib import Path
import time

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

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

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"


def download_asset(symbol, name):

    print(f"Downloading {name}")

    df = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df.empty:
        print(f"{name} returned empty data")
        return

    df.to_parquet(DATA_DIR / f"{name}.parquet")


def main():

    for name, symbol in ASSETS.items():

        download_asset(symbol, name)

        # prevent Yahoo throttling
        time.sleep(1)


if __name__ == "__main__":
    main()