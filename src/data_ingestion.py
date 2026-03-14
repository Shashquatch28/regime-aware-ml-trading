import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

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

START_DATE = "2005-01-01"
END_DATE = "2024-12-31"


def download_asset(symbol, name):
    df = yf.download(symbol, start=START_DATE, end=END_DATE, auto_adjust=False)

    if df.empty:
        print(f"Failed download: {symbol}")
        return

    df.to_parquet(DATA_DIR / f"{name}.parquet")


def download_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, symbol in ASSETS.items():
        print(f"Downloading {name}")
        download_asset(symbol, name)


if __name__ == "__main__":
    download_all()