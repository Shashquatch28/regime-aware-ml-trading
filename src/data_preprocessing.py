import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

FILES = [
    "SP500",
    "NASDAQ",
    "NIFTY50",
    "VIX",
    "DXY",
    "US10Y",
    "GOLD",
    "OIL",
    "BTC"
]


def load_data():

    dfs = []

    for file in FILES:
        df = pd.read_parquet(RAW_DIR / f"{file}.parquet")

        df = df[["Close", "Volume"]]
        df.columns = [f"{file}_close", f"{file}_volume"]

        dfs.append(df)

    data = pd.concat(dfs, axis=1)

    return data


def preprocess():

    df = load_data()

    df = df.sort_index()

    df = df.ffill()

    df = df.dropna()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(PROCESSED_DIR / "market_data.parquet")

    print("Saved processed dataset")


if __name__ == "__main__":
    preprocess()