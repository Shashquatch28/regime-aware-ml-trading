import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ASSETS = [
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


def load_asset(asset):
    df = pd.read_parquet(RAW_DIR / f"{asset}.parquet")
    df = df[["Close", "Volume"]]
    df.columns = [f"{asset}_close", f"{asset}_volume"]
    return df


def build_dataset():

    dfs = []

    for asset in ASSETS:
        df = load_asset(asset)
        dfs.append(df)

    data = pd.concat(dfs, axis=1)

    data = data.sort_index()

    data = data.ffill()

    data = data.dropna()

    return data


def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()

    df.to_parquet(PROCESSED_DIR / "market_data.parquet")

    print("Saved processed dataset")


if __name__ == "__main__":
    main()