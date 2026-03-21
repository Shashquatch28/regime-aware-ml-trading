import pandas as pd


INPUT_PATH = "data/processed/global_regime_dataset.parquet"
OUTPUT_PATH = "data/processed/baseline_dataset.parquet"


def create_direction(df):
    """
    Binary direction: up or down
    """

    future_return = df["spx_return"].shift(-1)

    df["direction"] = 1
    df.loc[future_return < 0, "direction"] = -1

    return df


def create_meta_label(df):
    """
    Trade filter:
    Only trade when movement is significant
    """

    future_return = df["spx_return"].shift(-1)

    # volatility-based threshold
    threshold = df["spx_return"].rolling(20).std()

    df["meta_label"] = 0
    df.loc[abs(future_return) > threshold, "meta_label"] = 1

    return df


def main():

    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    print("Creating direction target...")
    df = create_direction(df)

    print("Creating meta label...")
    df = create_meta_label(df)

    print("Dropping NaNs...")
    df = df.dropna()

    print("\nDirection distribution:")
    print(df["direction"].value_counts(normalize=True))

    print("\nMeta-label distribution:")
    print(df["meta_label"].value_counts(normalize=True))

    print("\nSaving dataset...")
    df.to_parquet(OUTPUT_PATH)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()