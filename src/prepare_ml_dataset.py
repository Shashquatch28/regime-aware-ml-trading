import pandas as pd


INPUT_PATH = "data/processed/global_regime_dataset.parquet"
OUTPUT_PATH = "data/processed/final_model_dataset.parquet"


def create_target(df):
    """
    Create ternary classification target:
    +1 → strong positive return
     0 → no trade (noise)
    -1 → strong negative return
    """

    future_return = df["spx_return"].shift(-1)

    # rolling volatility threshold
    threshold = df["spx_return"].rolling(20).std()

    df["target"] = 0

    df.loc[future_return > threshold, "target"] = 1
    df.loc[future_return < -threshold, "target"] = -1

    return df


def add_regime_features(df):
    """
    Add your NOVEL features
    """

    # -----------------------------
    # Regime Probability Momentum (Δπ)
    # -----------------------------
    for i in range(4):
        df[f"delta_regime_prob_{i}"] = df[f"regime_prob_{i}"].diff()

    # -----------------------------
    # Regime Persistence
    # -----------------------------
    df["regime_persistence"] = (
        df["regime_state"] == df["regime_state"].shift(1)
    ).astype(int)

    return df


def main():

    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    print("Creating target variable...")
    df = create_target(df)

    print("Adding regime-based features...")
    df = add_regime_features(df)

    print("Dropping NaNs...")
    df = df.dropna()

    print("\nTarget distribution:")
    print(df["target"].value_counts(normalize=True))

    print("\nSaving dataset...")
    df.to_parquet(OUTPUT_PATH)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()