import pandas as pd

from rolling_regime import rolling_hmm_regimes

FEATURES = [
    "spx_return",
    "spx_vol_20",
    "spx_vol_60",
    "vix_level",
    "stress_index",
    "us10y_vol"
]


def run_experiment(window):

    print(f"\nRunning rolling HMM with window={window}")

    df = pd.read_parquet("data/processed/regime_features.parquet")

    regimes = rolling_hmm_regimes(
        df,
        FEATURES,
        window=window
    )

    df = df.loc[regimes.index]

    df = pd.concat([df, regimes], axis=1)

    output_path = f"data/processed/regime_dataset_window_{window}.parquet"

    df.to_parquet(output_path)

    print("Saved:", output_path)


def main():

    for window in [750, 1500, 3000]:

        run_experiment(window)


if __name__ == "__main__":
    main()