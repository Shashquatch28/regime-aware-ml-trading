import pandas as pd

from ml_models_baseline import (
    train_models_per_regime,
    predict_with_regimes,
    generate_signal
)

INPUT_PATH = "data/processed/baseline_dataset.parquet"
OUTPUT_PATH = "data/processed/baseline_predictions.parquet"


def main():

    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    FEATURES = [
        'spx_vol_20', 'spx_vol_60', 'vol_ratio', 'vol_gradient',
        'spx_mom_20', 'spx_mom_60', 'gold_mom_20',
        'vix_level', 'vix_return',
        'spx_gold_corr', 'spx_vix_corr',
        'gold_return', 'oil_return', 'dxy_return',
        'inflation', 'volume_spike', 'amihud_illiquidity',
        'spx_zscore', 'spx_return', 'spx_sq_return',
        'us10y_vol', 'stress_index',
        'yield_spread_diff', 'fed_rate_diff',
        'unemployment_diff', 'industrial_prod_diff',

        # regime probabilities
        'regime_prob_0', 'regime_prob_1',
        'regime_prob_2', 'regime_prob_3'
    ]

    print("Training models...")
    dir_models, meta_models = train_models_per_regime(df, FEATURES)

    print("Predicting...")
    direction_pred, meta_pred = predict_with_regimes(
        df, dir_models, meta_models, FEATURES
    )

    print("Generating signals...")
    signals = generate_signal(direction_pred, meta_pred)

    df["signal"] = signals

    print("\nSignal distribution:")
    print(df["signal"].value_counts(normalize=True))

    print("Saving...")
    df.to_parquet(OUTPUT_PATH)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()