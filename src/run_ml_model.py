import pandas as pd

from ml_models import (
    train_regime_models,
    predict_proba_with_regimes,
    convert_probs_to_signal
)

INPUT_PATH = "data/processed/final_model_dataset.parquet"
OUTPUT_PATH = "data/processed/ml_predictions.parquet"


def main():

    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    # -----------------------------
    # FEATURES
    # -----------------------------
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
        'regime_prob_2', 'regime_prob_3',

        # NEW FEATURES
        'delta_regime_prob_0', 'delta_regime_prob_1',
        'delta_regime_prob_2', 'delta_regime_prob_3',
        'regime_persistence'
    ]

    print("Training regime models...")
    models = train_regime_models(df, FEATURES)

    print("Generating predictions...")
    probs = predict_proba_with_regimes(df, models, FEATURES)

    print("Converting to signals...")
    signals = convert_probs_to_signal(probs)

    df["signal"] = signals

    print("\nSignal distribution:")
    print(df["signal"].value_counts(normalize=True))

    print("Saving results...")
    df.to_parquet(OUTPUT_PATH)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()