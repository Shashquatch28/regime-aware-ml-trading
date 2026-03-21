import pandas as pd
import numpy as np

from ml_models_baseline import (
    train_models_per_regime,
    predict_with_regimes,
    generate_signal
)


def walkforward_backtest(
    df,
    features,
    train_window=2000,
    retrain_freq=50
):
    """
    Walk-forward backtest with periodic retraining
    """

    results = []

    dir_models = None
    meta_models = None

    for t in range(train_window, len(df)):

        train_df = df.iloc[t - train_window:t]
        test_row = df.iloc[t:t + 1]

        # ---------------------------------------------
        # Retrain models periodically
        # ---------------------------------------------
        if (t - train_window) % retrain_freq == 0:

            print(f"Retraining at step {t}...")

            dir_models, meta_models = train_models_per_regime(
                train_df,
                features
            )

        # ---------------------------------------------
        # Predict next step
        # ---------------------------------------------
        direction_pred, meta_pred = predict_with_regimes(
            test_row,
            dir_models,
            meta_models,
            features
        )

        signal = generate_signal(direction_pred, meta_pred)[0]

        results.append({
            "date": test_row.index[0],
            "signal": signal,
            "return": test_row["spx_return"].values[0]
        })

    # ---------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------
    result_df = pd.DataFrame(results).set_index("date")

    # ---------------------------------------------
    # Lag position (NO LOOKAHEAD)
    # ---------------------------------------------
    result_df["position"] = result_df["signal"].shift(1)

    # ---------------------------------------------
    # Strategy return
    # ---------------------------------------------
    result_df["strategy_return"] = (
        result_df["position"] * result_df["return"]
    )

    return result_df