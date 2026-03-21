import pandas as pd
import numpy as np

from walkforward import walkforward_backtest


INPUT_PATH = "data/processed/baseline_dataset.parquet"


def compute_metrics(df):

    returns = df["strategy_return"].dropna()

    sharpe = np.sqrt(252) * returns.mean() / returns.std()

    equity = (1 + returns).cumprod()

    max_dd = (equity / equity.cummax() - 1).min()

    total_return = equity.iloc[-1] - 1

    return sharpe, max_dd, total_return


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
        'regime_prob_0', 'regime_prob_1',
        'regime_prob_2', 'regime_prob_3'
    ]

    print("Running walk-forward backtest...")
    df_bt = walkforward_backtest(df, FEATURES)

    sharpe, max_dd, total_return = compute_metrics(df_bt)

    print("\nREAL Performance:")
    print(f"Sharpe: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")
    print(f"Total Return: {total_return:.4f}")


if __name__ == "__main__":
    main()