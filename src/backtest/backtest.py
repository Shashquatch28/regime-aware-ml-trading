import pandas as pd
import numpy as np


def run_backtest(df, cost_rate=0.0005):
    """
    Simple backtest with transaction costs
    """

    df = df.copy()

    # -----------------------------
    # Position = signal
    # -----------------------------
    df["position"] = df["signal"]

    # -----------------------------
    # Lag position (no lookahead)
    # -----------------------------
    df["position_lag"] = df["position"].shift(1)

    # -----------------------------
    # Strategy returns
    # -----------------------------
    df["strategy_return"] = df["position_lag"] * df["spx_return"]

    # -----------------------------
    # Turnover
    # -----------------------------
    df["turnover"] = (df["position"] - df["position_lag"]).abs()

    # -----------------------------
    # Transaction cost
    # -----------------------------
    df["cost"] = cost_rate * df["turnover"]

    # -----------------------------
    # Net return
    # -----------------------------
    df["net_return"] = df["strategy_return"] - df["cost"]

    # -----------------------------
    # Equity curve
    # -----------------------------
    df["equity_curve"] = (1 + df["net_return"]).cumprod()

    return df


def compute_metrics(df):

    returns = df["net_return"]

    sharpe = np.sqrt(252) * returns.mean() / returns.std()

    max_dd = (df["equity_curve"] / df["equity_curve"].cummax() - 1).min()

    total_return = df["equity_curve"].iloc[-1] - 1

    return {
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Total Return": total_return
    }