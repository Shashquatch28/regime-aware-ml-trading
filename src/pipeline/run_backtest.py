import pandas as pd

from backtest import run_backtest, compute_metrics

INPUT_PATH = "data/processed/baseline_predictions.parquet"


def main():

    print("Loading predictions...")
    df = pd.read_parquet(INPUT_PATH)

    print("Running backtest...")
    df = run_backtest(df)

    print("\nPerformance Metrics:")
    metrics = compute_metrics(df)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nFinal Equity:", df["equity_curve"].iloc[-1])


if __name__ == "__main__":
    main()