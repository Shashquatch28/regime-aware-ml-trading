import pandas as pd
from sklearn.preprocessing import StandardScaler

from regime_model import HMMRegimeModel


FEATURES = [
    "spx_return",
    "spx_vol_20",
    "spx_vol_60",
    "vix_level",
    "stress_index",
    "us10y_vol"
]


def main():

    print("Loading dataset")

    df = pd.read_parquet("data/processed/regime_features.parquet")

    X = df[FEATURES]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("Training global HMM")

    model = HMMRegimeModel(n_components=4)

    model.fit(X_scaled)

    print("Predicting regimes")

    states = model.predict_states(X_scaled)

    probs = model.predict_proba(X_scaled)

    df["regime_state"] = states

    for i in range(probs.shape[1]):
        df[f"regime_prob_{i}"] = probs[:, i]

    print("Saving dataset")

    df.to_parquet("data/processed/global_regime_dataset.parquet")


if __name__ == "__main__":
    main()