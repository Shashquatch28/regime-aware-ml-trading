import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------
# TRAIN MODELS PER REGIME
# ---------------------------------------------
def train_regime_models(df, features, target_col="target"):
    """
    Train one model per regime
    """

    models = {}

    regimes = sorted(df["regime_state"].unique())

    for regime in regimes:

        subset = df[df["regime_state"] == regime]

        X = subset[features]
        y = subset[target_col]

        print(f"Training model for regime {regime} | samples={len(subset)}")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1  # use all cores
        )

        model.fit(X, y)

        models[regime] = model

    return models


# ---------------------------------------------
# FAST VECTORIZED SOFT ENSEMBLE
# ---------------------------------------------
def predict_proba_with_regimes(df, models, features):
    """
    Vectorized soft ensemble using regime probabilities
    """

    n = len(df)

    # final probabilities: columns correspond to [-1, 0, 1]
    prob_agg = np.zeros((n, 3))

    X = df[features]

    for regime, model in models.items():

        # regime probabilities (N x 1)
        regime_probs = df[f"regime_prob_{regime}"].values.reshape(-1, 1)

        # model predictions (N x num_classes_present)
        probs = model.predict_proba(X)

        # align to fixed class order [-1, 0, 1]
        aligned = np.zeros((n, 3))

        for i, cls in enumerate(model.classes_):
            idx = {-1: 0, 0: 1, 1: 2}[cls]
            aligned[:, idx] = probs[:, i]

        # weighted contribution
        prob_agg += regime_probs * aligned

    return prob_agg


# ---------------------------------------------
# CONVERT PROBABILITIES → SIGNALS
# ---------------------------------------------
def convert_probs_to_signal(probs, min_confidence=0.05):
    """
    Argmax-based decision with confidence filter
    """

    signals = np.zeros(len(probs))

    for i in range(len(probs)):

        p = probs[i]

        best_class = np.argmax(p)  # 0=short, 1=hold, 2=long
        confidence = p[best_class] - np.sort(p)[-2]  # gap between top 2

        if confidence < min_confidence:
            signals[i] = 0
        else:
            if best_class == 2:
                signals[i] = 1
            elif best_class == 0:
                signals[i] = -1
            else:
                signals[i] = 0

    return signals.astype(int)