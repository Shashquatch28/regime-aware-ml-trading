import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------
# TRAIN MODELS PER REGIME
# ---------------------------------------------
def train_models_per_regime(df, features):

    direction_models = {}
    meta_models = {}

    regimes = sorted(df["regime_state"].unique())

    for regime in regimes:

        subset = df[df["regime_state"] == regime]

        X = subset[features]

        y_dir = subset["direction"]
        y_meta = subset["meta_label"]

        #print(f"\nRegime {regime} | samples={len(subset)}")

        # Direction model
        dir_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        dir_model.fit(X, y_dir)

        # Meta model
        meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        meta_model.fit(X, y_meta)

        direction_models[regime] = dir_model
        meta_models[regime] = meta_model

    return direction_models, meta_models


# ---------------------------------------------
# PREDICTION (SOFT REGIME WEIGHTING)
# ---------------------------------------------
def predict_with_regimes(df, direction_models, meta_models, features):

    n = len(df)

    direction_pred = np.zeros(n)
    meta_pred = np.zeros(n)

    X = df[features]

    for regime in direction_models.keys():

        # regime probabilities
        pi = df[f"regime_prob_{regime}"].values

        # direction predictions
        dir_model = direction_models[regime]
        dir_probs = dir_model.predict_proba(X)

        # convert to expected direction (-1, +1)
        dir_values = np.zeros(n)

        for i, cls in enumerate(dir_model.classes_):
            if cls == -1:
                dir_values -= dir_probs[:, i]
            elif cls == 1:
                dir_values += dir_probs[:, i]

        # meta predictions (probability of trade)
        meta_model = meta_models[regime]
        meta_probs = meta_model.predict_proba(X)

        # probability of class 1 (trade)
        meta_values = meta_probs[:, list(meta_model.classes_).index(1)]

        # weighted aggregation
        direction_pred += pi * dir_values
        meta_pred += pi * meta_values

    return direction_pred, meta_pred


# ---------------------------------------------
# FINAL SIGNAL
# ---------------------------------------------
def generate_signal(direction_pred, meta_pred, threshold=0.3, dir_threshold=0.1):

    signals = np.zeros(len(direction_pred))

    for i in range(len(signals)):

        if meta_pred[i] > threshold:

            if direction_pred[i] > dir_threshold:
                signals[i] = 1

            elif direction_pred[i] < -dir_threshold:
                signals[i] = -1

            else:
                signals[i] = 0

        else:
            signals[i] = 0

    return signals.astype(int)