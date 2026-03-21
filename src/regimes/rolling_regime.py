import pandas as pd
import numpy as np
import warnings

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from hmmlearn.base import ConvergenceMonitor

from regime_model import (
    HMMRegimeModel,
    match_regimes,
    reorder_hmm_parameters
)

# suppress hmmlearn convergence spam
warnings.filterwarnings("ignore", message="Model is not converging")


def rolling_hmm_regimes(df, features, window=750):

    X = df[features].values
    dates = df.index

    regime_states = []
    regime_probs = []
    result_dates = []

    prev_templates = None

    total_steps = len(df) - window

    for t in tqdm(
        range(window, len(df)),
        total=total_steps,
        desc="Rolling HMM",
        ncols=80
    ):

        X_train = X[t-window:t]

        # rolling scaling (no leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = HMMRegimeModel(n_components=4)

        try:
            model.fit(X_train_scaled)

        except Exception as e:

            # rare message only
            print(f"\nHMM failed at index {t}: {e}")
            continue

        # regime alignment
        if prev_templates is not None:

            new_templates = model.get_templates()

            mapping = match_regimes(prev_templates, new_templates)

            model.model = reorder_hmm_parameters(model.model, mapping)

        prev_templates = model.get_templates()

        # prediction
        X_pred = scaler.transform(X[t:t+1])

        prob = model.predict_proba(X_pred)[0]

        state = np.argmax(prob)

        regime_states.append(state)
        regime_probs.append(prob)
        result_dates.append(dates[t])

    prob_cols = [f"regime_prob_{i}" for i in range(model.n_components)]

    results = pd.DataFrame(
        regime_probs,
        index=result_dates,
        columns=prob_cols
    )

    results["regime_state"] = regime_states

    return results