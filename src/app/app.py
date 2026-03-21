import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from regime_diagnostics import (
    regime_summary,
    transition_matrix,
    regime_durations
)

# -----------------------------
# LOAD DATA
# -----------------------------
st.title("HMM Regime Detection Dashboard")

DATA_PATH = "data/processed/global_regime_dataset.parquet"

df = pd.read_parquet(DATA_PATH)

# -----------------------------
# MODEL INFO
# -----------------------------
st.header("Model Overview")

n_regimes = df["regime_state"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Number of Regimes", n_regimes)
col2.metric("Dataset Size", len(df))
col3.metric("Features Used", 6)

# -----------------------------
# REGIME DISTRIBUTION
# -----------------------------
st.header("Regime Distribution")

counts = df["regime_state"].value_counts().sort_index()

fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Regime")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# -----------------------------
# REGIME SUMMARY TABLE
# -----------------------------
st.header("Regime Summary Statistics")

summary = regime_summary(df)

st.dataframe(summary)

# -----------------------------
# TRANSITION MATRIX
# -----------------------------
st.header("Transition Matrix")

tm = transition_matrix(df)

fig, ax = plt.subplots()
sns.heatmap(tm, annot=True, cmap="Blues", ax=ax)
ax.set_xlabel("Next State")
ax.set_ylabel("Current State")

st.pyplot(fig)

# -----------------------------
# REGIME DURATIONS
# -----------------------------
st.header("Regime Duration Statistics")

dur = regime_durations(df)

st.write(dur)

# -----------------------------
# PRICE WITH REGIMES
# -----------------------------
st.header("Market Regimes Over Time")

price_col = "spx_price" if "spx_price" in df.columns else df.columns[0]

fig, ax = plt.subplots(figsize=(12, 5))

scatter = ax.scatter(
    df.index,
    df[price_col],
    c=df["regime_state"],
    cmap="tab10",
    s=5
)

plt.colorbar(scatter, label="Regime")
ax.set_title("Price Colored by Regime")
ax.set_xlabel("Time")
ax.set_ylabel("Price")

st.pyplot(fig)

# -----------------------------
# REGIME PROBABILITIES
# -----------------------------
st.header("Regime Probabilities Over Time")

prob_cols = [c for c in df.columns if "regime_prob" in c]

fig, ax = plt.subplots(figsize=(12, 5))

df[prob_cols].plot(ax=ax)

ax.set_title("Regime Probabilities")
ax.set_ylabel("Probability")

st.pyplot(fig)