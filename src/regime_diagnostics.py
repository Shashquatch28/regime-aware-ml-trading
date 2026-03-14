import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------
# REGIME SUMMARY STATISTICS
# ---------------------------------

def regime_summary(df,
                   regime_col="regime_state",
                   return_col="spx_return"):
    
    regimes = sorted(df[regime_col].dropna().unique())
    
    rows = []
    
    for r in regimes:
        
        subset = df[df[regime_col] == r]
        
        stats = {
            "regime": r,
            "count": len(subset),
            "mean_return": subset[return_col].mean(),
            "volatility": subset[return_col].std(),
            "mean_vix": subset["vix_level"].mean(),
            "mean_stress": subset["stress_index"].mean()
        }
        
        rows.append(stats)
    
    return pd.DataFrame(rows)


# ---------------------------------
# TRANSITION MATRIX
# ---------------------------------

def transition_matrix(df, regime_col="regime_state"):
    
    states = df[regime_col].dropna().astype(int)
    
    n_states = states.max() + 1
    
    matrix = np.zeros((n_states, n_states))
    
    for i in range(len(states)-1):
        
        current = states.iloc[i]
        nxt = states.iloc[i+1]
        
        matrix[current, nxt] += 1
    
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    
    return pd.DataFrame(matrix)


# ---------------------------------
# REGIME DURATION
# ---------------------------------

def regime_durations(df, regime_col="regime_state"):
    
    states = df[regime_col].dropna().astype(int)
    
    durations = []
    
    current = states.iloc[0]
    
    length = 1
    
    for i in range(1, len(states)):
        
        if states.iloc[i] == current:
            
            length += 1
        
        else:
            
            durations.append(length)
            
            current = states.iloc[i]
            
            length = 1
    
    durations.append(length)
    
    durations = np.array(durations)
    
    stats = {
        "mean_duration": durations.mean(),
        "median_duration": np.median(durations),
        "max_duration": durations.max(),
        "min_duration": durations.min()
    }
    
    return stats


# ---------------------------------
# REGIME VISUALIZATION
# ---------------------------------

def plot_regimes(df,
                 price_col="spx_price",
                 regime_col="regime_state"):
    
    plt.figure(figsize=(14,6))
    
    scatter = plt.scatter(
        df.index,
        df[price_col],
        c=df[regime_col],
        cmap="tab10",
        s=6
    )
    
    plt.colorbar(scatter,label="Regime")
    
    plt.title("Market Regimes")
    
    plt.xlabel("Time")
    
    plt.ylabel(price_col)
    
    plt.show()
