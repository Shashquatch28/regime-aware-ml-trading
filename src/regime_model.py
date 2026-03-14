import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment


class HMMRegimeModel:
    
    def __init__(self, n_components=3, covariance_type="full", random_state=42):
        
        self.n_components = n_components
        
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_iter=200,
            min_covar=1e-6
        )
    
    
    def fit(self, X):
        
        self.model.fit(X)
        
        return self
    
    
    def predict_states(self, X):
        
        return self.model.predict(X)
    
    
    def predict_proba(self, X):
        
        return self.model.predict_proba(X)
    
    
    def score(self, X):
        
        return self.model.score(X)
    
    
    def get_templates(self):
        
        means = self.model.means_
        covars = self.model.covars_
        
        return means, covars


# -----------------------------
# BIC MODEL SELECTION
# -----------------------------


def bic_for_hmm(model, X):
    
    n = model.n_components
    d = X.shape[1]
    N = X.shape[0]
    
    log_likelihood = model.score(X)
    
    p = (
        n * (n - 1) +
        (n - 1) +
        n * d +
        n * d * (d + 1) / 2
    )
    
    bic = -2 * log_likelihood + p * np.log(N)
    
    return bic


def select_hmm_model(X, k_candidates=(2,3,4)):
    
    best_model = None
    best_bic = np.inf
    best_k = None
    
    for k in k_candidates:
        
        model = HMMRegimeModel(n_components=k)
        
        model.fit(X)
        
        bic = bic_for_hmm(model.model, X)
        
        if bic < best_bic:
            
            best_bic = bic
            best_model = model
            best_k = k
    
    return best_model, best_k, best_bic


# -----------------------------
# WASSERSTEIN DISTANCE
# -----------------------------


def gaussian_wasserstein_distance(mean1, cov1, mean2, cov2):
    
    mean_term = np.sum((mean1 - mean2) ** 2)
    
    sqrt_cov1 = sqrtm(cov1)
    
    inner = sqrt_cov1 @ cov2 @ sqrt_cov1
    
    sqrt_inner = sqrtm(inner)
    
    trace_term = np.trace(cov1 + cov2 - 2 * sqrt_inner)
    
    return np.real(mean_term + trace_term)


# -----------------------------
# REGIME MATCHING
# -----------------------------


def match_regimes(prev_templates, new_templates):
    
    prev_means, prev_covs = prev_templates
    new_means, new_covs = new_templates
    
    k = prev_means.shape[0]
    
    cost_matrix = np.zeros((k, k))
    
    for i in range(k):
        
        for j in range(k):
            
            cost_matrix[i, j] = gaussian_wasserstein_distance(
                prev_means[i],
                prev_covs[i],
                new_means[j],
                new_covs[j]
            )
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    mapping = dict(zip(col_ind, row_ind))
    
    return mapping


# -----------------------------
# PARAMETER REORDERING
# -----------------------------


def reorder_hmm_parameters(model, mapping):
    
    order = [mapping[i] for i in sorted(mapping.keys())]
    
    model.means_ = model.means_[order]
    
    model.covars_ = model.covars_[order]
    
    model.startprob_ = model.startprob_[order]
    
    model.transmat_ = model.transmat_[order][:, order]
    
    return model
