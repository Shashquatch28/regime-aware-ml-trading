# Hidden Markov Model regime detection with BIC-based model selection, Wasserstein regime matching, and parameter reordering

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment


class HMMRegimeModel:

    def __init__(self, n_components=3, covariance_type="full", random_state=42):
        # Initialize Gaussian HMM with specified number of regimes and covariance structure
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_iter=200,
            min_covar=1e-6
        )

    def fit(self, X):
        # Fit the HMM to the input feature matrix X
        self.model.fit(X)
        return self

    def predict_states(self, X):
        # Return the most likely regime state sequence for X using Viterbi decoding
        return self.model.predict(X)

    def predict_proba(self, X):
        # Return the posterior probability of each regime state for every timestep
        return self.model.predict_proba(X)

    def score(self, X):
        # Return the log-likelihood of the data under the fitted model
        return self.model.score(X)

    def get_templates(self):
        # Extract the learned regime mean vectors and covariance matrices
        means = self.model.means_
        covars = self.model.covars_
        return means, covars


def bic_for_hmm(model, X):
    # Compute BIC score for a fitted HMM: penalizes model complexity to prevent overfitting
    n = model.n_components
    d = X.shape[1]
    N = X.shape[0]

    log_likelihood = model.score(X)

    # Count free parameters: transition probs + initial probs + means + covariances
    p = (
        n * (n - 1) +
        (n - 1) +
        n * d +
        n * d * (d + 1) / 2
    )

    bic = -2 * log_likelihood + p * np.log(N)
    return bic


def select_hmm_model(X, k_candidates=(2, 3, 4)):
    # Fit HMMs for each candidate number of regimes and return the one with the lowest BIC
    best_model = None
    best_bic = np.inf
    best_k = None

    for k in k_candidates:
        model = HMMRegimeModel(n_components=k)
        model.fit(X)
        bic = bic_for_hmm(model.model, X)

        # Keep track of the best model seen so far
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_k = k

    return best_model, best_k, best_bic


def gaussian_wasserstein_distance(mean1, cov1, mean2, cov2):
    # Compute the 2-Wasserstein distance between two Gaussian distributions
    # Used to measure how similar two regime templates are across model updates

    # Squared Euclidean distance between means
    mean_term = np.sum((mean1 - mean2) ** 2)

    # Trace term based on matrix square roots of covariances
    sqrt_cov1 = sqrtm(cov1)
    inner = sqrt_cov1 @ cov2 @ sqrt_cov1
    sqrt_inner = sqrtm(inner)
    trace_term = np.trace(cov1 + cov2 - 2 * sqrt_inner)

    return np.real(mean_term + trace_term)


def match_regimes(prev_templates, new_templates):
    # Match new regime states to previous ones using the Hungarian algorithm on Wasserstein distances
    # Ensures regime labels stay consistent across model refits
    prev_means, prev_covs = prev_templates
    new_means, new_covs = new_templates
    k = prev_means.shape[0]

    # Build cost matrix of pairwise Wasserstein distances between old and new regimes
    cost_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost_matrix[i, j] = gaussian_wasserstein_distance(
                prev_means[i], prev_covs[i],
                new_means[j], new_covs[j]
            )

    # Solve the optimal assignment problem to find the best regime label mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = dict(zip(col_ind, row_ind))

    return mapping


def reorder_hmm_parameters(model, mapping):
    # Reorder HMM internal parameters (means, covariances, transitions) to match the regime mapping
    order = [mapping[i] for i in sorted(mapping.keys())]

    model.means_ = model.means_[order]

    covars = model.covars_[order]

    # Enforce symmetry and numerical positive definiteness after reordering
    for i in range(len(covars)):
        cov = covars[i]
        cov = (cov + cov.T) / 2
        cov += np.eye(cov.shape[0]) * 1e-6
        covars[i] = cov

    model.covars_ = covars
    model.startprob_ = model.startprob_[order]

    # Reorder both rows and columns of the transition matrix to maintain consistency
    model.transmat_ = model.transmat_[order][:, order]

    return model
