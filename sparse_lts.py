"""
Sparse Least Trimmed Squares (Sparse LTS).

Implements the concentration-step (C-step) algorithm for Least Trimmed
Squares regression, combined with an L1 (Lasso) penalty for simultaneous
variable selection, following the general approach described in Alfons,
Croux & Gelper (2013), "Sparse Least Trimmed Squares Regression for
Analyzing High-Dimensional Large Data Sets," Annals of Applied Statistics.

Least Trimmed Squares achieves robustness to outliers by fitting the model
to minimize the sum of only the h smallest squared residuals (h < n),
rather than all n residuals as in ordinary least squares. This means a
minority of extreme, contaminated observations (e.g., sensor glitches or
transient anomalies in industrial telemetry) cannot dominate the fit.
"""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, RegressorMixin


class SparseLTS(BaseEstimator, RegressorMixin):
    """Sparse LTS via iterative concentration steps with an L1-penalized fit.

    Parameters
    ----------
    alpha : float
        L1 regularization strength passed to the inner Lasso fit at each
        concentration step.
    trim_fraction : float
        Fraction of observations trimmed as potential outliers (h = n * (1 - trim_fraction)
        observations are kept at each concentration step).
    n_starts : int
        Number of random elemental starts, to reduce sensitivity to local optima.
    max_iter : int
        Maximum concentration steps per start.
    """

    def __init__(self, alpha=1.0, trim_fraction=0.25, n_starts=10, max_iter=30, random_state=0):
        self.alpha = alpha
        self.trim_fraction = trim_fraction
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        h = int(np.ceil(n * (1 - self.trim_fraction)))
        rng = np.random.RandomState(self.random_state)

        best_obj = np.inf
        best_model = None
        best_subset = None

        for _ in range(self.n_starts):
            # Random elemental start: fit on a random subset, then concentrate.
            idx = rng.choice(n, size=h, replace=False)
            for _ in range(self.max_iter):
                model = Lasso(alpha=self.alpha, max_iter=5000)
                model.fit(X[idx], y[idx])
                residuals_sq = (y - model.predict(X)) ** 2
                new_idx = np.argsort(residuals_sq)[:h]
                if set(new_idx.tolist()) == set(idx.tolist()):
                    break
                idx = new_idx
            obj = np.sum(np.sort((y - model.predict(X)) ** 2)[:h])
            if obj < best_obj:
                best_obj = obj
                best_model = model
                best_subset = idx

        self.model_ = best_model
        self.trimmed_subset_ = best_subset
        self.coef_ = best_model.coef_
        self.intercept_ = best_model.intercept_
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))
