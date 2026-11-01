"""FastDAM and UniverDAM, paper equations (12), (17), (21), (27).

The target kernel is linear. We eliminate each latent decision value exactly
and solve the resulting convex, differentiable epsilon-Huber objective.
This avoids an N x N target Gram matrix, without changing the objective.
The generic precomputed-kernel reference implementation is included below.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.svm import SVR


def domain_weights(mmd2, beta=10000.0, convention="paper_literal"):
    """Normalize Eq. (30) in log space, avoiding floating-point underflow.

    In the supplied PDF, DIST is defined as a squared RKHS norm, but is
    squared again in Eq. (30). 'paper_literal' follows those printed symbols.
    'standard_mmd' treats DIST as the unsquared distance instead.
    """
    mmd2 = np.maximum(np.asarray(mmd2, dtype=float), 0)
    if convention not in {"paper_literal", "standard_mmd"}:
        raise ValueError("Unknown MMD convention")
    return softmax(-beta * (mmd2 ** 2 if convention == "paper_literal" else mmd2))


def epsilon_huber(residual, precision, C=1.0, epsilon=0.1):
    """min_f C|prediction-f|_eps + precision/2*(f-label)^2."""
    t = np.maximum(np.abs(residual) - epsilon, 0)
    cap = C / precision
    quadratic = np.minimum(t, cap)
    value = 0.5 * precision * quadratic ** 2 + C * (t - quadratic)
    derivative = np.sign(residual) * np.minimum(precision * t, C)
    return value, derivative


class LinearEpsilonDAM:
    def __init__(self, C=1.0, epsilon=0.1, tol=1e-7, max_iter=3000):
        self.C, self.epsilon, self.tol, self.max_iter = C, epsilon, tol, max_iter

    def fit(self, X, virtual_labels, precision):
        X = sparse.csr_matrix(X, dtype=np.float64)
        labels = np.asarray(virtual_labels, dtype=np.float64)
        precision = np.broadcast_to(np.asarray(precision, dtype=float), labels.shape).copy()
        if len(labels) != X.shape[0] or not len(labels):
            raise ValueError("Nonempty X and matching labels are required")
        if np.any(precision <= 0) or self.C <= 0 or self.epsilon < 0:
            raise ValueError("Invalid loss parameters")
        XT = X.T.tocsr()

        def objective(v):
            w, b = v[:-1], v[-1]
            residual = X @ w + b - labels
            losses, gradient = epsilon_huber(residual, precision, self.C, self.epsilon)
            value = 0.5 * np.dot(w, w) + losses.sum()
            grad = np.r_[w + XT @ gradient, gradient.sum()]
            return value, grad

        opt = minimize(objective, np.zeros(X.shape[1] + 1), method="L-BFGS-B", jac=True,
                       options={"gtol": self.tol, "ftol": 1e-14,
                                "maxiter": self.max_iter, "maxls": 60, "maxcor": 15})
        # An objective-change stop can precede the requested stationarity
        # tolerance. Restart from the candidate with that stopping rule off.
        if np.max(np.abs(opt.jac)) >= 5e-5:
            refined = minimize(objective, opt.x, method="L-BFGS-B", jac=True,
                               options={"gtol": self.tol, "ftol": 0,
                                        "maxiter": self.max_iter, "maxls": 100, "maxcor": 30})
            refined.nit += opt.nit
            opt = refined
        self.coef_, self.intercept_ = opt.x[:-1].copy(), float(opt.x[-1])
        self.objective_ = float(opt.fun)
        self.gradient_inf_ = float(np.max(np.abs(opt.jac)))
        self.iterations_ = int(opt.nit)
        self.converged_ = bool(self.gradient_inf_ < 1e-4)
        self.solver_message_ = str(opt.message)
        if not self.converged_:
            raise RuntimeError(f"DAM did not converge: gradient={self.gradient_inf_}; {opt.message}")
        _, derivative = epsilon_huber(self.decision_function(X) - labels, precision,
                                      self.C, self.epsilon)
        self.dual_coef_ = -derivative
        self.support_ = np.flatnonzero(np.abs(self.dual_coef_) > 1e-7)
        self.n_train_ = X.shape[0]
        return self

    def decision_function(self, X):
        return np.asarray(X @ self.coef_).ravel() + self.intercept_


class KernelEpsilonDAM:
    """Reference: train epsilon-SVR on K + diag(1/precision).

    Prediction MUST use the original cross-kernel, without the diagonal
    training correction, even for transductive training/evaluation samples.
    sklearn's signed SVR coefficients already have the prediction sign.
    """
    def __init__(self, C=1.0, epsilon=0.1, tol=1e-8):
        self.model = SVR(kernel="precomputed", C=C, epsilon=epsilon, tol=tol)

    def fit(self, K, virtual_labels, precision):
        K = np.asarray(K, dtype=float).copy()
        precision = np.broadcast_to(precision, (K.shape[0],))
        K.flat[::K.shape[0]+1] += 1.0 / precision
        self.model.fit(K, virtual_labels)
        return self

    def decision_function(self, original_cross_kernel):
        return self.model.predict(original_cross_kernel)


def fit_dam(X_target, labeled_indices, observed_labels, source_scores, weights,
            X_sources=None, *, lambda_L=1.0, lambda_D=1.0,
            lambda_U=1.0, threshold=0.3, C=1.0, epsilon=0.1):
    """Fit FastDAM, or UniverDAM when X_sources is supplied.

    Only the labels at labeled_indices are accepted by this API. Hidden target
    labels and source labels for Universum are deliberately not arguments.
    """
    labeled_indices = np.asarray(labeled_indices, dtype=int)
    observed_labels = np.asarray(observed_labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(labeled_indices) != len(observed_labels):
        raise ValueError("Only observed target labels must be passed")
    if len(np.unique(labeled_indices)) != len(labeled_indices):
        raise ValueError("Duplicate labeled indices")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Nonnegative, nonzero source weights are required")
    p = weights.sum()
    virtual = source_scores @ (weights / p)
    unlabeled = np.ones(X_target.shape[0], dtype=bool)
    unlabeled[labeled_indices] = False
    retained = np.flatnonzero(unlabeled & (np.abs(virtual) > threshold))
    indices = np.r_[labeled_indices, retained]
    X_parts = [X_target[indices]]
    labels = [np.r_[observed_labels, virtual[retained]]]
    precision = [np.r_[np.full(len(labeled_indices), lambda_L),
                       np.full(len(retained), p * lambda_D)]]
    if X_sources is not None:
        X_parts.append(X_sources)
        labels.append(np.zeros(X_sources.shape[0]))
        precision.append(np.full(X_sources.shape[0], lambda_U))
    model = LinearEpsilonDAM(C=C, epsilon=epsilon).fit(
        sparse.vstack(X_parts, format="csr"), np.concatenate(labels), np.concatenate(precision))
    model.retained_unlabeled_ = retained
    model.target_fit_indices_ = indices
    model.n_target_support_ = int(np.sum(model.support_ < len(indices)))
    model.n_source_support_ = len(model.support_) - model.n_target_support_
    return model
