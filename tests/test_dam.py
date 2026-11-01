import numpy as np
from scipy import sparse
from scipy.optimize import check_grad
from src.dam import LinearEpsilonDAM, KernelEpsilonDAM, domain_weights, epsilon_huber, fit_dam
from src.data import sample_target


def test_linear_objective_matches_independent_kernel_svr():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(31, 8))
    y = rng.normal(size=31)
    a = np.exp(rng.normal(size=31))
    test = rng.normal(size=(13, 8))
    primal = LinearEpsilonDAM(C=.8, epsilon=.13).fit(X, y, a)
    dual = KernelEpsilonDAM(C=.8, epsilon=.13).fit(X @ X.T, y, a)
    np.testing.assert_allclose(primal.decision_function(test),
                               dual.decision_function(test @ X.T), atol=3e-5)
    np.testing.assert_allclose(primal.decision_function(X),
                               dual.decision_function(X @ X.T), atol=3e-5)
    assert abs(primal.dual_coef_.sum()) < 1e-5
    np.testing.assert_allclose(X.T @ primal.dual_coef_, primal.coef_, atol=1e-5)


def test_huber_gradient_against_finite_differences():
    r = np.array([-2., -.52, -.03, .2, .7, 2.3])
    a = np.array([.5, .7, 1, 1.5, 2, 3])
    error = check_grad(lambda v: epsilon_huber(v, a)[0].sum(),
                       lambda v: epsilon_huber(v, a)[1], r)
    assert error < 1e-6


def test_unsupervised_fit_and_universum_limit():
    rng = np.random.default_rng(3)
    X = sparse.csr_matrix(rng.normal(size=(25, 5)))
    S = sparse.csr_matrix(rng.normal(size=(10, 5)))
    scores = np.c_[np.linspace(-1, 1, 25), np.linspace(-.8, .9, 25)]
    fast = fit_dam(X, [], [], scores, [.4, .6])
    weak = fit_dam(X, [], [], scores, [.4, .6], S, lambda_U=1e-9)
    np.testing.assert_allclose(fast.decision_function(X), weak.decision_function(X), atol=2e-4)
    assert fast.n_source_support_ == 0
    assert np.all(np.abs(scores[fast.retained_unlabeled_] @ np.array([.4, .6])) > .3)


def test_weights_stable_and_balanced_reproducible_labels():
    weights = domain_weights([10., 11., 12.])
    assert np.isfinite(weights).all()
    assert abs(weights.sum() - 1) < 1e-12
    assert weights.argmax() == 0
    y = np.r_[np.ones(30), -np.ones(30)]
    a, b = sample_target(y, 4, 42), sample_target(y, 4, 42)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 8 and y[a].sum() == 0
    assert len(sample_target(y, 0, 42)) == 0
