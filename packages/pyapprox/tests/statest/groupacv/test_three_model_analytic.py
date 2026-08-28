"""Closed-form validation of the three-model, three-subset estimator covariance.

For a three-model mean-estimation problem with the model subsets
``{0, 1, 2}``, ``{1}`` and ``{2}``, the optimal estimator variance has a closed
form. This test derives that closed form symbolically -- from first principles,
by building the term-wise covariance of the estimator and optimizing its two
control-variate weights -- and compares it against the variance computed
numerically by the estimators:

* :class:`MLBLUEEstimator`, which draws **independent** samples for each
  subset, and
* :class:`GroupACVEstimatorNested`, which draws **nested / shared** samples
  across subsets.

Both variances are read from ``_covariance_from_npartition_samples(...)``, the
entry point that accepts an unrounded partition allocation. Because the
symbolic derivation shares no code with the estimator implementations, it is an
independent oracle: it guards the per-subset covariance (``MLBLUEEstimator``)
and the nested-partition covariance (``GroupACVEstimatorNested``) against a
wholly separate derivation of the same quantity.

Sample-count conventions (model 0 is the high-fidelity model):

* per-model evaluation totals are ``n`` for model 0, ``m1`` for model 1 and
  ``m2`` for model 2;
* independent sampling uses unique low-fidelity sets of size ``m1 - n`` and
  ``m2 - n``;
* nested sampling reuses the ``n`` shared samples in subset ``{1}`` and the
  ``m1`` shared samples in subset ``{2}``.
"""

import functools
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from pyapprox.statest.groupacv import (
    BaseGroupACVEstimator,
    GroupACVEstimatorNested,
    MLBLUEEstimator,
)
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.util.backends.protocols import Array, Backend

# Signature of the lambdified analytic-variance callables: nine positional
# scalars (n, m1, m2, s1, s2, s3, rho12, rho13, rho23) -> estimator variance.
_VarianceFn = Callable[
    [float, float, float, float, float, float, float, float, float], float
]


@functools.lru_cache(maxsize=1)
def _analytic_variance_funcs() -> tuple[_VarianceFn, _VarianceFn]:
    """Symbolically derive the two optimal estimator variances.

    Returns two callables ``(indep_var, nested_var)``, each with signature
    ``f(n, m1, m2, s1, s2, s3, rho12, rho13, rho23)`` returning the optimal
    estimator variance. ``s1, s2, s3`` are the model standard deviations and
    ``rho12 = corr(Q0, Q1)``, ``rho13 = corr(Q0, Q2)``, ``rho23 = corr(Q1, Q2)``
    (model 0 is the high-fidelity model). Cached because the symbolic solve is
    the only slow part of the test.
    """
    import sympy as sp

    s1, s2, s3, rho12, rho23, rho13, a, b = sp.symbols(
        "s1 s2 s3 rho12 rho23 rho13 a b")
    n, m1, m2 = sp.symbols("n m1 m2")

    # sympy ships no type stubs (no py.typed, none on typeshed), so its
    # expression objects are Any to mypy; a precise type is not available here.
    def optimize(estimator_variance: Any) -> Any:
        # Optimal control-variate weights: set the derivatives to zero.
        da = sp.diff(estimator_variance, a)
        a_opt = sp.simplify(sp.solve(da, a)[0])
        var_a_opt = sp.simplify(estimator_variance.subs(a, a_opt))
        db = sp.diff(var_a_opt, b)
        b_opt = sp.simplify(sp.solve(db, b)[0])
        return sp.simplify(var_a_opt.subs(b, b_opt))

    # --- Independent per-subset samples -------------------------------------
    # Subsets {1} and {2} use unique sample sets of size m1 - n and m2 - n.
    n2, n3 = m1 - n, m2 - n
    var_indep = (
        s1**2 / n
        + a**2 * s2**2 / n + a**2 * s2**2 / n2
        + b**2 * s3**2 / n + b**2 * s3**2 / n3
        + 2 * a * rho12 * s1 * s2 / n
        + 2 * b * rho13 * s1 * s3 / n
        + 2 * a * b * rho23 * s2 * s3 / n
    )
    indep_opt = optimize(var_indep)

    # --- Nested / shared samples --------------------------------------------
    # Subset {1} reuses the n shared samples; subset {2} reuses the m1 shared
    # samples, which introduces cross-subset covariance terms.
    n2, n3 = m1, m2
    var_nested = s1**2 / n + a**2 * s2**2 / n + b**2 * s3**2 / n
    var_nested += a**2 * s2**2 / n2 + b**2 * s3**2 / n3
    var_nested += 2 * a * rho12 * s1 * s2 / n + 2 * b * rho13 * s1 * s3 / n
    var_nested += -2 * (a / (n2 * n)) * n * rho12 * s1 * s2
    var_nested += -2 * (b / (n3 * n)) * n * rho13 * s1 * s3
    var_nested += 2 * a * b * rho23 * s2 * s3 / n
    var_nested += -2 * a**2 / (n * n2) * n * s2**2
    var_nested += -2 * a * b / (n * n3) * n * rho23 * s2 * s3
    var_nested += -2 * a * b / (n * n2) * n * rho23 * s2 * s3
    var_nested += -2 * b**2 / (n * n3) * n * s3**2
    var_nested += 2 * a * b / (n2 * n3) * n2 * rho23 * s2 * s3
    nested_opt = optimize(sp.simplify(var_nested))

    args = (n, m1, m2, s1, s2, s3, rho12, rho13, rho23)
    return (sp.lambdify(args, indep_opt, "numpy"),
            sp.lambdify(args, nested_opt, "numpy"))


def _corr_params(
    cov: NDArray[np.float64],
) -> tuple[float, float, float, float, float, float]:
    """(s1, s2, s3, rho12, rho13, rho23) from a 3x3 covariance matrix."""
    s = np.sqrt(np.diag(cov))
    return (s[0], s[1], s[2],
            cov[0, 1] / (s[0] * s[1]),
            cov[0, 2] / (s[0] * s[2]),
            cov[1, 2] / (s[1] * s[2]))


def _numeric_variance(
    estimator_class: type[BaseGroupACVEstimator[Array]],
    cov: NDArray[np.float64],
    npartition_samples: Sequence[float],
    bkd: Backend[Array],
) -> float:
    """Analytic estimator variance at an (unrounded) partition allocation.

    Uses ``_covariance_from_npartition_samples`` directly because the public
    fitted estimator requires integer-typed allocations, whereas the closed
    form is compared at real-valued sample counts.
    """
    costs = bkd.array([1.0, 1.0, 1.0])
    subsets = [
        bkd.array([0, 1, 2], dtype=int),
        bkd.array([1], dtype=int),
        bkd.array([2], dtype=int),
    ]
    stat = MultiOutputMean(1, bkd)
    stat.set_pilot_quantities(bkd.array(cov))
    est = estimator_class(stat, costs, model_subsets=subsets)
    covariance = est._covariance_from_npartition_samples(
        bkd.array([float(v) for v in npartition_samples]))
    return float(np.asarray(covariance).flat[0])


# Two positive-definite correlation structures plus a non-unit-variance case.
_COV_A = np.array([[1.0, 0.95, 0.80],
                   [0.95, 1.0, 0.90],
                   [0.80, 0.90, 1.0]])
_COV_B = np.array([[1.0, 0.95, 0.93],
                   [0.95, 1.0, 0.90],
                   [0.93, 0.90, 1.0]])
# Non-unit standard deviations (2, 1.5, 0.5) with the _COV_A correlations.
_D = np.diag([2.0, 1.5, 0.5])
_COV_SCALED = _D @ _COV_A @ _D

_COVS = {"cov_a": _COV_A, "cov_b": _COV_B, "cov_scaled": _COV_SCALED}


@pytest.mark.parametrize("cov_key", list(_COVS))
@pytest.mark.parametrize("n", [5, 8])
@pytest.mark.parametrize("m1_extra", [5, 20, 60])
@pytest.mark.parametrize("dm", [2, 30, 150])
class TestThreeModelAnalytic:
    """Numeric three-model estimators match the closed-form derivation."""

    def _setup(
        self, cov_key: str, n: int, m1_extra: int, dm: int
    ) -> tuple[NDArray[np.float64], int, int, int]:
        cov = _COVS[cov_key]
        m1 = n + m1_extra
        # Nested requires m2 > m1 > n; independent requires m1, m2 > n.
        m2 = m1 + dm
        return cov, n, m1, m2

    def test_mlblue_matches_analytic(
        self,
        bkd: Backend[Array],
        cov_key: str,
        n: int,
        m1_extra: int,
        dm: int,
    ) -> None:
        cov, n, m1, m2 = self._setup(cov_key, n, m1_extra, dm)
        indep_var, _ = _analytic_variance_funcs()
        expected = float(indep_var(n, m1, m2, *_corr_params(cov)))
        got = _numeric_variance(
            MLBLUEEstimator, cov, [n, m1 - n, m2 - n], bkd)
        bkd.assert_allclose(
            bkd.asarray([got]), bkd.asarray([expected]), rtol=1e-12)

    def test_nested_matches_analytic(
        self,
        bkd: Backend[Array],
        cov_key: str,
        n: int,
        m1_extra: int,
        dm: int,
    ) -> None:
        cov, n, m1, m2 = self._setup(cov_key, n, m1_extra, dm)
        _, nested_var = _analytic_variance_funcs()
        expected = float(nested_var(n, m1, m2, *_corr_params(cov)))
        got = _numeric_variance(
            GroupACVEstimatorNested, cov, [n, m1 - n, m2 - m1], bkd)
        bkd.assert_allclose(
            bkd.asarray([got]), bkd.asarray([expected]), rtol=1e-12)


def test_estimator_variances_differ(bkd: Backend[Array]) -> None:
    """The two sampling structures give different variances at one allocation.

    Guards against a degenerate implementation where both estimators return the
    same value, while still confirming each matches its own closed form.
    """
    indep_var, nested_var = _analytic_variance_funcs()
    n, m1, m2 = 5, 20, 30
    params = _corr_params(_COV_A)

    indep_expected = float(indep_var(n, m1, m2, *params))
    nested_expected = float(nested_var(n, m1, m2, *params))
    indep_got = _numeric_variance(
        MLBLUEEstimator, _COV_A, [n, m1 - n, m2 - n], bkd)
    nested_got = _numeric_variance(
        GroupACVEstimatorNested, _COV_A, [n, m1 - n, m2 - m1], bkd)

    bkd.assert_allclose(
        bkd.asarray([indep_got]), bkd.asarray([indep_expected]), rtol=1e-12)
    bkd.assert_allclose(
        bkd.asarray([nested_got]), bkd.asarray([nested_expected]), rtol=1e-12)
    assert not np.isclose(indep_got, nested_got)
