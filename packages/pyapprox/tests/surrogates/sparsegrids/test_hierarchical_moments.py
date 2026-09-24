"""Tests for HierarchicalMoments.

The hierarchical mean is the inner product of the surpluses with the
hierarchical quadrature weights, both of which the surrogate already
holds. Tested against that definition and against analytic values on
targets the basis reproduces exactly.
"""

import pytest
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.surrogates.affine.indices.admissibility import (
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids.basis.hierarchical_basis_1d import (
    HierarchicalBasis1D,
)
from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_fitter import (
    SingleFidelityHierarchicalFitter,
)
from pyapprox.surrogates.sparsegrids.statistics.hierarchical_moments import (
    HierarchicalMoments,
)


def _fit(bkd, fun, nvars=1, p_max=2, max_level=3, max_steps=50):
    bases_1d = [
        HierarchicalBasis1D(bkd, p_max=p_max, boundary_mode="include")
        for _ in range(nvars)
    ]
    admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=bkd)
    fitter = SingleFidelityHierarchicalFitter(bkd, bases_1d, admis)
    return fitter.refine_to_tolerance(
        FunctionFromCallable(1, nvars, fun, bkd),
        tol=1e-15,
        max_steps=max_steps,
    ).surrogate


class TestHierarchicalMoments:
    """mean = surpluses . quad_weights."""

    def test_matches_its_definition(self, bkd) -> None:
        """Against the inner product computed in the test."""
        surrogate = _fit(bkd, lambda x: x**2)
        expected = bkd.dot(
            surrogate.surpluses(), surrogate.quad_weights()
        )
        bkd.assert_allclose(
            HierarchicalMoments(surrogate).mean(), expected, rtol=1e-12
        )

    @pytest.mark.parametrize("p_max", [1, 2])
    def test_mean_of_linear(self, bkd, p_max: int) -> None:
        """E[x] = 1/2 on [0,1]."""
        surrogate = _fit(bkd, lambda x: x, p_max=p_max, max_level=2)
        bkd.assert_allclose(
            HierarchicalMoments(surrogate).mean(),
            bkd.asarray([0.5]),
            atol=1e-14,
        )

    def test_mean_of_quadratic(self, bkd) -> None:
        """E[x^2] = 1/3 on [0,1], exact once p_max is 2."""
        surrogate = _fit(bkd, lambda x: x**2, p_max=2)
        bkd.assert_allclose(
            HierarchicalMoments(surrogate).mean(),
            bkd.asarray([1.0 / 3.0]),
            atol=1e-13,
        )

    def test_mean_in_two_dimensions(self, bkd) -> None:
        """E[x + y] = 1 on [0,1]^2."""

        def fun(x):
            return x[0:1, :] + x[1:2, :]

        surrogate = _fit(bkd, fun, nvars=2, p_max=1, max_level=3)
        bkd.assert_allclose(
            HierarchicalMoments(surrogate).mean(),
            bkd.asarray([1.0]),
            atol=1e-13,
        )

    def test_memoizes(self, bkd) -> None:
        surrogate = _fit(bkd, lambda x: x**2)
        moments = HierarchicalMoments(surrogate)
        assert moments.mean() is moments.mean()

    def test_rejects_non_surrogate(self, bkd) -> None:
        with pytest.raises(TypeError, match="HierarchicalSurrogate"):
            HierarchicalMoments("not a surrogate")

    def test_surrogate_accessor(self, bkd) -> None:
        surrogate = _fit(bkd, lambda x: x**2)
        assert HierarchicalMoments(surrogate).surrogate() is surrogate
