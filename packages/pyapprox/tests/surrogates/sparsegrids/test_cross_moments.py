"""Tests for CrossMomentMoments.

Correctness is established against closed-form answers, not against
another quadrature rule. A dense product rule is no more exact than the
class it would be testing once the basis is piecewise --- it cannot see
the breakpoints either --- so agreement between the two would show only
that they fail alike.

The device throughout is to pick a target the grid reproduces exactly.
Then I_K f = f, the surrogate's moments are the target's, and those are
known analytically. Where a second exact route exists (PCEMoments, on
polynomial bases) the two are also compared, since they are independent
derivations of the same quantity.
"""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    ClenshawCurtisGrowthRule,
    LinearGrowthRule,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.statistics.cross_moments import (
    CrossMomentMoments,
)
from pyapprox.surrogates.sparsegrids.statistics.moments import PCEMoments
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)

# E[2x + 3y + 1] and Var[2x + 3y + 1] over [0,1]^2, the latter being
# 4 Var[x] + 9 Var[y] = 13/12. Linear, so every basis here reproduces
# it exactly and the surrogate's moments are these.
_LINEAR_MEAN = 3.5
_LINEAR_VARIANCE = 13.0 / 12.0

# E[x^2 + y^2] = 2/3 and Var[x^2 + y^2] = 8/45 over [0,1]^2. Quadratic,
# so the polynomial bases reproduce it but piecewise-linear does not.
_QUADRATIC_MEAN = 2.0 / 3.0
_QUADRATIC_VARIANCE = 8.0 / 45.0


def _fit(bkd, basis_type, level, target_fn):
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
    factories = create_basis_factories(marginals, bkd, basis_type)
    growth = (
        ClenshawCurtisGrowthRule()
        if basis_type.startswith(("clenshaw", "piecewise"))
        else LinearGrowthRule(scale=1, shift=1)
    )
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    fitter = IsotropicSparseGridFitter(bkd, tp_factory, level)
    result = fitter.fit(target_fn(fitter.get_samples()))
    return result.surrogate, marginals


def _linear(bkd):
    """f = 2x + 3y + 1."""

    def fun(samples):
        return bkd.reshape(2 * samples[0] + 3 * samples[1] + 1, (1, -1))

    return fun


def _sum_of_squares(bkd):
    """f = x^2 + y^2."""

    def fun(samples):
        return bkd.reshape(samples[0] ** 2 + samples[1] ** 2, (1, -1))

    return fun


def _assert_reproduces(bkd, surrogate, target_fn, atol=1e-12):
    """The premise every analytic assertion below rests on."""
    np.random.seed(0)
    samples = bkd.asarray(np.random.uniform(0.0, 1.0, (2, 200)))
    bkd.assert_allclose(
        surrogate(samples), target_fn(samples), atol=atol
    )


class TestAgainstAnalyticMoments:
    """On a target the grid reproduces, the moments are known."""

    @pytest.mark.parametrize(
        "basis_type,level",
        [
            ("gauss", 2),
            ("leja", 2),
            ("clenshaw_curtis", 2),
            ("piecewise_linear", 4),
            ("piecewise_quadratic", 3),
        ],
    )
    def test_linear_target(
        self, bkd, basis_type: str, level: int
    ) -> None:
        """Every basis here reproduces a linear function exactly."""
        surrogate, marginals = _fit(bkd, basis_type, level, _linear(bkd))
        _assert_reproduces(bkd, surrogate, _linear(bkd))

        moments = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd)
        )
        bkd.assert_allclose(
            moments.mean(), bkd.asarray([_LINEAR_MEAN]), rtol=1e-10
        )
        bkd.assert_allclose(
            moments.variance(),
            bkd.asarray([_LINEAR_VARIANCE]),
            rtol=1e-10,
        )

    @pytest.mark.parametrize(
        "basis_type", ["gauss", "leja", "clenshaw_curtis"]
    )
    def test_quadratic_target(self, bkd, basis_type: str) -> None:
        """Polynomial bases reproduce x^2 + y^2 at this level."""
        surrogate, marginals = _fit(
            bkd, basis_type, 3, _sum_of_squares(bkd)
        )
        _assert_reproduces(bkd, surrogate, _sum_of_squares(bkd))

        moments = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd)
        )
        bkd.assert_allclose(
            moments.mean(), bkd.asarray([_QUADRATIC_MEAN]), rtol=1e-10
        )
        bkd.assert_allclose(
            moments.variance(),
            bkd.asarray([_QUADRATIC_VARIANCE]),
            rtol=1e-10,
        )

    def test_constant_target_has_zero_variance(self, bkd) -> None:
        surrogate, marginals = _fit(
            bkd,
            "gauss",
            2,
            lambda s: bkd.full((1, s.shape[1]), 4.25),
        )
        moments = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd)
        )
        bkd.assert_allclose(
            moments.mean(), bkd.asarray([4.25]), rtol=1e-12
        )
        bkd.assert_allclose(
            moments.variance(), bkd.asarray([0.0]), atol=1e-12
        )


class TestAgreesWithTheOtherExactRoute:
    """PCEMoments derives the same quantity a different way."""

    @pytest.mark.parametrize("basis_type", ["gauss", "leja"])
    @pytest.mark.parametrize("level", [2, 3])
    def test_same_moments(
        self, bkd, basis_type: str, level: int
    ) -> None:
        """Held on an under-resolved grid too, where I_K f is not f."""
        surrogate, marginals = _fit(
            bkd,
            basis_type,
            level,
            lambda s: bkd.reshape(
                bkd.cos(3 * s[0]) * bkd.exp(s[1]), (1, -1)
            ),
        )
        cross = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd)
        )
        exact = PCEMoments(surrogate, create_bases_1d(marginals, bkd))
        bkd.assert_allclose(cross.mean(), exact.mean(), rtol=1e-9)
        bkd.assert_allclose(
            cross.variance(), exact.variance(), rtol=1e-9
        )


class TestQuadratureConvergence:
    """Raising extra_points must not move a converged answer."""

    def test_stable_under_extra_points_for_polynomial(self, bkd) -> None:
        """The rule is already exact, so more points change nothing."""
        surrogate, marginals = _fit(
            bkd, "gauss", 3, _sum_of_squares(bkd)
        )
        base = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd), extra_points=0
        ).variance()
        richer = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd), extra_points=20
        ).variance()
        bkd.assert_allclose(base, richer, rtol=1e-12)

    def test_converges_for_piecewise(self, bkd) -> None:
        """No Gauss rule spans a breakpoint, so this one converges.

        Against the analytic value, not another quadrature, and the
        error must fall as the rule is refined.
        """
        surrogate, marginals = _fit(
            bkd, "piecewise_linear", 4, _sum_of_squares(bkd)
        )
        errors = []
        for extra in (0, 10, 60):
            got = CrossMomentMoments(
                surrogate,
                create_bases_1d(marginals, bkd),
                extra_points=extra,
            ).variance()
            errors.append(
                abs(bkd.to_float(got[0]) - _QUADRATIC_VARIANCE)
            )
        assert errors[-1] < errors[0], errors


class TestConstruction:
    """Validation and memoization."""

    def test_memoizes(self, bkd) -> None:
        surrogate, marginals = _fit(bkd, "gauss", 2, _linear(bkd))
        moments = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd)
        )
        assert moments.mean() is moments.mean()
        assert moments.second_moment() is moments.second_moment()

    def test_rejects_non_surrogate(self, bkd) -> None:
        with pytest.raises(TypeError, match="CombinationSurrogate"):
            CrossMomentMoments("not a surrogate", [])

    def test_rejects_wrong_number_of_bases(self, bkd) -> None:
        surrogate, marginals = _fit(bkd, "gauss", 2, _linear(bkd))
        with pytest.raises(ValueError, match="2 variables"):
            CrossMomentMoments(
                surrogate, create_bases_1d(marginals[:1], bkd)
            )

    def test_rejects_negative_extra_points(self, bkd) -> None:
        surrogate, marginals = _fit(bkd, "gauss", 2, _linear(bkd))
        with pytest.raises(ValueError, match="non-negative"):
            CrossMomentMoments(
                surrogate, create_bases_1d(marginals, bkd), extra_points=-1
            )

    def test_surrogate_accessor(self, bkd) -> None:
        surrogate, marginals = _fit(bkd, "gauss", 2, _linear(bkd))
        moments = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd)
        )
        assert moments.surrogate() is surrogate


class TestCoversWhatPCECannot:
    """The reason this class exists."""

    def test_pce_route_rejects_piecewise(self, bkd) -> None:
        surrogate, marginals = _fit(
            bkd, "piecewise_linear", 4, _linear(bkd)
        )
        with pytest.raises(ValueError, match="globally polynomial"):
            PCEMoments(
                surrogate, create_bases_1d(marginals, bkd)
            ).variance()

    def test_cross_moments_handles_it(self, bkd) -> None:
        surrogate, marginals = _fit(
            bkd, "piecewise_linear", 4, _linear(bkd)
        )
        moments = CrossMomentMoments(
            surrogate, create_bases_1d(marginals, bkd)
        )
        bkd.assert_allclose(
            moments.variance(),
            bkd.asarray([_LINEAR_VARIANCE]),
            rtol=1e-10,
        )
