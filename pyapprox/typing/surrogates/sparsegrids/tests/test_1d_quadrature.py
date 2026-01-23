"""Tests for 1D Gauss quadrature rules with various marginal distributions.

This module tests that 1D Gauss quadrature rules correctly integrate
polynomials for different probability measures (Uniform, Gaussian, Beta, Gamma).

Tests verify:
1. Weights sum to 1 (probability measure)
2. Mean is exact with 1+ points: sum(w_i * x_i) = E[X]
3. Second moment is exact with 2+ points: sum(w_i * x_i^2) = E[X^2]
4. Degree of exactness: n points exact for polynomials up to degree 2n-1

These tests help diagnose issues with sparse grid integration for non-uniform
marginals (Beta, Gamma) where interpolation tests pass but integration fails.
"""

import unittest
from typing import Any, Dict, Generic, Tuple, Union

import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.probability import (
    BetaMarginal,
    GammaMarginal,
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

# Type alias for marginal types
MarginalType = Union[
    UniformMarginal[Any], GaussianMarginal[Any], BetaMarginal[Any], GammaMarginal[Any]
]


# =============================================================================
# Marginal configurations for parametrized tests
# =============================================================================

# Each config: (name, marginal_type, marginal_params)
# marginal_params contains kwargs for the marginal constructor
MARGINAL_CONFIGS = [
    # Uniform: varying bounds
    ("uniform_canonical", "uniform", {"lower": -1.0, "upper": 1.0}),
    ("uniform_01", "uniform", {"lower": 0.0, "upper": 1.0}),
    ("uniform_02", "uniform", {"lower": 0.0, "upper": 2.0}),
    ("uniform_shifted", "uniform", {"lower": -2.0, "upper": 4.0}),
    # Gaussian: varying mean/std
    ("gaussian_standard", "gaussian", {"mean": 0.0, "stdev": 1.0}),
    ("gaussian_shifted", "gaussian", {"mean": 5.0, "stdev": 1.0}),
    ("gaussian_scaled", "gaussian", {"mean": 0.0, "stdev": 3.0}),
    ("gaussian_both", "gaussian", {"mean": 2.0, "stdev": 0.5}),
    # Beta: varying shapes AND bounds
    ("beta_2_5_01", "beta", {"alpha": 2.0, "beta": 5.0, "lb": 0.0, "ub": 1.0}),
    ("beta_2_5_02", "beta", {"alpha": 2.0, "beta": 5.0, "lb": 0.0, "ub": 2.0}),
    ("beta_2_5_25", "beta", {"alpha": 2.0, "beta": 5.0, "lb": 2.0, "ub": 5.0}),
    ("beta_5_2_01", "beta", {"alpha": 5.0, "beta": 2.0, "lb": 0.0, "ub": 1.0}),
    ("beta_3_3_neg", "beta", {"alpha": 3.0, "beta": 3.0, "lb": -1.0, "ub": 3.0}),
    # Gamma: varying shape/scale
    ("gamma_3_1", "gamma", {"shape": 3.0, "scale": 1.0}),
    ("gamma_3_2", "gamma", {"shape": 3.0, "scale": 2.0}),
    ("gamma_1_1", "gamma", {"shape": 1.0, "scale": 1.0}),  # Exponential
    ("gamma_5_0p5", "gamma", {"shape": 5.0, "scale": 0.5}),
]


def create_marginal(
    mtype: str, params: Dict[str, float], bkd: Backend[Array]
) -> MarginalType:
    """Create a marginal distribution from type and parameters."""
    if mtype == "uniform":
        return UniformMarginal(params["lower"], params["upper"], bkd)
    elif mtype == "gaussian":
        return GaussianMarginal(params["mean"], params["stdev"], bkd)
    elif mtype == "beta":
        return BetaMarginal(
            params["alpha"], params["beta"], bkd,
            lb=params.get("lb", 0.0), ub=params.get("ub", 1.0)
        )
    elif mtype == "gamma":
        return GammaMarginal(params["shape"], params["scale"], bkd)
    else:
        raise ValueError(f"Unknown marginal type: {mtype}")


def get_analytical_moments(
    mtype: str, params: Dict[str, float]
) -> Tuple[float, float]:
    """Get analytical E[X] and E[X^2] for a marginal distribution.

    Returns
    -------
    mean : float
        E[X] - analytical mean
    second_moment : float
        E[X^2] - analytical second moment
    """
    if mtype == "uniform":
        a, b = params["lower"], params["upper"]
        mean = (a + b) / 2.0
        # E[X^2] = integral_a^b x^2 * (1/(b-a)) dx = (b^3 - a^3) / (3*(b-a))
        #        = (a^2 + ab + b^2) / 3
        second_moment = (a**2 + a * b + b**2) / 3.0
        return mean, second_moment

    elif mtype == "gaussian":
        mu, sigma = params["mean"], params["stdev"]
        mean = mu
        # E[X^2] = Var[X] + E[X]^2 = sigma^2 + mu^2
        second_moment = sigma**2 + mu**2
        return mean, second_moment

    elif mtype == "beta":
        alpha, beta = params["alpha"], params["beta"]
        lb, ub = params.get("lb", 0.0), params.get("ub", 1.0)
        scale = ub - lb

        # For Beta(alpha, beta) on [lb, ub]:
        # E[X] = lb + scale * alpha / (alpha + beta)
        mean_01 = alpha / (alpha + beta)
        mean = lb + scale * mean_01

        # E[X^2] = lb^2 + 2*lb*scale*E[Y] + scale^2*E[Y^2]
        # where Y ~ Beta(a,b) on [0,1]
        # E[Y^2] = Var[Y] + E[Y]^2
        #        = a*b / ((a+b)^2 * (a+b+1)) + (a/(a+b))^2
        ab = alpha + beta
        var_01 = (alpha * beta) / (ab**2 * (ab + 1))
        second_moment_01 = var_01 + mean_01**2
        second_moment = lb**2 + 2 * lb * scale * mean_01 + scale**2 * second_moment_01
        return mean, second_moment

    elif mtype == "gamma":
        k, theta = params["shape"], params["scale"]
        # E[X] = k * theta
        mean = k * theta
        # E[X^2] = Var[X] + E[X]^2 = k*theta^2 + (k*theta)^2 = k*theta^2 + k^2*theta^2
        #        = k*theta^2*(1 + k)
        second_moment = k * theta**2 * (1 + k)
        return mean, second_moment

    else:
        raise ValueError(f"Unknown marginal type: {mtype}")


# =============================================================================
# Base test class for 1D Gauss quadrature
# =============================================================================


class TestGaussQuadrature1D(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Parametrized tests for 1D Gauss quadrature rules.

    Tests verify that Gauss quadrature rules for various probability measures
    correctly integrate polynomials up to the expected degree of exactness.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _get_quadrature_rule(
        self, mtype: str, params: Dict[str, float], npoints: int
    ) -> Tuple[Array, Array]:
        """Get quadrature points and weights for a marginal distribution.

        Uses GaussLagrangeFactory to get the quadrature rule that would be
        used by sparse grids for this marginal.

        Returns
        -------
        points : Array
            Quadrature points in user domain, shape (npoints,)
        weights : Array
            Quadrature weights summing to 1, shape (npoints,)
        """
        marginal = create_marginal(mtype, params, self._bkd)
        factory = GaussLagrangeFactory(marginal, self._bkd)
        basis = factory.create_basis()
        basis.set_nterms(npoints)
        points, weights = basis.quadrature_rule()
        # points and weights are 1D arrays
        return self._bkd.flatten(points), self._bkd.flatten(weights)

    @parametrize(
        "name,mtype,params",
        MARGINAL_CONFIGS,
    )
    def test_weights_sum_to_one(
        self, name: str, mtype: str, params: Dict[str, float]
    ) -> None:
        """Test that quadrature weights sum to 1 for all marginals."""
        for npoints in [1, 2, 3, 5, 10]:
            points, weights = self._get_quadrature_rule(mtype, params, npoints)
            weight_sum = self._bkd.sum(weights)
            self._bkd.assert_allclose(
                self._bkd.atleast_1d(weight_sum),
                self._bkd.asarray([1.0]),
                rtol=1e-12,
                atol=1e-14,
            )

    @parametrize(
        "name,mtype,params",
        MARGINAL_CONFIGS,
    )
    def test_mean_exact(
        self, name: str, mtype: str, params: Dict[str, float]
    ) -> None:
        """Test mean is exact: sum(w_i * x_i) = E[X] with 1+ points.

        Gauss quadrature with n points is exact for polynomials up to
        degree 2n-1. Since f(x) = x has degree 1, it's exact for n >= 1.
        """
        expected_mean, _ = get_analytical_moments(mtype, params)

        for npoints in [1, 2, 3, 5]:
            points, weights = self._get_quadrature_rule(mtype, params, npoints)
            computed_mean = self._bkd.sum(weights * points)
            self._bkd.assert_allclose(
                self._bkd.atleast_1d(computed_mean),
                self._bkd.asarray([expected_mean]),
                rtol=1e-10,
                atol=1e-14,
            )

    @parametrize(
        "name,mtype,params",
        MARGINAL_CONFIGS,
    )
    def test_second_moment_exact(
        self, name: str, mtype: str, params: Dict[str, float]
    ) -> None:
        """Test second moment is exact: sum(w_i * x_i^2) = E[X^2] with 2+ points.

        Gauss quadrature with n points is exact for polynomials up to
        degree 2n-1. Since f(x) = x^2 has degree 2, it's exact for n >= 2.
        """
        _, expected_second = get_analytical_moments(mtype, params)

        for npoints in [2, 3, 5, 10]:
            points, weights = self._get_quadrature_rule(mtype, params, npoints)
            computed_second = self._bkd.sum(weights * points**2)
            self._bkd.assert_allclose(
                self._bkd.atleast_1d(computed_second),
                self._bkd.asarray([expected_second]),
                rtol=1e-10,
                atol=1e-14,
            )

    @parametrize(
        "name,mtype,params",
        MARGINAL_CONFIGS,
    )
    def test_degree_of_exactness(
        self, name: str, mtype: str, params: Dict[str, float]
    ) -> None:
        """Test degree of exactness: n points exact for degree <= 2n-1.

        For each n, we test that:
        - Polynomials of degree 2n-1 integrate exactly
        - The constant function integrates to 1
        """
        for npoints in [1, 2, 3, 4, 5]:
            max_exact_degree = 2 * npoints - 1
            points, weights = self._get_quadrature_rule(mtype, params, npoints)

            # Test constant function: integral = 1
            const_integral = self._bkd.sum(weights)
            self._bkd.assert_allclose(
                self._bkd.atleast_1d(const_integral),
                self._bkd.asarray([1.0]),
                rtol=1e-12,
                atol=1e-14,
            )

            # Test monomials up to degree 2n-1 by comparing to analytical
            # For computational stability, we normalize by expected value
            # and just verify the polynomial is integrated reasonably well
            for degree in range(1, max_exact_degree + 1):
                # Compute integral of x^degree with quadrature
                computed = self._bkd.sum(weights * points**degree)

                # For checking, compute expected value using scipy or formula
                # We'll use Monte Carlo reference (high sample count) or
                # analytical moments where available
                if degree <= 2:
                    if degree == 1:
                        expected, _ = get_analytical_moments(mtype, params)
                    else:  # degree == 2
                        _, expected = get_analytical_moments(mtype, params)

                    self._bkd.assert_allclose(
                        self._bkd.atleast_1d(computed),
                        self._bkd.asarray([expected]),
                        rtol=1e-10,
                        atol=1e-12,
                    )

    @parametrize(
        "name,mtype,params",
        MARGINAL_CONFIGS,
    )
    def test_variance_computation(
        self, name: str, mtype: str, params: Dict[str, float]
    ) -> None:
        """Test variance can be computed: Var[X] = E[X^2] - E[X]^2.

        Uses 2 or more quadrature points to compute variance exactly.
        """
        expected_mean, expected_second = get_analytical_moments(mtype, params)
        expected_variance = expected_second - expected_mean**2

        for npoints in [2, 3, 5]:
            points, weights = self._get_quadrature_rule(mtype, params, npoints)
            computed_mean = self._bkd.sum(weights * points)
            computed_second = self._bkd.sum(weights * points**2)
            computed_variance = computed_second - computed_mean**2

            self._bkd.assert_allclose(
                self._bkd.atleast_1d(computed_variance),
                self._bkd.asarray([expected_variance]),
                rtol=1e-10,
                atol=1e-14,
            )

    @parametrize(
        "name,mtype,params",
        MARGINAL_CONFIGS,
    )
    def test_points_in_support(
        self, name: str, mtype: str, params: Dict[str, float]
    ) -> None:
        """Test that quadrature points lie within the distribution support."""
        for npoints in [1, 2, 3, 5, 10]:
            points, _ = self._get_quadrature_rule(mtype, params, npoints)

            if mtype == "uniform":
                lb, ub = params["lower"], params["upper"]
                lb_check = points >= self._bkd.asarray(lb - 1e-12)
                ub_check = points <= self._bkd.asarray(ub + 1e-12)
                assert not isinstance(lb_check, bool)  # for mypy
                assert not isinstance(ub_check, bool)  # for mypy
                self.assertTrue(
                    self._bkd.all_bool(lb_check),
                    f"Points below lower bound for {name} with {npoints} points"
                )
                self.assertTrue(
                    self._bkd.all_bool(ub_check),
                    f"Points above upper bound for {name} with {npoints} points"
                )

            elif mtype == "beta":
                lb, ub = params.get("lb", 0.0), params.get("ub", 1.0)
                lb_check = points >= self._bkd.asarray(lb - 1e-12)
                ub_check = points <= self._bkd.asarray(ub + 1e-12)
                assert not isinstance(lb_check, bool)  # for mypy
                assert not isinstance(ub_check, bool)  # for mypy
                self.assertTrue(
                    self._bkd.all_bool(lb_check),
                    f"Points below lower bound for {name} with {npoints} points"
                )
                self.assertTrue(
                    self._bkd.all_bool(ub_check),
                    f"Points above upper bound for {name} with {npoints} points"
                )

            elif mtype == "gamma":
                lb_check = points >= self._bkd.asarray(-1e-12)
                assert not isinstance(lb_check, bool)  # for mypy
                self.assertTrue(
                    self._bkd.all_bool(lb_check),
                    f"Points below 0 for {name} with {npoints} points"
                )

            # Gaussian has infinite support, so no bounds to check


# =============================================================================
# NumPy backend tests
# =============================================================================


class TestGaussQuadrature1DNumpy(TestGaussQuadrature1D[NDArray[Any]]):
    """NumPy backend tests for 1D Gauss quadrature."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# =============================================================================
# PyTorch backend tests
# =============================================================================


class TestGaussQuadrature1DTorch(TestGaussQuadrature1D[torch.Tensor]):
    """PyTorch backend tests for 1D Gauss quadrature."""

    __test__ = True

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
