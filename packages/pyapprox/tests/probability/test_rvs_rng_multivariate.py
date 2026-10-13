"""The joints and multivariate Gaussians draw from a generator when given one.

Given ``rng``, a draw depends on the generator alone; without one,
numpy's global state is used as before.
"""

from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.gaussian import (
    DenseCholeskyMultivariateGaussian,
    DiagonalMultivariateGaussian,
    GaussianCanonicalForm,
    OperatorBasedMultivariateGaussian,
)
from pyapprox.probability.joint.group_independent import GroupIndependentJoint
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

NSAMPLES = 6

_Factory = Callable[[Backend[Any]], Any]


def _covariance(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[2.0, 0.5], [0.5, 1.0]])


def _mean(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[1.0], [-1.0]])


DISTRIBUTIONS: List[Tuple[str, _Factory]] = [
    (
        "independent",
        lambda bkd: IndependentJoint(
            [
                UniformMarginal(0.0, 1.0, bkd),
                GaussianMarginal(0.0, 2.0, bkd),
                BetaMarginal(2.0, 3.0, bkd),
            ],
            bkd,
        ),
    ),
    (
        "group_independent",
        lambda bkd: GroupIndependentJoint(
            [
                UniformMarginal(0.0, 1.0, bkd),
                IndependentJoint([GaussianMarginal(0.0, 1.0, bkd)] * 2, bkd),
            ],
            bkd,
        ),
    ),
    (
        "dense",
        lambda bkd: DenseCholeskyMultivariateGaussian(
            _mean(bkd), _covariance(bkd), bkd
        ),
    ),
    (
        "diagonal",
        lambda bkd: DiagonalMultivariateGaussian(
            _mean(bkd), bkd.asarray([2.0, 0.5]), bkd
        ),
    ),
    (
        "operator",
        lambda bkd: OperatorBasedMultivariateGaussian(
            _mean(bkd),
            DenseCholeskyCovarianceOperator(_covariance(bkd), bkd),
            bkd,
        ),
    ),
    (
        "canonical",
        lambda bkd: GaussianCanonicalForm(
            bkd.asarray([[2.0, 0.5], [0.5, 1.0]]),
            bkd.asarray([1.0, 0.0]),
            0.0,
            bkd,
        ),
    ),
]
IDS = [name for name, _ in DISTRIBUTIONS]


@pytest.mark.parametrize("name,make", DISTRIBUTIONS, ids=IDS)
class TestRvsWithGenerator:
    def test_the_generator_alone_fixes_the_draw(
        self, bkd: Backend[Array], name: str, make: _Factory
    ) -> None:
        distribution = make(bkd)
        first = distribution.rvs(NSAMPLES, rng=np.random.default_rng(3))
        np.random.uniform(size=11)  # unrelated global draws in between
        second = distribution.rvs(NSAMPLES, rng=np.random.default_rng(3))
        assert first.shape == (distribution.nvars(), NSAMPLES)
        bkd.assert_allclose(first, second, rtol=0.0)

    def test_without_one_the_global_state_is_used(
        self, bkd: Backend[Array], name: str, make: _Factory
    ) -> None:
        distribution = make(bkd)
        np.random.seed(9)
        first = distribution.rvs(NSAMPLES)
        np.random.seed(9)
        second = distribution.rvs(NSAMPLES)
        bkd.assert_allclose(first, second, rtol=0.0)
