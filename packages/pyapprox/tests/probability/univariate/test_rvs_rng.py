"""Every univariate marginal draws from a generator when given one.

Given ``rng``, a draw depends on the generator alone, so the same seed
repeats it whatever else has drawn from numpy's global state. Without
one, the global state is used as before.
"""

from typing import Any, Callable, List, Tuple

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.probability.univariate.discrete import (
    CustomDiscreteMarginal,
    DiscreteChebyshevMarginal,
)
from pyapprox.probability.univariate.gamma import GammaMarginal
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.sas_normal import SASNormalMarginal
from pyapprox.probability.univariate.scipy_continuous import (
    ScipyContinuousMarginal,
)
from pyapprox.probability.univariate.scipy_discrete import ScipyDiscreteMarginal
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

NSAMPLES = 7

_Factory = Callable[[Backend[Any]], Any]

MARGINALS: List[Tuple[str, _Factory]] = [
    ("uniform", lambda bkd: UniformMarginal(1.0, 4.0, bkd)),
    ("gaussian", lambda bkd: GaussianMarginal(0.5, 2.0, bkd)),
    ("beta", lambda bkd: BetaMarginal(2.0, 3.0, bkd, lb=1.0, ub=4.0)),
    ("gamma", lambda bkd: GammaMarginal(2.0, 1.5, bkd)),
    ("sas_normal", lambda bkd: SASNormalMarginal(0.0, 1.0, 0.5, 1.2, bkd)),
    (
        "custom_discrete",
        lambda bkd: CustomDiscreteMarginal(
            bkd.asarray([0.0, 1.0, 3.0]), bkd.asarray([0.2, 0.5, 0.3]), bkd
        ),
    ),
    ("chebyshev_discrete", lambda bkd: DiscreteChebyshevMarginal(4, bkd)),
    (
        "scipy_continuous",
        lambda bkd: ScipyContinuousMarginal(stats.norm(1.0, 2.0), bkd),
    ),
    (
        "scipy_discrete",
        lambda bkd: ScipyDiscreteMarginal(stats.poisson(3.0), bkd),
    ),
]


@pytest.mark.parametrize("name,make", MARGINALS, ids=[m[0] for m in MARGINALS])
class TestRvsWithGenerator:
    def test_the_generator_alone_fixes_the_draw(
        self, bkd: Backend[Array], name: str, make: _Factory
    ) -> None:
        marginal = make(bkd)
        first = marginal.rvs(NSAMPLES, rng=np.random.default_rng(11))
        np.random.uniform(size=13)  # unrelated global draws in between
        second = marginal.rvs(NSAMPLES, rng=np.random.default_rng(11))
        assert first.shape == (1, NSAMPLES)
        bkd.assert_allclose(first, second, rtol=0.0)

    def test_without_one_the_global_state_is_used(
        self, bkd: Backend[Array], name: str, make: _Factory
    ) -> None:
        marginal = make(bkd)
        np.random.seed(5)
        first = marginal.rvs(NSAMPLES)
        np.random.seed(5)
        second = marginal.rvs(NSAMPLES)
        bkd.assert_allclose(first, second, rtol=0.0)
