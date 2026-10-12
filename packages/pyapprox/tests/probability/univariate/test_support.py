"""The bounded marginals outside their support.

Below the support the CDF is 0 and above it 1; the PDF is 0 and the log
PDF -inf outside it; every derivative of those constants is zero. Each
case is checked against scipy at points below, at, inside and above the
support, and with torch, points outside must leave the gradients of the
points inside exactly as they would be without them.
"""

from typing import Any, Callable, List, Tuple

import numpy as np
import pytest
from scipy import stats

from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.probability.univariate.gamma import GammaMarginal
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

_Factory = Callable[[Backend[Any]], Any]

# name, marginal, scipy reference, points (below, at, inside, at, above),
# indices of the points strictly outside, and of those strictly inside.
CASES: List[Tuple[str, _Factory, Any, List[float], List[int], List[int]]] = [
    (
        "gamma",
        lambda bkd: GammaMarginal(2.0, 1.5, bkd),
        stats.gamma(2.0, scale=1.5),
        [-2.0, -1e-12, 0.0, 0.5, 2.0],
        [0, 1],
        [3, 4],
    ),
    (
        "beta",
        lambda bkd: BetaMarginal(2.0, 3.0, bkd, lb=1.0, ub=4.0),
        stats.beta(2.0, 3.0, loc=1.0, scale=3.0),
        [0.0, 1.0, 2.5, 4.0, 5.0],
        [0, 4],
        [2],
    ),
    (
        "uniform",
        lambda bkd: UniformMarginal(1.0, 4.0, bkd),
        stats.uniform(1.0, 3.0),
        [0.0, 1.0, 2.5, 4.0, 5.0],
        [0, 4],
        [2],
    ),
]
IDS = [case[0] for case in CASES]


@pytest.mark.parametrize("name,make,reference,points,outside,inside", CASES, ids=IDS)
class TestOutsideTheSupport:
    def test_values_match_scipy(
        self,
        bkd: Backend[Array],
        name: str,
        make: _Factory,
        reference: Any,
        points: List[float],
        outside: List[int],
        inside: List[int],
    ) -> None:
        marginal = make(bkd)
        x = bkd.asarray([points])
        bkd.assert_allclose(
            marginal.cdf(x), bkd.asarray([reference.cdf(points)]), atol=1e-12
        )
        bkd.assert_allclose(
            marginal.pdf(x), bkd.asarray([reference.pdf(points)]), atol=1e-12
        )
        expected_log = bkd.asarray([reference.logpdf(np.asarray(points)[outside])])
        bkd.assert_allclose(marginal.logpdf(x)[:, outside], expected_log, rtol=0.0)

    def test_derivatives_vanish_outside(
        self,
        bkd: Backend[Array],
        name: str,
        make: _Factory,
        reference: Any,
        points: List[float],
        outside: List[int],
        inside: List[int],
    ) -> None:
        marginal = make(bkd)
        x = bkd.asarray([[points[ii] for ii in outside + inside]])
        nout = len(outside)
        wrt_x = marginal.logpdf_jacobian(x)
        wrt_params = marginal.logpdf_jacobian_wrt_params(x)
        bkd.assert_allclose(wrt_x[:, :nout], bkd.zeros((1, nout)), atol=0.0)
        bkd.assert_allclose(
            wrt_params[:nout], bkd.zeros((nout, wrt_params.shape[1])), atol=0.0
        )
        assert bkd.all_bool(bkd.isfinite(wrt_x))
        assert bkd.all_bool(bkd.isfinite(wrt_params))

    def test_points_outside_leave_the_gradients_inside_unchanged(
        self,
        torch_bkd: Backend[Array],
        name: str,
        make: _Factory,
        reference: Any,
        points: List[float],
        outside: List[int],
        inside: List[int],
    ) -> None:
        """No NaN from the excluded branch reaches the autograd graph."""
        import torch

        def gradient(indices: List[int]) -> Any:
            marginal = make(torch_bkd)
            hyp = marginal.hyp_list()
            values = hyp.get_values().clone().requires_grad_(True)
            hyp.set_values(values)
            x = torch.tensor([[points[ii] for ii in indices]])
            total = marginal.cdf(x).sum() + marginal.pdf(x).sum()
            (grad,) = torch.autograd.grad(total, values)
            return grad

        with_outside = gradient(outside + inside)
        assert bool(torch.isfinite(with_outside).all())
        torch_bkd.assert_allclose(with_outside, gradient(inside), rtol=1e-12)
