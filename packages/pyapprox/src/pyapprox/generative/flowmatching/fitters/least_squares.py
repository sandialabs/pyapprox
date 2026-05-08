"""Least-squares fitter for flow matching.

Solves the CFM loss in one shot via weighted least squares, exploiting
the linearity of the BasisExpansion vector field.
"""

import copy
from typing import Generic, Optional

from pyapprox.generative.flowmatching.fitters.results import (
    FlowMatchingFitResult,
)
from pyapprox.generative.flowmatching.protocols import (
    ProbabilityPathProtocol,
    TimeWeightProtocol,
)
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.generative.flowmatching.time_weight import UniformWeight
from pyapprox.optimization.linear import LeastSquaresSolver
from pyapprox.util.backends.protocols import Array, Backend


class LeastSquaresFitter(Generic[Array]):
    """Direct weighted least-squares fitter for polynomial VFs.

    Exploits that the CFM loss with a BasisExpansion VF is quadratic in
    the coefficients: ``L(theta) = ||W^{1/2}(Phi @ theta - U)||^2_F``.

    This is solved in one shot via ``BasisExpansion.fit()`` with a
    ``LeastSquaresSolver`` that has the combined weights set.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def fit(
        self,
        vf: object,
        path: ProbabilityPathProtocol[Array],
        quad_data: FlowMatchingQuadData[Array],
        time_weight: Optional[TimeWeightProtocol[Array]] = None,
    ) -> FlowMatchingFitResult[Array]:
        """Fit a vector field via weighted least squares.

        Parameters
        ----------
        vf : BasisExpansion
            Vector field to fit. Deep-cloned internally.
        path : ProbabilityPathProtocol[Array]
            Probability path defining interpolation.
        quad_data : FlowMatchingQuadData[Array]
            Pre-assembled quadrature data.
        time_weight : TimeWeightProtocol[Array], optional
            Time-dependent weight. Defaults to uniform.

        Returns
        -------
        FlowMatchingFitResult[Array]
            Result containing the fitted VF and training loss.
        """
        bkd = self._bkd
        fitted_vf = copy.deepcopy(vf)

        tw = time_weight if time_weight is not None else UniformWeight(bkd)

        qd = quad_data
        t = qd.t()
        x0 = qd.x0()
        x1 = qd.x1()

        x_t = path.interpolate(t, x0, x1)
        u_t = path.target_field(t, x0, x1)

        if qd.c() is not None:
            vf_input = bkd.vstack([t, x_t, qd.c()])
        else:
            vf_input = bkd.vstack([t, x_t])

        w_t = tw(t)  # (1, n_quad)
        combined_w = qd.weights() * w_t[0, :]
        assert combined_w.shape == (qd.n_quad(),), (
            f"Expected combined weights shape ({qd.n_quad()},), "
            f"got {combined_w.shape}"
        )

        solver = LeastSquaresSolver(bkd)
        solver.set_weights(combined_w)
        fitted_vf.fit(vf_input, u_t, solver=solver)

        v_t = fitted_vf(vf_input)
        diff = v_t - u_t
        pointwise = bkd.sum(diff * diff, axis=0)
        final_loss = bkd.to_float(
            bkd.sum(qd.weights() * w_t[0, :] * pointwise)
        )

        return FlowMatchingFitResult(
            surrogate=fitted_vf,
            training_loss=final_loss,
        )
