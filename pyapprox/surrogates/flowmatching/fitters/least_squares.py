"""Least-squares fitter for flow matching.

Solves the CFM loss in one shot via weighted least squares, exploiting
the linearity of the BasisExpansion vector field.
"""

import copy
from typing import Generic

from pyapprox.optimization.linear import LeastSquaresSolver
from pyapprox.surrogates.flowmatching.cfm_loss import CFMLoss
from pyapprox.surrogates.flowmatching.fitters.results import (
    FlowMatchingFitResult,
)
from pyapprox.surrogates.flowmatching.protocols import (
    ProbabilityPathProtocol,
)
from pyapprox.surrogates.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
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
        loss: CFMLoss[Array],
        quad_data: FlowMatchingQuadData[Array],
    ) -> FlowMatchingFitResult[Array]:
        """Fit a vector field via weighted least squares.

        Parameters
        ----------
        vf : BasisExpansion
            Vector field to fit. Deep-cloned internally.
        path : ProbabilityPathProtocol[Array]
            Probability path defining interpolation.
        loss : CFMLoss[Array]
            CFM loss function (used for its time weight).
        quad_data : FlowMatchingQuadData[Array]
            Pre-assembled quadrature data.

        Returns
        -------
        FlowMatchingFitResult[Array]
            Result containing the fitted VF and training loss.
        """
        bkd = self._bkd
        fitted_vf = copy.deepcopy(vf)

        qd = quad_data
        t = qd.t()
        x0 = qd.x0()
        x1 = qd.x1()

        # Compute interpolated state and target field
        x_t = path.interpolate(t, x0, x1)
        u_t = path.target_field(t, x0, x1)

        # Build VF input
        if qd.c() is not None:
            vf_input = bkd.vstack([t, x_t, qd.c()])
        else:
            vf_input = bkd.vstack([t, x_t])

        # Combine weights: quadrature weights * time weight
        w_t = loss.weight()(t)  # (1, n_quad)
        combined_w = qd.weights() * w_t[0, :]
        assert combined_w.shape == (qd.n_quad(),), (
            f"Expected combined weights shape ({qd.n_quad()},), got {combined_w.shape}"
        )

        # Solve via BasisExpansion.fit() with weighted solver
        solver = LeastSquaresSolver(bkd)
        solver.set_weights(combined_w)
        fitted_vf.fit(vf_input, u_t, solver=solver)  # type: ignore[union-attr]

        # Compute final loss
        final_loss_val = loss(fitted_vf, path, t, x0, x1, qd.weights(), qd.c())
        final_loss = float(bkd.to_numpy(bkd.reshape(final_loss_val, (1,)))[0])

        return FlowMatchingFitResult(
            surrogate=fitted_vf,
            training_loss=final_loss,
        )
