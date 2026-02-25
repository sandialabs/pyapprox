"""Optimizer-based fitter for flow matching.

Iteratively optimizes the VF coefficients by minimizing the CFM loss
via a general-purpose optimizer.
"""

import copy
from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)

from pyapprox.surrogates.flowmatching.protocols import (
    ProbabilityPathProtocol,
)
from pyapprox.surrogates.flowmatching.cfm_loss import (
    CFMLoss,
    FlowMatchingObjective,
)
from pyapprox.surrogates.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.surrogates.flowmatching.fitters.results import (
    FlowMatchingFitResult,
)


class OptimizerFitter(Generic[Array]):
    """Iterative optimization fitter for flow matching.

    Deep-clones the VF, creates a FlowMatchingObjective, binds an
    optimizer, and minimizes.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : BindableOptimizerProtocol[Array], optional
        Optimizer instance. If None, uses ScipyTrustConstrOptimizer
        with maxiter=1000.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> None:
        self._bkd = bkd
        self._optimizer = optimizer

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
        """Fit a vector field via iterative optimization.

        Parameters
        ----------
        vf : BasisExpansion
            Vector field to fit. Deep-cloned internally.
        path : ProbabilityPathProtocol[Array]
            Probability path defining interpolation.
        loss : CFMLoss[Array]
            CFM loss function.
        quad_data : FlowMatchingQuadData[Array]
            Pre-assembled quadrature data.

        Returns
        -------
        FlowMatchingFitResult[Array]
            Result containing the fitted VF and training loss.
        """
        bkd = self._bkd
        fitted_vf = copy.deepcopy(vf)

        # Create objective wrapping loss + quad data
        objective = FlowMatchingObjective(
            fitted_vf, path, loss, quad_data, bkd
        )

        # Get bounds from VF hyperparameter list
        bounds = fitted_vf.hyp_list().get_active_bounds()  # type: ignore[union-attr]

        # Get optimizer
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer,
            )
            optimizer = ScipyTrustConstrOptimizer(
                verbosity=0, maxiter=1000
            )

        # Bind and minimize
        optimizer.bind(objective, bounds)

        init_guess = fitted_vf.hyp_list().get_active_values()  # type: ignore[union-attr]
        if len(init_guess.shape) == 1:
            init_guess = bkd.reshape(init_guess, (-1, 1))

        opt_result = optimizer.minimize(init_guess)

        # Update VF with optimal parameters
        optimal_params = opt_result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        fitted_vf.hyp_list().set_active_values(optimal_params)  # type: ignore[union-attr]
        fitted_vf._sync_from_hyp_list()  # type: ignore[union-attr]

        # Compute final loss
        qd = quad_data
        final_loss_val = loss(
            fitted_vf, path, qd.t(), qd.x0(), qd.x1(),
            qd.weights(), qd.c(),
        )
        final_loss = float(
            bkd.to_numpy(bkd.reshape(final_loss_val, (1,)))[0]
        )

        return FlowMatchingFitResult(
            surrogate=fitted_vf,
            training_loss=final_loss,
        )
