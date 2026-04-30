"""Optimizer-based fitter for flow matching.

Iteratively optimizes the VF coefficients by minimizing the CFM loss
via a general-purpose optimizer.
"""

import copy
from typing import Generic, Optional

from pyapprox.generative.flowmatching.fitters.results import (
    FlowMatchingFitResult,
)
from pyapprox.generative.flowmatching.objective import (
    FlowMatchingObjective,
)
from pyapprox.generative.flowmatching.protocols import (
    ProbabilityPathProtocol,
    TimeWeightProtocol,
)
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.generative.flowmatching.time_weight import UniformWeight
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


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
        quad_data: FlowMatchingQuadData[Array],
        time_weight: Optional[TimeWeightProtocol[Array]] = None,
    ) -> FlowMatchingFitResult[Array]:
        """Fit a vector field via iterative optimization.

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

        objective = FlowMatchingObjective(
            fitted_vf, path, quad_data, bkd, time_weight
        )

        bounds = fitted_vf.hyp_list().get_active_bounds()

        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer,
            )

            optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=1000)

        optimizer.bind(objective, bounds)

        init_guess = fitted_vf.hyp_list().get_active_values()
        if len(init_guess.shape) == 1:
            init_guess = bkd.reshape(init_guess, (-1, 1))

        opt_result = optimizer.minimize(init_guess)

        optimal_params = opt_result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        fitted_vf.hyp_list().set_active_values(optimal_params)
        fitted_vf.sync_params()

        tw = time_weight if time_weight is not None else UniformWeight(bkd)
        qd = quad_data
        t, x0, x1 = qd.t(), qd.x0(), qd.x1()
        x_t = path.interpolate(t, x0, x1)
        u_t = path.target_field(t, x0, x1)
        if qd.c() is not None:
            vf_input = bkd.vstack([t, x_t, qd.c()])
        else:
            vf_input = bkd.vstack([t, x_t])
        v_t = fitted_vf(vf_input)
        diff = v_t - u_t
        w_t = tw(t)
        pointwise = bkd.sum(diff * diff, axis=0)
        final_loss = bkd.to_float(
            bkd.sum(qd.weights() * w_t[0, :] * pointwise)
        )

        return FlowMatchingFitResult(
            surrogate=fitted_vf,
            training_loss=final_loss,
        )
