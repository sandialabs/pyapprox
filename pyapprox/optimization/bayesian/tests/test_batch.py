"""Tests for KrigingBeliever batch selection."""

import numpy as np

from pyapprox.optimization.bayesian.acquisition.analytic import (
    ExpectedImprovement,
)
from pyapprox.optimization.bayesian.acquisition_optimizer import (
    AcquisitionOptimizer,
)
from pyapprox.optimization.bayesian.batch.greedy import KrigingBeliever
from pyapprox.optimization.bayesian.domain.box import BoxDomain
from pyapprox.optimization.bayesian.fitter_adapter import GPFitterAdapter
from pyapprox.optimization.bayesian.protocols import AcquisitionContext
from pyapprox.optimization.minimize.scipy.slsqp import (
    ScipySLSQPOptimizer,
)
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.fitters.maximum_likelihood_fitter import (
    GPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel
class TestKrigingBeliever:
    def test_batch_well_separated(self, bkd) -> None:
        """Batch points from KB are well-separated in [0,1]."""
        np.random.seed(42)
        nvars = 1
        kernel = Matern52Kernel([1.0], (0.1, 10.0), nvars, bkd)
        gp_template = ExactGaussianProcess(kernel, nvars, bkd, nugget=1e-6)

        # Train on endpoints
        X_train = bkd.array([[0.0, 1.0]])
        y_train = bkd.array([[0.0, 0.0]])

        fitter = GPMaximumLikelihoodFitter(bkd)
        adapter = GPFitterAdapter(fitter)
        fitted_gp = adapter.fit(gp_template, X_train, y_train)

        domain = BoxDomain(bkd.array([[0.0, 1.0]]), bkd)
        ei = ExpectedImprovement()
        scipy_opt = ScipySLSQPOptimizer(maxiter=100)
        acq_opt = AcquisitionOptimizer(
            scipy_opt, bkd, n_restarts=5, n_raw_candidates=100
        )

        def context_factory(pending_X):
            if pending_X is not None:
                fantasy_y = fitted_gp.predict(pending_X)
                X_aug = bkd.hstack([X_train, pending_X])
                y_aug = bkd.hstack([y_train, fantasy_y])
                surr = adapter.fit(gp_template, X_aug, y_aug)
            else:
                surr = fitted_gp
            best_value = bkd.array([0.0])  # best_value = min(y) = 0
            return AcquisitionContext(
                surrogate=surr,
                best_value=best_value,
                bkd=bkd,
                pending_X=pending_X,
                minimize=True,
            )

        kb = KrigingBeliever()
        batch = kb.select_batch(
            batch_size=3,
            acquisition=ei,
            context_factory=context_factory,
            acquisition_optimizer=acq_opt,
            domain=domain,
        )

        assert batch.shape == (1, 3)
        batch_np = bkd.to_numpy(batch)[0]

        # Check well-separated: min pairwise distance > 0.05
        for i in range(3):
            for j in range(i + 1, 3):
                dist = abs(batch_np[i] - batch_np[j])
                assert dist > 0.05, (
                    f"Points {i} and {j} too close: {dist:.4f}"
                )
