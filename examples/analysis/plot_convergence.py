r"""
Convergence Studies
-------------------
This tutorial demonstrates how to investigate the convergence of parameterized numerical approximations, for example tensor product quadrature.

"""

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.benchmarks import GenzBenchmark
from pyapprox.analysis.convergence_studies import (
    ConvergenceStudy,
    ConvergenceErrorEstimator,
)
from pyapprox.interface.model import (
    SingleSampleModel,
    MultiIndexModelEnsemble,
)
from pyapprox.surrogates.bases.univariate.local import (
    setup_univariate_piecewise_polynomial_basis,
)
from pyapprox.surrogates.bases.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.bases.basisexp import TensorProductInterpolant


# %%
# First, define the model to approximate with tensor product quadrature
nvars = 2
benchmark = GenzBenchmark(name="oscillatory", nvars=nvars, cfactor=5)
train_model = benchmark.model()


# %%
# Now define a model which can be used to integrate multivariate functions with tensor product quadrature, that is compute


class TensorProductIntegrationModel(SingleSampleModel):
    def __init__(self, train_model, basis_type, nnodes_1d):
        self._train_model = train_model
        self._basis_type = basis_type
        self._nnodes_1d = nnodes_1d
        super().__init__()
        self._setup()

    def nqoi(self):
        return 1

    def nvars(self):
        return self._train_model.nvars()

    def _setup(self):
        bounds = [0, 1]
        bases_1d = [
            setup_univariate_piecewise_polynomial_basis(
                self._basis_type, bounds
            )
            for ii in range(self.nvars())
        ]
        basis = TensorProductInterpolatingBasis(bases_1d)
        self._interp = TensorProductInterpolant(basis)
        basis.set_tensor_product_indices(self._nnodes_1d)
        train_samples = basis.tensor_product_grid()
        train_values = self._train_model(train_samples)
        self._interp.fit(train_values)


# %%
# Now create a MutiIndexModelEnsemble which is required to run a
# ConvergenceStudy
class TensorProductIntegrationModelEnsemble(MultiIndexModelEnsemble):
    def __init__(self, train_model, basis_type, nrefinement_vars):
        self._basis_type = basis_type
        self._train_model = train_model
        super().__init__([5] * nvars)

    def model_id_to_grid_resolutions(self, model_id):
        resolutions = 2**model_id + 1
        resolutions[model_id == 0] = 1
        return resolutions

    def setup_model(self, model_id):
        nnodes_1d = self.model_id_to_grid_resolutions(model_id)
        return TensorProductIntegrationModel(
            self._train_model, self._basis_type, nnodes_1d
        )


model_ensemble = TensorProductIntegrationModelEnsemble(
    train_model, "quadratic", nvars
)


# %%
# Now define how the model and costs are computed
class MeanErrorEstimator(ConvergenceErrorEstimator):
    def _estimate(self, model):
        return (
            model._bkd.abs(model._interp.mean() - benchmark.integral()),
            model._interp.basis().nterms(),
        )


error_est = MeanErrorEstimator()


# %%
# Now run the convergence study
study = ConvergenceStudy(model_ensemble, error_est, np.array([[0, 5]] * nvars))
study.run()

# %%
# Plot the convergence
axs = plt.subplots(1, 2, figsize=(2 * 8, 6))[1]
_ = study.plot(axs)

# %%
# The left plots depicts the convergence of the estimated integral as the number of quadrature points in the first dimension :math:`n_1` are increased for varying values of the number of quadrature points in the second dimension :math:`n_2`. The roles of the discretizations are reveresed in the right plot. These plots confirm that the estimated intergral converges as the expected quadratic rate,  until the error introduced by fixing the discretization in one dimension dominates the discretization error introduced by the other dimension.
#
