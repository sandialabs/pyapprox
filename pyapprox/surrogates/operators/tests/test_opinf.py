import unittest

import numpy as np

from pyapprox.surrogates.operators.opinf import DynamicOperatorInference
from pyapprox.benchmarks import LotkaVolterraBenchmark
from pyapprox.pde.timeintegration import ForwardEulerResidual
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.affine.kle import PrincipalComponentAnalysis
from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.affine.basis import (
    MultiIndexBasis,
    OrthonormalPolynomialBasis,
)
from pyapprox.surrogates.affine.linearsystemsolvers import LstSqSolver
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


class TestOpInf:

    def setUp(self):
        np.random.seed(1)

    def test_op_inf_ode(self):
        bkd = self.get_backend()
        time_residual_cls = ForwardEulerResidual
        benchmark = LotkaVolterraBenchmark(bkd, time_residual_cls)
        model = benchmark.model()
        delattr(model, "_functional")
        prior = IndependentMarginalsVariable(
            [marginal for marginal in benchmark.prior().marginals()[:2]]
        )

        def solve(sample):
            expanded_sample = bkd.hstack(
                (sample, benchmark.prior().mean()[2:, 0])
            )[:, None]
            return model.forward_solve(expanded_sample)[0]

        ntrain_samples = 3
        samples = prior.rvs(ntrain_samples)
        raw_snapshots = [solve(sample) for sample in samples.T]
        opinf = DynamicOperatorInference(3, model.nvars(), backend=bkd)
        snapshots, snapshot_samples, ntsteps = opinf.stack_raw_snapshots(
            raw_snapshots, samples
        )
        self.assertEqual(snapshot_samples.shape[1], snapshots.shape[1])
        for snapshot_sample in snapshot_samples[
            :, : raw_snapshots[0].shape[1]
        ].T:
            bkd.assert_allclose(snapshot_sample, samples[:, 0])
        # do not compress. Use all states but invoking PCA with full rank
        # to test machinery
        pca = PrincipalComponentAnalysis(snapshots, opinf.nqoi(), bkd)
        opinf.set_state_compressor(pca)

        state_basis = MultiIndexBasis(
            [Monomial1D(backend=bkd) for ii in range(opinf.nreduced_states())]
        )
        param_bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=prior._bkd
            )
            for marginal in prior.marginals()
        ]
        param_basis = OrthonormalPolynomialBasis(param_bases_1d)
        opinf.set_time_derivative_operator_bases(state_basis, param_basis)
        opinf.time_derivative_operator_basis().set_hyperbolic_indices(3, 1.0)
        opinf.set_linear_system_solver(LstSqSolver(bkd))
        opinf.fit(snapshot_samples, ntsteps)
        print(opinf)


class TestNumpyOpInf(TestOpInf, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


# class TestTorchOpInf(TestOpInf, unittest.TestCase):
#     def get_backend(self):
#         return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
