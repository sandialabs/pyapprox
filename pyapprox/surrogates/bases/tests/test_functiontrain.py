import unittest
import copy

import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.basis import MonomialBasis
from pyapprox.surrogates.bases.basisexp import MonomialExpansion
from pyapprox.surrogates.loss import RMSELoss
from pyapprox.surrogates.bases.functiontrain import (
    AdditiveFunctionTrain,
    AlternatingLstSqOptimizer,
    FunctionTrainAlternatingLstSqLoss,
)
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer,
    MultiStartOptimizer,
    RandomUniformOptimzerIterateGenerator,
)


class TestFunctionTrain:
    def setUp(self):
        np.random.seed(1)

    def test_additive_function_train(self):
        nvars = 3
        ntrain_samples = 50
        bkd = self.get_backend()
        train_samples = bkd._la_asarray(
            np.random.uniform(-1, 1, (nvars, ntrain_samples))
        )

        nterms = 3
        nqoi = 2
        basis = MonomialBasis(backend=bkd)
        basis.set_tensor_product_indices([nterms])
        basisexp = MonomialExpansion(basis, solver=None, nqoi=nqoi)
        univariate_funs = [copy.deepcopy(basisexp) for ii in range(nvars)]
        ft = AdditiveFunctionTrain(univariate_funs, nqoi)
        true_active_opt_params = bkd._la_full(
            (ft.hyp_list.nactive_vars(),), 1.0
        )
        ft.hyp_list.set_active_opt_params(true_active_opt_params)
        ft_vals = ft(train_samples)
        univariate_vals = [
            fun(train_samples[ii : ii + 1])
            for ii, fun in enumerate(univariate_funs)
        ]
        train_values = sum(univariate_vals)
        assert bkd._la_allclose(train_values, ft_vals)

        ft.hyp_list.set_all_active()
        jac_ans = [
            ft._core_jacobian(train_samples, ii) for ii in range(ft.nvars())
        ]
        if bkd._la_jacobian_implemented():
            for ii in range(ft.nvars()):
                for qq in range(ft.nqoi()):
                    assert bkd._la_allclose(
                        ft._core_jacobian_ad(train_samples, ii)[qq],
                        jac_ans[ii][qq],
                        atol=1e-14,
                    )
        # else: skip autograd test as backend does not support it

        # when learning an AdditiveFunctionTrain then tensor is linear so
        # one alternating least squares pass will suffice regardless
        # of the initial guess
        params = bkd._la_asarray(
            np.random.normal(0, 1, (ft.hyp_list.nactive_vars(),))
        )
        ft.hyp_list.set_values(params)

        ft.hyp_list.set_all_active()
        optimizer = AlternatingLstSqOptimizer(tol=1e-8, verbosity=0)
        ms_optimizer = MultiStartOptimizer(optimizer, ncandidates=1)
        iterate_gen = RandomUniformOptimzerIterateGenerator(
            ft.hyp_list.nactive_vars(), backend=bkd
        )
        iterate_gen.set_bounds([-2, 2])
        ms_optimizer.set_initial_iterate_generator(iterate_gen)
        ft.set_optimizer(ms_optimizer)
        ft.set_loss(FunctionTrainAlternatingLstSqLoss())
        ft.fit(train_samples, train_values)
        ft_vals = ft(train_samples)
        assert bkd._la_allclose(train_values, ft_vals)

        if not bkd._la_jacobian_implemented():
            return
        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(
            gtol=1e-8, ftol=1e-12, maxiter=1000, method="L-BFGS-B"
        )
        optimizer.set_verbosity(0)
        ms_optimizer = MultiStartOptimizer(optimizer, ncandidates=1)
        iterate_gen = RandomUniformOptimzerIterateGenerator(
            ft.hyp_list.nactive_vars(), backend=bkd
        )
        # need to set bounds to be small because initial guess effects
        # optimization
        iterate_gen.set_bounds([-2, 2])
        ms_optimizer.set_initial_iterate_generator(iterate_gen)
        ms_optimizer.set_verbosity(2)
        ft.set_optimizer(ms_optimizer)
        ft.set_loss(RMSELoss())
        ft.fit(train_samples, train_values)
        ft_vals = ft(train_samples)
        assert bkd._la_allclose(train_values, ft_vals)


class TestNumpyFunctionTrain(TestFunctionTrain, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchFunctionTrain(TestFunctionTrain, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)
