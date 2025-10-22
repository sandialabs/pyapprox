import unittest
from functools import partial

import numpy as np
import sympy as sp

from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.interface.wrappers import (
    create_active_set_variable_model,
    PoolModelWrapper,
    ScipyModelWrapper,
    ChangeModelSignWrapper,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


def _pickable_function(bkd, sample):
    return bkd.stack(
        (bkd.sum(sample**2, axis=0), bkd.sum(sample**3, axis=0)), axis=1
    )


class TestWrappers:
    def setUp(self):
        np.random.seed(1)

    def _evaluate_sp_lambda(self, sp_lambda, sample):
        # sp_lambda returns a single function output
        bkd = self.get_backend()
        assert sample.ndim == 2 and sample.shape[1] == 1
        vals = bkd.asarray(sp_lambda(*bkd.to_numpy(sample[:, 0])))
        return bkd.atleast2d(vals)

    def test_scipy_wrapper(self):
        bkd = self.get_backend()
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = sum([s * (ii + 1) for ii, s in enumerate(symbs)]) ** 4
        sp_grad = [sp_fun.diff(x) for x in symbs]
        sp_hessian = [[sp_fun.diff(x).diff(y) for x in symbs] for y in symbs]
        model = ModelFromSingleSampleCallable(
            1,
            3,
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample
            ),
            jacobian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample
            ),
            hessian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample
            )[None, ...],
            backend=bkd,
        )
        scipy_model = ScipyModelWrapper(model)
        # check scipy model works with 1D sample array
        sample = bkd.asarray(np.random.uniform(0, 1, (nvars)))
        assert np.allclose(
            scipy_model.jac(sample),
            self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample[:, None]
            ),
        )
        assert np.allclose(
            scipy_model.hess(sample),
            self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian, "numpy"), sample[:, None]
            ),
        )

    def test_active_set_variable_model(self):
        bkd = self.get_backend()
        nvars = 3
        model = ModelFromSingleSampleCallable(
            1,
            nvars,
            lambda x: ((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2) + x[2],
            jacobian=lambda x: bkd.array(
                [[2 * (x[0] - 1), 2 * (x[1] - 2.5), x[2] * 0 + 1]]
            ),
            hessian=lambda x: bkd.array([[2.0, 0, 0], [0, 2, 0], [0, 0, 0]])[
                None, ...
            ],
            sample_ndim=1,
            values_ndim=0,
            backend=bkd,
        )

        active_var_indices = bkd.array([0, 2])
        nominal_sample = bkd.arange(1.0, nvars + 1)[:, None]
        inactive_var_values = nominal_sample[
            np.delete(np.arange(nvars), active_var_indices)
        ]
        active_set_model = create_active_set_variable_model(
            model, nvars, inactive_var_values, active_var_indices
        )

        active_sample = nominal_sample[active_var_indices]
        assert isinstance(active_set_model, ModelFromSingleSampleCallable)
        assert active_set_model.nvars() == active_var_indices.shape[0]
        assert active_set_model.nqoi() == model.nqoi()
        assert active_set_model.noriginal_vars() == model.nvars()
        assert bkd.allclose(
            active_set_model(active_sample), model(nominal_sample)
        )
        assert bkd.allclose(
            active_set_model.jacobian(active_sample),
            model.jacobian(nominal_sample)[:, active_var_indices],
        )
        vec = bkd.array(np.random.normal(0, 1, (nvars, 1)))
        assert bkd.allclose(
            active_set_model.apply_jacobian(
                active_sample, vec[active_var_indices]
            ),
            model.jacobian(nominal_sample)[:, active_var_indices]
            @ vec[active_var_indices],
        )
        assert bkd.allclose(
            active_set_model.apply_hessian(
                active_sample, vec[active_var_indices]
            ),
            model.hessian(nominal_sample)[0][
                np.ix_(active_var_indices, active_var_indices)
            ]
            @ vec[active_var_indices],
        )

    def test_change_sign_model(self):
        bkd = self.get_backend()
        nvars = 3
        model = ModelFromSingleSampleCallable(
            1,
            nvars,
            lambda x: ((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2) + x[2],
            jacobian=lambda x: bkd.array(
                [[2 * (x[0] - 1), 2 * (x[1] - 2.5), x[2] * 0 + 1]]
            ),
            hessian=lambda x: bkd.array([[2.0, 0, 0], [0, 2, 0], [0, 0, 0]])[
                None, ...
            ],
            sample_ndim=1,
            values_ndim=0,
            backend=bkd,
        )

        sample = bkd.arange(1.0, nvars + 1)[:, None]
        signed_model = ChangeModelSignWrapper(model)
        assert bkd.allclose(signed_model(sample), -model(sample))
        assert bkd.allclose(
            signed_model.jacobian(sample),
            -model.jacobian(sample),
        )
        vec = bkd.array(np.random.normal(0, 1, (nvars, 1)))
        assert bkd.allclose(
            signed_model.apply_jacobian(sample, vec),
            -model.apply_jacobian(sample, vec),
        )
        assert bkd.allclose(
            signed_model.hessian(sample),
            -model.hessian(sample),
        )
        assert bkd.allclose(
            signed_model.apply_hessian(sample, vec),
            -model.apply_hessian(sample, vec),
        )

    def test_pool_model_wrapper(self):
        bkd = self.get_backend()
        nvars, nsamples = 3, 4
        model = ModelFromSingleSampleCallable(
            2,
            nvars,
            partial(_pickable_function, bkd),
            backend=bkd,
        )
        pool_model = PoolModelWrapper(model, nprocs=2, assert_omp=False)
        pool_model.model().work_tracker().set_active(True)
        pool_model.work_tracker().set_active(True)
        samples = bkd.asarray(np.random.uniform(0, 1, (nvars, nsamples)))
        values = pool_model(samples)
        assert bkd.allclose(values, _pickable_function(bkd, samples))
        assert pool_model.work_tracker().nevaluations("val") == 4
        assert pool_model.model().work_tracker().nevaluations("val") == 4


class TestNumpyWrappers(TestWrappers, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchWrappers(TestWrappers, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
