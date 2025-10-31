import unittest
from functools import partial

import numpy as np
import sympy as sp

from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.interface.wrappers import (
    create_active_set_variable_model,
    create_pool_model,
    ScipyModelWrapper,
    ChangeModelSignWrapper,
    create_active_set_qoi_model,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


def _pickable_function(bkd, sample):
    return bkd.stack(
        (bkd.sum(sample**2, axis=0), bkd.sum(sample**3, axis=0)), axis=1
    )


def _pickable_jacobian(bkd, sample):
    return bkd.stack((2 * sample[:, 0], 3 * sample[:, 0] ** 2), axis=0)


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
        model.work_tracker().set_active(True)

        active_var_indices = bkd.array([0, 2])
        nominal_sample = bkd.arange(1.0, nvars + 1)[:, None]
        inactive_var_values = nominal_sample[
            np.delete(np.arange(nvars), active_var_indices)
        ]
        wrapped_model = create_active_set_variable_model(
            model, nvars, inactive_var_values, active_var_indices
        )

        active_sample = nominal_sample[active_var_indices]
        assert isinstance(wrapped_model, ModelFromSingleSampleCallable)
        assert wrapped_model.nvars() == active_var_indices.shape[0]
        assert wrapped_model.nqoi() == model.nqoi()
        assert wrapped_model.noriginal_vars() == model.nvars()
        assert bkd.allclose(
            wrapped_model(active_sample), model(nominal_sample)
        )
        # wrapped_model._worktracker updates model._worktractker
        # so nevals must be 2 because the check abouve calls
        # model twice
        assert model.work_tracker().nevaluations("val") == 2

        assert bkd.allclose(
            wrapped_model.jacobian(active_sample),
            model.jacobian(nominal_sample)[:, active_var_indices],
        )
        assert model.work_tracker().nevaluations("jac") == 2
        vec = bkd.array(np.random.normal(0, 1, (nvars, 1)))
        assert bkd.allclose(
            wrapped_model.apply_jacobian(
                active_sample, vec[active_var_indices]
            ),
            model.jacobian(nominal_sample)[:, active_var_indices]
            @ vec[active_var_indices],
        )
        # Because jvp is not implemented jac will be called
        # twice more
        assert model.work_tracker().nevaluations("jac") == 4
        assert bkd.allclose(
            wrapped_model.apply_hessian(
                active_sample, vec[active_var_indices]
            ),
            model.hessian(nominal_sample)[0][
                np.ix_(active_var_indices, active_var_indices)
            ]
            @ vec[active_var_indices],
        )
        # Because hvp is not implemented hess will be called
        # twice
        assert model.work_tracker().nevaluations("hess") == 2

        # test plot runs
        wrapped_model.plot_contours(
            wrapped_model.get_plot_axis()[1], [0, 1, 0, 1]
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
        nvars, nsamples = 2, 4
        model = ModelFromSingleSampleCallable(
            2,
            nvars,
            partial(_pickable_function, bkd),
            jacobian=partial(_pickable_jacobian, bkd),
            backend=bkd,
        )
        wrapped_model = create_pool_model(model, nprocs=2, assert_omp=False)
        wrapped_model.model().work_tracker().set_active(True)
        wrapped_model.work_tracker().set_active(True)
        samples = bkd.asarray(np.random.uniform(0, 1, (nvars, nsamples)))
        values = wrapped_model(samples)
        assert bkd.allclose(values, _pickable_function(bkd, samples))
        assert wrapped_model.work_tracker().nevaluations("val") == 4
        assert wrapped_model.model().work_tracker().nevaluations("val") == 4
        errors = wrapped_model.check_apply_jacobian(samples[:, :1])
        # check functions from model are attributes of the wrapped model
        # E.g. check jacobian is correct.
        assert errors.min() / errors.max() < 1e-6

        # test plot runs
        wrapped_model.plot_contours(
            wrapped_model.get_plot_axis()[1], [0, 1, 0, 1]
        )

    def test_active_qoi_model(self):
        bkd = self.get_backend()

        def fun(x):
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

        def jac(x):
            return bkd.array([[2 * (x[0] - 1), 2 * (x[1] - 2.5)]])

        nvars = 2
        nqoi = 4
        model = ModelFromSingleSampleCallable(
            nqoi,
            nvars,
            lambda x: bkd.stack(
                [(ii + 1) * fun(x) for ii in range(nqoi)], axis=0
            ),
            jacobian=lambda x: bkd.vstack(
                [(ii + 1) * jac(x) for ii in range(nqoi)]
            ),
            sample_ndim=1,
            values_ndim=1,
            backend=bkd,
        )
        model.work_tracker().set_active(True)

        sample = bkd.arange(1.0, nvars + 1)[:, None]
        active_qoi_indices = bkd.array([0, 1], dtype=int)
        wrapped_model = create_active_set_qoi_model(model, active_qoi_indices)
        assert bkd.allclose(
            wrapped_model(sample), model(sample)[:, active_qoi_indices]
        )
        # wrapped_model._worktracker updates model._worktractker
        # so nevals must be 2 because the check abouve calls
        # model twice
        assert model.work_tracker().nevaluations("val") == 2
        assert bkd.allclose(
            wrapped_model.jacobian(sample),
            model.jacobian(sample)[active_qoi_indices],
        )
        assert model.work_tracker().nevaluations("jac") == 2
        vec = bkd.array(np.random.normal(0, 1, (nvars, 1)))
        assert bkd.allclose(
            wrapped_model.apply_jacobian(sample, vec),
            model.apply_jacobian(sample, vec)[active_qoi_indices],
        )
        # Because jvp is not implemented jac will be called
        # twice more
        assert model.work_tracker().nevaluations("jac") == 4

        # test plot runs
        wrapped_model.plot_contours(
            wrapped_model.get_plot_axis()[1], [0, 1, 0, 1]
        )


class TestNumpyWrappers(TestWrappers, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchWrappers(TestWrappers, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
