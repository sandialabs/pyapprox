"""Tests for YoungModulusParameterization for 2D linear elasticity.

Verifies:
1. isinstance check against ParameterizationProtocol
2. apply sets mu and lambda correctly
3. param_jacobian matches finite differences via DerivativeChecker
4. initial_param_jacobian returns zeros
5. Dynamic binding (with/without field_map.jacobian)
6. Dual backend support (NumPy and PyTorch)
"""

import math
import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import (
    create_uniform_mesh_2d,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapProtocol,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.pde.parameterizations.lame import (
    YoungModulusParameterization,
    create_youngs_modulus_parameterization,
)


def _create_elasticity_physics_and_basis(bkd, npts_1d=6):
    """Create 2D linear elasticity physics with basis for testing."""
    mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    npts = basis.npts()

    nodes_x = basis.nodes_x()
    nodes_y = basis.nodes_y()
    xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
    nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

    # Constant forcing for a non-trivial problem
    forcing_flat = bkd.concatenate([
        bkd.ones((npts,)),
        bkd.zeros((npts,)),
    ])

    physics = LinearElasticityPhysics(
        basis, bkd, lamda=1.0, mu=1.0, forcing=lambda t: forcing_flat
    )
    return physics, basis, nodes


class TestLameParameterization(Generic[Array], unittest.TestCase):
    """Tests for YoungModulusParameterization."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_isinstance_protocol(self):
        """YoungModulusParameterization satisfies ParameterizationProtocol."""
        bkd = self._bkd
        physics, basis, nodes = _create_elasticity_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        param = create_youngs_modulus_parameterization(bkd, basis, fm, 0.3)
        self.assertTrue(isinstance(param, ParameterizationProtocol))

    def test_type_error_non_field_map(self):
        """TypeError when passing non-FieldMap object."""
        bkd = self._bkd
        with self.assertRaises(TypeError):
            YoungModulusParameterization("not_a_field_map", [], bkd, 0.3)

    def test_apply_sets_lame_params(self):
        """apply() correctly converts E to mu and lambda."""
        bkd = self._bkd
        physics, basis, nodes = _create_elasticity_physics_and_basis(bkd)
        npts = basis.npts()

        # Field map: E(x) = 2.0 + p0 * ones
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 2.0, [phi0])
        nu = 0.3
        param = create_youngs_modulus_parameterization(bkd, basis, fm, nu)

        params = bkd.array([0.5])  # E = 2.5
        param.apply(physics, params)

        E_val = 2.5
        expected_mu = E_val / (2.0 * (1.0 + nu))
        expected_lam = E_val * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        bkd.assert_allclose(
            physics._mu_array, bkd.full((npts,), expected_mu), rtol=1e-12
        )
        bkd.assert_allclose(
            physics._lambda_array,
            bkd.full((npts,), expected_lam),
            rtol=1e-12,
        )

    def test_nparams(self):
        """nparams matches field_map.nvars()."""
        bkd = self._bkd
        physics, basis, nodes = _create_elasticity_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        phi1 = nodes[0, :]  # x-coordinate
        fm = BasisExpansion(bkd, 1.0, [phi0, phi1])
        param = create_youngs_modulus_parameterization(bkd, basis, fm, 0.3)
        self.assertEqual(param.nparams(), 2)
        self.assertEqual(param.nparams(), fm.nvars())

    def test_initial_param_jacobian_zeros(self):
        """initial_param_jacobian returns zeros of correct shape."""
        bkd = self._bkd
        physics, basis, nodes = _create_elasticity_physics_and_basis(bkd)
        npts = basis.npts()
        nstates = 2 * npts
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 1.0, [phi0])
        param = create_youngs_modulus_parameterization(bkd, basis, fm, 0.3)
        params = bkd.array([0.5])
        result = param.initial_param_jacobian(physics, params)
        expected = bkd.zeros((nstates, 1))
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_dynamic_binding_without_jacobian(self):
        """No param_jacobian when field_map lacks jacobian method."""
        bkd = self._bkd

        class NoJacFieldMap:
            """Field map without jacobian method."""

            def nvars(self):
                return 1

            def __call__(self, params_1d):
                return bkd.full((10,), 1.0)

        fm = NoJacFieldMap()
        # Must satisfy FieldMapProtocol at least structurally
        self.assertTrue(isinstance(fm, FieldMapProtocol))
        param = YoungModulusParameterization(fm, [], bkd, 0.3)
        self.assertFalse(hasattr(param, "param_jacobian"))

    def test_nonpositive_E_raises(self):
        """apply() raises ValueError when E field is non-positive."""
        bkd = self._bkd
        physics, basis, nodes = _create_elasticity_physics_and_basis(bkd)
        npts = basis.npts()
        phi0 = bkd.ones((npts,))
        fm = BasisExpansion(bkd, 0.5, [phi0])
        param = create_youngs_modulus_parameterization(bkd, basis, fm, 0.3)
        # E = 0.5 + (-1.0)*ones = -0.5, non-positive
        params = bkd.array([-1.0])
        with self.assertRaises(ValueError):
            param.apply(physics, params)

    def test_param_jacobian_fd(self):
        """param_jacobian matches FD via DerivativeChecker."""
        bkd = self._bkd
        physics, basis, nodes = _create_elasticity_physics_and_basis(bkd)
        npts = basis.npts()
        nstates = 2 * npts

        # Two-parameter field map: E(x) = 2.0 + p0*1 + p1*cos(pi*x)
        x = nodes[0, :]
        phi0 = bkd.ones((npts,))
        phi1 = bkd.cos(math.pi * x)
        fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
        nu = 0.3
        param = create_youngs_modulus_parameterization(bkd, basis, fm, nu)

        # Non-trivial state
        y = nodes[1, :]
        state = bkd.concatenate([
            bkd.sin(math.pi * x) * bkd.cos(math.pi * y) * 0.1,
            bkd.cos(math.pi * x) * bkd.sin(math.pi * y) * 0.1,
        ])
        time = 0.0

        def residual_of_params(samples):
            results = []
            for i in range(samples.shape[1]):
                p = samples[:, i]
                param.apply(physics, p)
                res = physics.residual(state, time)
                results.append(res)
            return bkd.stack(results, axis=1)

        def jac_of_params(sample):
            p = sample[:, 0]
            param.apply(physics, p)
            return param.param_jacobian(physics, state, time, p)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=param.nparams(),
            fun=residual_of_params,
            jacobian=jac_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        params = bkd.array([0.1, -0.05])[:, None]
        errors = checker.check_derivatives(params, verbosity=0)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        self.assertLessEqual(ratio, 1e-5)


class TestLameParameterizationNumpy(TestLameParameterization[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLameParameterizationTorch(TestLameParameterization[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
