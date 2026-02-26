"""Tests for zoo factory functions."""

import pytest
import math


from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.protocols import (
    FunctionProtocol,
    FunctionWithJacobianProtocol,
)
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.time.config import TimeIntegrationConfig
from pyapprox.pde.zoo.diffusion import (
    create_steady_diffusion_1d,
    create_transient_diffusion_1d,
)
def _make_kle_field_map(bkd, nodes, num_kle_terms=2):
    """Helper: create lognormal KLE field map on Chebyshev nodes."""
    npts = nodes.shape[0]
    mesh_coords = ((nodes + 1.0) / 2.0)[None, :]  # map [-1,1] -> [0,1]
    mean_log = bkd.zeros((npts,))
    return create_lognormal_kle_field_map(
        mesh_coords,
        mean_log,
        bkd,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
    )


class TestSteadyDiffusionZoo:
    def _create_model(self, bkd) :
        npts = 20
        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.mesh import TransformedMesh1D

        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        field_map = _make_kle_field_map(bkd, nodes)

        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        return create_steady_diffusion_1d(
            bkd=bkd,
            npts=npts,
            domain=(-1.0, 1.0),
            forcing=forcing,
            field_map=field_map,
        )

    def test_factory_produces_valid_model(self, bkd):
        """Zoo factory produces a working forward model."""
        fwd = self._create_model(bkd)
        samples = bkd.zeros((2, 1))
        result = fwd(samples)
        assert result.shape[0] == fwd.nqoi()
        assert result.shape[1] == 1

    def test_has_jacobian(self, bkd):
        """Zoo model has jacobian."""
        fwd = self._create_model(bkd)
        assert hasattr(fwd, "jacobian")

    def test_isinstance_function_protocol(self, bkd):
        """Zoo model satisfies FunctionProtocol."""
        fwd = self._create_model(bkd)
        assert isinstance(fwd, FunctionProtocol)

    def test_isinstance_jacobian_protocol(self, bkd):
        """Zoo model satisfies FunctionWithJacobianProtocol."""
        fwd = self._create_model(bkd)
        assert isinstance(fwd, FunctionWithJacobianProtocol)

    def test_jacobian_derivative_checker(self, bkd):
        """Zoo model Jacobian passes DerivativeChecker."""
        fwd = self._create_model(bkd)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(sample)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5

    def test_matches_manual_construction(self, bkd):
        """Zoo factory matches manually constructed forward model."""
        npts = 20
        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
        from pyapprox.pde.collocation.forward_models.steady import (
            SteadyForwardModel,
        )
        from pyapprox.pde.collocation.mesh import (
            TransformedMesh1D,
            create_uniform_mesh_1d,
        )
        from pyapprox.pde.collocation.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.pde.parameterizations.diffusion import (
            create_diffusion_parameterization,
        )

        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
        nodes = basis.nodes()

        field_map = _make_kle_field_map(bkd, nodes)

        def forcing(t):
            return (math.pi**2) * bkd.sin(math.pi * nodes)

        physics = AdvectionDiffusionReaction(
            basis,
            bkd,
            diffusion=1.0,
            forcing=forcing,
        )
        left_idx = mesh_obj.boundary_indices(0)
        right_idx = mesh_obj.boundary_indices(1)
        physics.set_boundary_conditions(
            [
                zero_dirichlet_bc(bkd, left_idx),
                zero_dirichlet_bc(bkd, right_idx),
            ]
        )
        param = create_diffusion_parameterization(bkd, basis, field_map)
        init_state = bkd.zeros((npts,))
        fwd_manual = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param
        )

        fwd_zoo = self._create_model(bkd)

        samples = bkd.zeros((2, 1))
        bkd.assert_allclose(
            fwd_zoo(samples), fwd_manual(samples), rtol=1e-10, atol=1e-14
        )
        bkd.assert_allclose(
            fwd_zoo.jacobian(samples),
            fwd_manual.jacobian(samples),
            rtol=1e-10,
            atol=1e-14,
        )

    def test_cannot_specify_both_field_map_and_basis_funs(self, bkd):
        """Raises ValueError if both field_map and basis_funs given."""
        npts = 10
        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.mesh import TransformedMesh1D

        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        field_map = _make_kle_field_map(bkd, nodes)

        with pytest.raises(ValueError):
            create_steady_diffusion_1d(
                bkd=bkd,
                npts=npts,
                domain=(-1.0, 1.0),
                forcing=lambda t: bkd.zeros((npts,)),
                field_map=field_map,
                basis_funs=[bkd.ones((npts,))],
            )


class TestTransientDiffusionZoo:
    def _create_model(self, bkd) :
        npts = 15
        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.mesh import TransformedMesh1D

        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()

        field_map = _make_kle_field_map(bkd, nodes)

        time_config = TimeIntegrationConfig(
            method="backward_euler",
            init_time=0.0,
            final_time=0.1,
            deltat=0.02,
            newton_tol=1e-10,
            newton_maxiter=20,
        )

        return create_transient_diffusion_1d(
            bkd=bkd,
            npts=npts,
            domain=(-1.0, 1.0),
            init_state_func=lambda nodes: bkd.sin(math.pi * nodes),
            time_config=time_config,
            field_map=field_map,
        )

    def test_factory_produces_valid_model(self, bkd):
        """Zoo factory produces a working transient forward model."""
        fwd = self._create_model(bkd)
        samples = bkd.zeros((2, 1))
        result = fwd(samples)
        assert result.shape[0] == fwd.nqoi()
        assert result.shape[1] == 1

    def test_has_jacobian(self, bkd):
        """Zoo transient model has jacobian."""
        fwd = self._create_model(bkd)
        assert hasattr(fwd, "jacobian")

    def test_isinstance_protocols(self, bkd):
        """Zoo transient model satisfies expected protocols."""
        fwd = self._create_model(bkd)
        assert isinstance(fwd, FunctionProtocol)
        assert isinstance(fwd, FunctionWithJacobianProtocol)

    def test_jacobian_derivative_checker(self, bkd):
        """Zoo transient model Jacobian passes DerivativeChecker."""
        fwd = self._create_model(bkd)
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(sample, direction=None, relative=True)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5


class TestDiffusionPositivityValidation:
    """Tests for strict positivity validation in DiffusionParameterization."""
    def _make_param_and_physics(self, bkd, npts) :
        from pyapprox.pde.collocation.basis import ChebyshevBasis1D
        from pyapprox.pde.collocation.mesh import TransformedMesh1D
        from pyapprox.pde.collocation.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.pde.field_maps.basis_expansion import (
            BasisExpansion,
        )
        from pyapprox.pde.parameterizations.diffusion import (
            create_diffusion_parameterization,
        )

        mesh = TransformedMesh1D(npts, bkd)
        basis = ChebyshevBasis1D(mesh, bkd)
        physics = AdvectionDiffusionReaction(basis, bkd, diffusion=1.0)
        # Single basis function = ones, so field = base + p[0]*ones
        fm = BasisExpansion(bkd, 0.0, [bkd.ones((npts,))])
        param = create_diffusion_parameterization(bkd, basis, fm)
        return param, physics

    def test_nonpositive_diffusion_raises(self, bkd):
        """ValueError raised when parameterized diffusion is non-positive."""
        npts = 5
        param, physics = self._make_param_and_physics(bkd, npts)
        # field = 0.0 + (-0.1)*ones = -0.1 everywhere
        with pytest.raises(ValueError) as ctx:
            param.apply(physics, bkd.array([-0.1]))
        assert "positive" in str(ctx.value)

    def test_zero_diffusion_raises(self, bkd):
        """ValueError raised when parameterized diffusion is zero."""
        npts = 5
        param, physics = self._make_param_and_physics(bkd, npts)
        with pytest.raises(ValueError):
            param.apply(physics, bkd.array([0.0]))

    def test_positive_diffusion_succeeds(self, bkd):
        """No error when parameterized diffusion is positive."""
        npts = 5
        param, physics = self._make_param_and_physics(bkd, npts)
        # field = 0.0 + 1.5*ones = 1.5 everywhere
        param.apply(physics, bkd.array([1.5]))  # Should not raise
