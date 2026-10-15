"""The affine-in-parameters declaration of Galerkin ADR parameterizations.

Setup: -(kappa u')' = f on [0, 1], u(0) = 1, u(1) = 2, P1 on 10 cells,
with kappa and f nodal fields. A linear map is kappa = 1 + W p with
three smooth modes (amplitude 0.2, so kappa stays positive); a nonlinear
map is kappa = exp(W p).

The declaration is checked against the residual, not against the code
that computes it: F(u, p) is affine in p exactly when, for every u, p1,
p2 and alpha,

    F(u, alpha p1 + (1 - alpha) p2) = alpha F(u, p1) + (1 - alpha) F(u, p2).

The raw residual (before row replacement) is evaluated at random u and
p. Wrong-model control: the exponential map is declared not affine and
the identity fails for it, by far more than roundoff.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    NodalFieldForcing,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.models.galerkin import GalerkinParameterizedResidual
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.protocol import AffineInParamsProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NumpyArray


class _Problem:
    def __init__(self, bkd: NumpyBkd) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        self.basis = LagrangeBasis(mesh, degree=1)
        self.physics = AdvectionDiffusionReaction(
            basis=self.basis,
            diffusivity=NodalFieldDiffusion(self.basis),
            bkd=bkd,
            forcing=NodalFieldForcing(self.basis),
        )
        self.system = compose_galerkin_system(
            self.physics,
            [
                DirichletBC(self.basis, "left", 1.0, bkd),
                DirichletBC(self.basis, "right", 2.0, bkd),
            ],
        )
        self.coords = bkd.to_numpy(self.basis.dof_coordinates())[0]
        self.bkd = bkd

    def _modes(self) -> NumpyArray:
        return np.stack(
            [0.2 * np.sin((k + 1) * np.pi * self.coords) for k in range(3)],
            axis=1,
        )

    def linear_map(self, mean: float) -> MeshKLEFieldMap[NumpyArray]:
        bkd = self.bkd
        return MeshKLEFieldMap(
            bkd,
            bkd.asarray(np.full(self.coords.shape[0], mean)),
            bkd.asarray(self._modes()),
        )

    def exp_map(self) -> TransformedFieldMap[NumpyArray]:
        bkd = self.bkd
        exp = _ExpTransform(bkd)
        return TransformedFieldMap(
            self.linear_map(0.0), exp, exp, bkd, transform_deriv2=exp
        )

    def residual(
        self, param_obj: AdvectionDiffusionParameterization[NumpyArray]
    ) -> GalerkinParameterizedResidual[NumpyArray]:
        return GalerkinParameterizedResidual(self.system.spatial_operator(), param_obj)


def _line_identity_gap(
    residual: GalerkinParameterizedResidual[NumpyArray], seed: int
) -> float:
    """max |F(u, a p1 + (1-a) p2) - a F(u, p1) - (1-a) F(u, p2)| / max |F|."""
    rng = np.random.default_rng(seed)
    nparams = residual.nparams()
    state = rng.uniform(1.0, 2.0, size=residual.nstates())
    p1 = rng.uniform(-1.0, 1.0, size=nparams)
    p2 = rng.uniform(-1.0, 1.0, size=nparams)
    alpha = 0.3
    mixed = residual.residual(state, 0.0, alpha * p1 + (1.0 - alpha) * p2)
    combined = alpha * residual.residual(state, 0.0, p1) + (
        1.0 - alpha
    ) * residual.residual(state, 0.0, p2)
    return float(np.max(np.abs(mixed - combined)) / np.max(np.abs(combined)))


class TestAffineDeclaration:
    def test_linear_diffusivity_is_affine(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        param_obj = AdvectionDiffusionParameterization(
            problem.physics, diffusivity_map=problem.linear_map(1.0), bkd=numpy_bkd
        )
        assert isinstance(param_obj, AffineInParamsProtocol)
        assert param_obj.is_affine_in_params()
        assert _line_identity_gap(problem.residual(param_obj), seed=0) < 1e-13

    def test_linear_diffusivity_and_forcing_is_affine(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        problem = _Problem(numpy_bkd)
        param_obj = AdvectionDiffusionParameterization(
            problem.physics,
            diffusivity_map=problem.linear_map(1.0),
            forcing_map=problem.linear_map(0.0),
            bkd=numpy_bkd,
        )
        assert param_obj.is_affine_in_params()
        assert _line_identity_gap(problem.residual(param_obj), seed=1) < 1e-13

    def test_exponential_diffusivity_is_not_affine(self, numpy_bkd: NumpyBkd) -> None:
        problem = _Problem(numpy_bkd)
        param_obj = AdvectionDiffusionParameterization(
            problem.physics, diffusivity_map=problem.exp_map(), bkd=numpy_bkd
        )
        assert not param_obj.is_affine_in_params()
        assert _line_identity_gap(problem.residual(param_obj), seed=2) > 1e-8

    def test_one_nonlinear_part_makes_composite_not_affine(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        problem = _Problem(numpy_bkd)
        param_obj = AdvectionDiffusionParameterization(
            problem.physics,
            diffusivity_map=problem.linear_map(1.0),
            forcing_map=problem.exp_map(),
            bkd=numpy_bkd,
        )
        assert not param_obj.is_affine_in_params()
        assert _line_identity_gap(problem.residual(param_obj), seed=3) > 1e-8
