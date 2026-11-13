r"""Two parameterized steady Galerkin problems with time-invariant data.

``steady_adr``: :math:`-(\kappa u')' (+ u^2) = 1` on [0, 1], u(0) = u(1)
= 0, P1 on 10 cells, :math:`\kappa = \exp(W p)` with the three modes
:math:`0.4 \sin((k+1)\pi x)/(k+1)`; ``ADR_PARAM`` = (0.4, -0.3, 0.2).

``elastic_plate``: 2D linear elasticity on [0, 1]^2, P1 vector elements
on 6 x 6 cells, Lame values from (E, nu) through
``create_galerkin_lame_parameterization``; clamped on the left, a Robin
support :math:`\sigma n + 2 u = (0.1, -0.05)` on the right, a traction
(0.2, 0.05) on the top and a body force (0.3, -0.1); ``PLATE_PARAM`` =
(1, 0.3).
"""

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    TimeIndependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import TransformedFieldMap, _ExpTransform
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.util.backends.numpy import NumpyBkd

_Np = NDArray[Any]

ADR_NPARAMS = 3
ADR_PARAM = np.array([[0.4], [-0.3], [0.2]])
PLATE_PARAM = np.array([[1.0], [0.3]])


def _unit_forcing(x: _Np) -> _Np:
    return np.ones(x.shape[1])


def _square(x: _Np, u: _Np) -> _Np:
    return u**2


def _square_deriv(x: _Np, u: _Np) -> _Np:
    return 2.0 * u


def _square_deriv2(x: _Np, u: _Np) -> _Np:
    return np.full_like(u, 2.0)


def steady_adr(
    bkd: NumpyBkd, nonlinear: bool, curvature: bool = True
) -> Tuple[
    GalerkinSystem[_Np],
    AdvectionDiffusionParameterization[_Np],
    LagrangeBasis[_Np],
]:
    """The 1D diffusion problem, with the reaction :math:`u^2` if
    ``nonlinear``: the system, its parameterization and its basis.
    Without ``curvature`` the reaction declares no second derivative, so
    the physics has no state curvature."""
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    reaction = (
        CallableReaction(_square, _square_deriv, _square_deriv2 if curvature else None)
        if nonlinear
        else None
    )
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        reaction=reaction,
        forcing=TimeIndependent(_unit_forcing),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
        ],
    )
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = np.stack(
        [0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1) for k in range(ADR_NPARAMS)],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    parameterization = AdvectionDiffusionParameterization(
        physics,
        diffusivity_map=TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp),
        bkd=bkd,
    )
    return system, parameterization, basis


class _ConstantVector:
    """A constant vector field, shape (2, npts); picklable."""

    def __init__(self, value: _Np) -> None:
        self._value = np.asarray(value, dtype=float)

    def __call__(self, coords: _Np) -> _Np:
        return np.outer(self._value, np.ones(coords.shape[1]))


def elastic_plate(
    bkd: NumpyBkd,
) -> Tuple[GalerkinSystem[_Np], ParameterizationProtocol[_Np]]:
    """The elastic plate: the system and its (E, nu) parameterization."""
    mesh = StructuredMesh2D(nx=6, ny=6, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
    basis = VectorLagrangeBasis(mesh, degree=1)
    physics = CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        bkd=bkd,
        body_force=TimeIndependent(_ConstantVector(np.array([0.3, -0.1]))),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 0.0, bkd),
            RobinBC(
                basis,
                "right",
                2.0,
                TimeIndependent(_ConstantVector(np.array([0.1, -0.05]))),
                bkd,
            ),
            NeumannBC(
                basis,
                "top",
                TimeIndependent(_ConstantVector(np.array([0.2, 0.05]))),
                bkd,
            ),
        ],
    )
    return system, create_galerkin_lame_parameterization(physics, bkd)
