r"""The 1D heat problem with a moving Dirichlet end, parameterized.

:math:`u_t = (\kappa(x) u')' + (1 + x)(1 + \sin 5t)` on [0, 1], P1 on 10
cells, :math:`\kappa = \exp(W p)` from a three-term log-KLE (modes
:math:`0.4 \sin((k+1)\pi x)/(k+1)`), :math:`u(0, t) = 0` and a moving
right end :math:`u(1, t) = 0.5 \sin 4t` with its rate :math:`2 \cos 4t`
supplied, from :math:`u(x, 0) = 0.3 \sin \pi x`.

Summed over every row, the diffusion term vanishes (the interpolant of
1 annihilates it) and the load is :math:`\int (1 + x)(1 + \sin 5t)\,dx =
1.5 (1 + \sin 5t)`, so the content :math:`\int u_h` changes only by the
source and the flux through both ends.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeDependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd

_Np = NDArray[Any]

HEAT_NPARAMS = 3
HEAT_PARAM: _Np = np.array([[0.4], [-0.3], [0.2]])
"""The log-KLE coefficients the problem is solved at, as a column."""


def heat_forcing(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * (1.0 + np.sin(5.0 * t))
    return value


def heat_source_total(t: float) -> float:
    r""":math:`\int_0^1` of the forcing at time ``t``."""
    return 1.5 * (1.0 + float(np.sin(5.0 * t)))


def _right_value(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 0.5 * np.sin(4.0 * t))
    return value


def _right_rate(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 2.0 * np.cos(4.0 * t))
    return value


class MovingEndHeat:
    """The parameterized heat system, its initial state, its end DOFs and
    the DOF next to the moving end; ``supply_rate`` False leaves the moving
    end's rate undeclared."""

    def __init__(self, bkd: NumpyBkd, supply_rate: bool = True) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            forcing=TimeDependent(heat_forcing),
        )
        right = BoundarySignal(
            TimeDependent(_right_value),
            [TimeDependent(_right_rate)] if supply_rate else [],
        )
        self.system: GalerkinSystem[_Np] = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", right, bkd),
            ],
        )
        x = np.asarray(basis.dof_coordinates())[0]
        self.x: _Np = x
        modes = np.stack(
            [
                0.4 * np.sin((k + 1) * np.pi * x) / (k + 1)
                for k in range(HEAT_NPARAMS)
            ],
            axis=1,
        )
        exp = _ExpTransform(bkd)
        self.parameterization = AdvectionDiffusionParameterization(
            physics,
            diffusivity_map=TransformedFieldMap(
                MeshKLEFieldMap(bkd, bkd.zeros((x.shape[0],)), bkd.asarray(modes)),
                exp,
                exp,
                bkd,
                transform_deriv2=exp,
            ),
            bkd=bkd,
        )
        self.init_state: _Np = 0.3 * np.sin(np.pi * x)
        self.ends: _Np = np.concatenate(
            [np.asarray(basis.get_dofs("left")), np.asarray(basis.get_dofs("right"))]
        )
        interior = np.setdiff1d(
            np.arange(x.shape[0]), np.asarray(self.system.constraint_set().dofs())
        )
        self.near_right = int(interior[np.argmax(x[interior])])

    def content(self, state: _Np) -> float:
        r""":math:`\int_0^1 u_h` (the trapezoid rule on the nodes, exact for
        P1); ``state`` of any shape holding one value per DOF."""
        order = np.argsort(self.x)
        return float(np.trapezoid(np.ravel(state)[order], self.x[order]))
