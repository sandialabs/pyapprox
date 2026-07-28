"""Steering a plume: distributed-forcing control of obstructed transport.

A fixed Gaussian release near the inlet feeds a contaminant plume that
a frozen Navier-Stokes flow advects through the gaps of three staggered
blocks toward a protected zone above the uppermost block. K Gaussian
actuators (amplitudes may be negative: sinks) add controllable forcing;
the objective trades time-integrated zone contamination against a
quadratic actuation cost:

    J(p) = int_0^T int w(x) u(x, t; p)^2 dx dt + (alpha/2) ||p||^2.

The transport is linear in the state and affine in ``p``, so J is a
strictly convex quadratic; the interest is that gradients and
Hessian-vector products come from one adjoint sweep each, at a cost
independent of K.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Generic, Optional, Tuple

if TYPE_CHECKING:
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.zone_weights import ZoneWeightProtocol
    from pyapprox.pde.models.galerkin.transient import (
        GalerkinTransientForwardModel,
    )

import numpy as np
from pyapprox.util.backends.protocols import Array, Backend

# Actuator layout: label -> center. Bumps sit in the gap channels and
# along the flow path (see build docstring); two deliberately
# low-leverage placements (inside the zone's feed path vs downstream of
# the zone) make the learned strategy legible.
_DEFAULT_ACTUATORS: Dict[str, Tuple[float, float]] = {
    "below_B_gap": (0.357, 0.125),
    "above_B": (0.357, 0.60),
    "corridor_mid": (0.50, 0.375),
    "gap_AC": (0.643, 0.375),
    "corridor_upper": (0.50, 0.65),
    "zone_inlet_left": (0.50, 0.85),
    "outflow_lower": (0.85, 0.45),
    "downstream_of_zone": (0.90, 0.85),
}

# The rectangle directly above the uppermost block: edges on coarse
# tensor-grid lines, so ElementAlignedRectangleZone assembles exactly
# at every refinement level.
_ZONE_XLIM = (4.0 / 7.0, 5.0 / 7.0)
_ZONE_YLIM = (0.75, 1.0)


class ObstructedFlowControlProblem(Generic[Array]):
    """Distributed-forcing control of transport through an obstructed
    channel.

    Composes the obstructed-flow substrate
    (:mod:`pyapprox.pde.zoo.obstructed_flow`) with the transient
    galerkin adjoint stack: the returned model satisfies
    ``ObjectiveProtocol`` with an adjoint gradient and a
    second-order-adjoint HVP, ready for
    ``ScipyTrustConstrOptimizer(...).bind(problem.model(),
    problem.bounds())``.

    Parameters
    ----------
    bkd : Backend
        Computational backend. Must be NumpyBkd: the forward solve
        uses skfem, which is NumPy-only.
    nstokes_refine : int
        Uniform refinements of the Navier-Stokes mesh.
    ntransport_refine : int
        Uniform refinements of the transport mesh.
    reynolds_num : float
        Reynolds number of the frozen flow (viscosity ``1/Re``).
    vel_shape_params : tuple of 2 floats
        Parabolic-inlet shape parameters ``(a, b)``.
    diffusivity : float
        Constant transport diffusivity. The default gives an
        advection-dominated plume (domain Peclet ~8-27 at the realized
        speeds) with cell Peclet ~0.5 on the default meshes, so
        unstabilized Galerkin remains oscillation-free; coarser meshes
        (e.g. one refinement) need a larger value to keep the cell
        Peclet below one.
    final_time : float
        Transport horizon T. The default matches the release-to-zone
        transit time of the frozen flow (path speeds ~0.05); much
        shorter horizons never deliver the plume to the zone.
    deltat : float
        Time step.
    alpha : float
        Actuation cost coefficient (exchange rate between zone
        contamination and pumping effort). The default fully prices
        out the source-sink cancellation direction (smaller alpha lets
        the optimizer pair positive sources with overshooting sinks to
        cancel negative concentration) yielding an all-sink optimum
        with a ~20x contamination reduction.
    release_center, release_width, release_amplitude : floats
        Fixed Gaussian release upstream of the blocks.
    actuator_centers : dict, optional
        ``label -> (x, y)`` actuator layout; defaults to gap-channel
        placements.
    actuator_width : float
        Gaussian actuator width.
    control_bound : float
        Symmetric amplitude bound ``|p_k| <= control_bound``.
    zone_weight : ZoneWeightProtocol, optional
        Spatial weighting of the protected zone. Defaults to the
        element-aligned rectangle above the uppermost block.
    time_integrator : str
        ``"crank_nicolson"`` (default) or ``"backward_euler"``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        *,
        nstokes_refine: int = 2,
        ntransport_refine: int = 2,
        reynolds_num: float = 20.0,
        vel_shape_params: Tuple[float, float] = (2.5, 2.5),
        diffusivity: float = 0.01,
        final_time: float = 20.0,
        deltat: float = 0.5,
        alpha: float = 1e-3,
        release_center: Tuple[float, float] = (0.15, 0.4),
        release_width: float = 0.05,
        release_amplitude: float = 10.0,
        actuator_centers: Optional[Dict[str, Tuple[float, float]]] = None,
        actuator_width: float = 0.05,
        control_bound: float = 10.0,
        zone_weight: Optional["ZoneWeightProtocol[Array]"] = None,
        time_integrator: str = "crank_nicolson",
    ) -> None:
        from pyapprox.util.backends.numpy import NumpyBkd

        # Assigned before the isinstance guard so the runtime
        # narrowing does not leak the concrete backend into the
        # generic Array parameter.
        self._bkd: Backend[Array] = bkd
        if not isinstance(bkd, NumpyBkd):
            raise TypeError(
                "ObstructedFlowControlProblem requires NumpyBkd because "
                "its forward solve uses skfem, which is NumPy-only. Got "
                f"{type(bkd).__name__}."
            )
        if time_integrator not in ("backward_euler", "crank_nicolson"):
            raise ValueError(
                f"time_integrator must be 'backward_euler' or "
                f"'crank_nicolson', got {time_integrator!r}"
            )
        if actuator_centers is None:
            actuator_centers = dict(_DEFAULT_ACTUATORS)
        self._actuator_labels = tuple(actuator_centers)
        self._actuator_centers = np.array(
            [actuator_centers[label] for label in self._actuator_labels]
        ).T
        self._actuator_width = float(actuator_width)
        self._release_center = release_center
        self._release_width = float(release_width)
        self._release_amplitude = float(release_amplitude)
        self._alpha = float(alpha)
        self._control_bound = float(control_bound)

        self._build_model(
            nstokes_refine,
            ntransport_refine,
            reynolds_num,
            vel_shape_params,
            diffusivity,
            final_time,
            deltat,
            zone_weight,
            time_integrator,
        )

    def _gaussian_nodal(
        self,
        coords: np.ndarray,
        center: Tuple[float, float],
        width: float,
    ) -> np.ndarray:
        """Evaluate a unit Gaussian bump at nodal coordinates."""
        return np.asarray(
            np.exp(
                -(
                    (coords[0] - center[0]) ** 2
                    + (coords[1] - center[1]) ** 2
                )
                / (2.0 * width**2)
            )
        )

    def _build_model(
        self,
        nstokes_refine: int,
        ntransport_refine: int,
        reynolds_num: float,
        vel_shape_params: Tuple[float, float],
        diffusivity: float,
        final_time: float,
        deltat: float,
        zone_weight: Optional["ZoneWeightProtocol[Array]"],
        time_integrator: str,
    ) -> None:
        from pyapprox.ode.config import TimeIntegrationConfig
        from pyapprox.ode.functionals.tikhonov import (
            TikhonovAugmentedFunctional,
        )
        from pyapprox.ode.functionals.time_integrated_weighted_l2 import (
            TimeIntegratedWeightedL2Functional,
        )
        from pyapprox.pde.constitutive.coefficient_functions import (
            NodalFieldDiffusion,
            NodalFieldForcing,
        )
        from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
        from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
        from pyapprox.pde.galerkin.boundary.implementations import (
            DirichletBC,
        )
        from pyapprox.pde.galerkin.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.pde.galerkin.zone_weights import (
            ElementAlignedRectangleZone,
        )
        from pyapprox.pde.models.galerkin.transient import (
            GalerkinTransientForwardModel,
        )
        from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
            AdvectionDiffusionParameterization,
        )
        from pyapprox.pde.zoo.obstructed_flow import (
            build_obstructed_mesh,
            extract_velocity_callable,
            solve_obstructed_stokes,
        )

        bkd = self._bkd

        # Frozen flow: solved once on its own (finer) mesh.
        stokes_mesh = build_obstructed_mesh(bkd, nstokes_refine)
        stokes_sol, stokes, vel_basis, pres_basis = solve_obstructed_stokes(
            stokes_mesh, bkd, reynolds_num, list(vel_shape_params)
        )

        transport_mesh = build_obstructed_mesh(bkd, ntransport_refine)
        self._basis: "LagrangeBasis[Array]" = LagrangeBasis(
            transport_mesh, degree=1
        )
        self._velocity = extract_velocity_callable(
            stokes_sol, stokes, vel_basis, pres_basis, self._basis, bkd
        )

        coords = bkd.to_numpy(self._basis.dof_coordinates())
        release_nodal = self._release_amplitude * self._gaussian_nodal(
            coords, self._release_center, self._release_width
        )
        actuator_fields = [
            bkd.asarray(
                self._gaussian_nodal(
                    coords,
                    (
                        float(self._actuator_centers[0, kk]),
                        float(self._actuator_centers[1, kk]),
                    ),
                    self._actuator_width,
                )
            )
            for kk in range(self.ncontrols())
        ]
        # Affine control map: forcing dofs = release + sum_k p_k q_k.
        forcing_map = BasisExpansion(
            bkd, bkd.asarray(release_nodal), actuator_fields
        )

        physics = AdvectionDiffusionReaction(
            basis=self._basis,
            diffusivity=NodalFieldDiffusion(
                self._basis,
                dofs=diffusivity * np.ones(self._basis.ndofs()),
            ),
            bkd=bkd,
            velocity=self._velocity,
            forcing=NodalFieldForcing(self._basis, dofs=release_nodal),
            boundary_conditions=[
                DirichletBC(self._basis, "left", 0.0, bkd)
            ],
        )
        parameterization = AdvectionDiffusionParameterization(
            physics, forcing_map=forcing_map, bkd=bkd
        )

        if zone_weight is None:
            zone_weight = ElementAlignedRectangleZone(
                _ZONE_XLIM, _ZONE_YLIM
            )
        self._zone_weight: "ZoneWeightProtocol[Array]" = zone_weight
        weight_matrix = zone_weight.assemble_weighted_mass(self._basis, bkd)

        functional = TikhonovAugmentedFunctional(
            TimeIntegratedWeightedL2Functional(
                weight_matrix, self.ncontrols(), bkd
            ),
            self._alpha,
            bkd,
        )

        config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
            method=time_integrator,
            init_time=0.0,
            final_time=final_time,
            deltat=deltat,
            newton_tol=1e-12,
            newton_maxiter=5,
            lumped_mass=False,
            verbosity=0,
        )
        self._model: "GalerkinTransientForwardModel[Array]" = (
            GalerkinTransientForwardModel(
                physics,
                parameterization,
                bkd.zeros((physics.nstates(),)),
                config,
                bkd,
                functional=functional,
            )
        )

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def ncontrols(self) -> int:
        """Return the number of actuator amplitudes K."""
        return len(self._actuator_labels)

    def model(self) -> "GalerkinTransientForwardModel[Array]":
        """Return the control objective J(p).

        Satisfies ``ObjectiveProtocol``: adjoint gradient and
        second-order-adjoint HVP via ``derivatives()``.
        """
        return self._model

    def bounds(self) -> Array:
        """Return amplitude bounds. Shape: ``(ncontrols, 2)``."""
        bound = self._control_bound
        return self._bkd.asarray(
            np.tile([-bound, bound], (self.ncontrols(), 1))
        )

    def release_center(self) -> Tuple[float, float]:
        """Return the fixed release center (for plotting)."""
        return self._release_center

    def actuator_labels(self) -> Tuple[str, ...]:
        """Return the actuator labels in parameter order."""
        return self._actuator_labels

    def actuator_centers(self) -> np.ndarray:
        """Return actuator centers. Shape: ``(2, ncontrols)``."""
        return self._actuator_centers

    def zone_weight(self) -> "ZoneWeightProtocol[Array]":
        """Return the protected-zone weighting."""
        return self._zone_weight

    def basis(self) -> "LagrangeBasis[Array]":
        """Return the transport basis (for plotting)."""
        return self._basis

    def velocity(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the frozen-flow velocity callable (for plotting)."""
        return self._velocity

    def alpha(self) -> float:
        """Return the actuation cost coefficient."""
        return self._alpha

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ncontrols={self.ncontrols()}, "
            f"alpha={self._alpha})"
        )
