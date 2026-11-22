"""Steering a plume: extraction-rate control of obstructed transport.

A fixed Gaussian release near the inlet feeds a contaminant plume that
a frozen Navier-Stokes flow advects through the gaps of three staggered
blocks toward a protected zone above the uppermost block. K Gaussian
extraction devices remove contaminant at rate
:math:`r(x)\\,u = \\sum_k p_k q_k(x)\\, u` with nonnegative rates
``p >= 0`` — removal proportional to the local concentration preserves
the maximum principle (a constant-rate sink keeps withdrawing once
u ~ 0 and manufactures negative mass). The objective trades
time-integrated zone contamination against a quadratic actuation cost:

    J(p) = int_0^T int w(x) u(x, t; p)^2 dx dt + (alpha/2) ||p||^2.

The extraction term is bilinear in (p, u): control authority saturates
where the plume is thin, dR/dp is state-dependent, and genuine mixed
state-parameter curvature enters the Hessian-vector products.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from pyapprox.implicitfunction.implicit_map import ImplicitMapFunction
    from pyapprox.ode.stepping_methods import SteppingMethodProtocol
    from pyapprox.pde.boundary import BoundaryConditionRole
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.zone_weights import ZoneWeightProtocol
    from pyapprox.pde.models.galerkin.transient_system import (
        GalerkinTransientSystem,
    )
    from pyapprox.util.linalg.scipy_stack import CSRFormat

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd

# Bound to NumPy: the Stokes and transport solves assemble with skfem.
_Np = NDArray[Any]

# Extraction-device layout: label -> center. Footprints sit in the gap
# channels and along the flow path; two deliberately low-leverage
# placements (off the feed path / downstream of the zone) make the
# learned strategy legible.
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


class _InletNormalSpeed:
    """Inlet normal speed |v.n|(y) for the Danckwerts coefficient.

    A picklable class (stored BC callables must pickle) evaluating the
    x-component of the parabolic inlet profile, which equals the
    inflow speed on the left boundary where n = (-1, 0).
    """

    def __init__(self, a: float, b: float) -> None:
        self._a = float(a)
        self._b = float(b)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x[1]
        return np.asarray(y ** (self._a - 1.0) * (1.0 - y) ** (self._b - 1.0))


@runtime_checkable
class InletConditionProtocol(Protocol):
    """The transport condition on the inlet (the left boundary)."""

    def boundary_condition(
        self,
        basis: "LagrangeBasis[_Np]",
        vel_shape_params: Tuple[float, float],
        bkd: NumpyBkd,
    ) -> "BoundaryConditionRole[_Np]":
        """The condition on ``basis``'s left boundary, for the parabolic
        inlet profile of shape ``vel_shape_params``."""
        ...


class DanckwertsInlet:
    """The total-flux inflow condition kappa*grad(u).n = (v.n) u: a Robin
    condition whose coefficient is the parabolic inlet speed, so mass
    enters and leaves the inlet only with the flow, closing the mass
    ledger."""

    def boundary_condition(
        self,
        basis: "LagrangeBasis[_Np]",
        vel_shape_params: Tuple[float, float],
        bkd: NumpyBkd,
    ) -> "BoundaryConditionRole[_Np]":
        from pyapprox.pde.galerkin.boundary.implementations import RobinBC

        return RobinBC(basis, "left", _InletNormalSpeed(*vel_shape_params), 0.0, bkd)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class DirichletInlet:
    """u = 0 on the inlet: simpler, but an absorbing wall, so a release
    placed diffusively close to the inlet loses roughly half its mass
    upstream through it."""

    def boundary_condition(
        self,
        basis: "LagrangeBasis[_Np]",
        vel_shape_params: Tuple[float, float],
        bkd: NumpyBkd,
    ) -> "BoundaryConditionRole[_Np]":
        from pyapprox.pde.galerkin.boundary.implementations import DirichletBC

        return DirichletBC(basis, "left", 0.0, bkd)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ObstructedFlowControlProblem:
    """Extraction-rate control of transport through an obstructed
    channel.

    The controls are the rates of extraction devices, which enter the
    LINEAR REACTION coefficient: each removes contaminant in proportion
    to the local concentration. The forcing slot carries the fixed
    Gaussian release and is not parameterized.

    Composes the obstructed-flow substrate
    (:mod:`pyapprox.pde.zoo.obstructed_flow`) with the transient
    galerkin adjoint stack: the returned model satisfies
    ``ObjectiveProtocol`` with an adjoint gradient and a
    second-order-adjoint HVP, ready for
    ``ScipyTrustConstrOptimizer(...).bind(problem.model(),
    problem.bounds())``.

    Parameters
    ----------
    bkd : NumpyBkd
        Computational backend. The forward solve uses skfem, which is
        NumPy-only.
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
        contamination and extraction effort).
    release_center, release_width, release_amplitude : floats
        Fixed Gaussian release upstream of the blocks.
    actuator_centers : dict, optional
        ``label -> (x, y)`` extraction-device layout; defaults to
        gap-channel placements.
    actuator_width : float
        Gaussian extraction-footprint width.
    control_bound : float
        Extraction-rate bounds ``0 <= p_k <= control_bound``: removal
        only, preserving the maximum principle.
    zone_weight : ZoneWeightProtocol, optional
        Spatial weighting of the protected zone. Defaults to the
        element-aligned rectangle above the uppermost block.
    inlet : InletConditionProtocol, optional
        The inflow condition. Default :class:`DanckwertsInlet`, which
        closes the mass ledger; :class:`DirichletInlet` absorbs.
    method : SteppingMethodProtocol, optional
        The time-stepping method. Default Crank--Nicolson with Newton
        (absolute tolerance 1e-12, at most 5 iterations).

    Raises
    ------
    TypeError
        If ``bkd`` is not a ``NumpyBkd``, or ``inlet`` or ``method`` does
        not satisfy its protocol.
    """

    def __init__(
        self,
        bkd: NumpyBkd,
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
        zone_weight: Optional["ZoneWeightProtocol[_Np]"] = None,
        inlet: Optional[InletConditionProtocol] = None,
        method: Optional["SteppingMethodProtocol[_Np, _Np]"] = None,
    ) -> None:
        from pyapprox.ode.stepping_methods import (
            RungeKutta,
            SteppingMethodProtocol,
        )
        from pyapprox.ode.tableau import crank_nicolson
        from pyapprox.util.rootfinding.nonlinear import Newton

        self._bkd = bkd
        if not isinstance(bkd, NumpyBkd):
            raise TypeError(
                "ObstructedFlowControlProblem requires NumpyBkd because "
                "its forward solve uses skfem, which is NumPy-only. Got "
                f"{type(bkd).__name__}."
            )
        if inlet is None:
            inlet = DanckwertsInlet()
        if not isinstance(inlet, InletConditionProtocol):
            raise TypeError(
                f"inlet must satisfy InletConditionProtocol, got {type(inlet).__name__}"
            )
        if method is None:
            method = RungeKutta(
                crank_nicolson(), Newton(atol=1e-12, rtol=0.0, maxiters=5)
            )
        if not isinstance(method, SteppingMethodProtocol):
            raise TypeError(
                "method must satisfy SteppingMethodProtocol, got "
                f"{type(method).__name__}"
            )
        self._inlet = inlet
        self._method = method
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
                -((coords[0] - center[0]) ** 2 + (coords[1] - center[1]) ** 2)
                / (2.0 * width**2)
            )
        )

    def _actuator_fields_nodal(self) -> np.ndarray:
        """Evaluate every unit actuator bump at the nodal coordinates.

        Returns
        -------
        np.ndarray
            One field per device. Shape: (ncontrols, ndofs).
        """
        squared_dist = (
            (self._dof_coords[:, None, :] - self._actuator_centers[:, :, None]) ** 2
        ).sum(axis=0)
        return np.asarray(np.exp(-squared_dist / (2.0 * self._actuator_width**2)))

    def _build_model(
        self,
        nstokes_refine: int,
        ntransport_refine: int,
        reynolds_num: float,
        vel_shape_params: Tuple[float, float],
        diffusivity: float,
        final_time: float,
        deltat: float,
        zone_weight: Optional["ZoneWeightProtocol[_Np]"],
    ) -> None:
        from pyapprox.implicitfunction.implicit_map import (
            ArrayParameters,
            ImplicitMapFunction,
        )
        from pyapprox.implicitfunction.setups import RecomputeAlways
        from pyapprox.implicitfunction.tikhonov_term import TikhonovTerm
        from pyapprox.ode.step_plan import FixedStep
        from pyapprox.ode.trajectory_terms import TrajectoryFunctional
        from pyapprox.ode.transient_implicit_map import (
            FixedTrajectoryFunctional,
            ParameterFanOut,
            SummedFunctional,
            TransientImplicitMap,
        )
        from pyapprox.pde.constitutive.coefficient_functions import (
            NodalFieldDiffusion,
            NodalFieldForcing,
            NodalFieldLinearReaction,
        )
        from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
        from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
        from pyapprox.pde.galerkin.compose import compose_galerkin_system
        from pyapprox.pde.galerkin.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
            skfem_discretization,
        )
        from pyapprox.pde.galerkin.zone_weights import (
            ElementAlignedRectangleZone,
        )
        from pyapprox.pde.models.galerkin.transient_observations import (
            time_integrated_weighted_l2,
        )
        from pyapprox.pde.models.galerkin.transient_system import (
            galerkin_transient_system,
        )
        from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
            AdvectionDiffusionParameterization,
        )
        from pyapprox.pde.parameterizations.parameter_state import ParameterState
        from pyapprox.pde.zoo.obstructed_flow import (
            build_obstructed_mesh,
            extract_velocity_callable,
            solve_obstructed_stokes,
        )
        from pyapprox.util.linalg.scipy_stack import ScipySparseLU

        bkd = self._bkd

        # Frozen flow: solved once on its own (finer) mesh.
        stokes_mesh = build_obstructed_mesh(bkd, nstokes_refine)
        stokes_sol, stokes, vel_basis, pres_basis = solve_obstructed_stokes(
            stokes_mesh, bkd, reynolds_num, list(vel_shape_params)
        )

        transport_mesh = build_obstructed_mesh(bkd, ntransport_refine)
        self._basis: "LagrangeBasis[_Np]" = LagrangeBasis(transport_mesh, degree=1)
        self._velocity = extract_velocity_callable(
            stokes_sol, stokes, vel_basis, pres_basis, self._basis, bkd
        )

        coords = bkd.to_numpy(self._basis.dof_coordinates())
        release_nodal = self._release_amplitude * self._gaussian_nodal(
            coords, self._release_center, self._release_width
        )
        self._release_nodal = release_nodal
        self._dof_coords = coords
        # The physics' linear reaction enters as du/dt = ... + r*u, so
        # a nonnegative extraction RATE p_k maps to r = -sum_k p_k q_k:
        # the map's basis fields carry the sign.
        actuator_fields = self._actuator_fields_nodal()
        extraction_map = BasisExpansion(
            bkd,
            0.0,
            [bkd.asarray(-actuator_fields[kk]) for kk in range(self.ncontrols())],
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
            reaction=NodalFieldLinearReaction(
                self._basis, dofs=np.zeros(self._basis.ndofs())
            ),
        )
        # Transport BCs at the inflow, by default Danckwerts total-flux
        # kappa*grad(u).n = (v.n)(u - u_in) with u_in = 0, mapped
        # to the Robin convention -D du/dn = alpha u - g as
        # alpha(y) = |v.n|(y) (the parabolic inlet speed), g = 0 —
        # mass exchanges with the inlet only advectively, closing
        # the ledger (measured: the Dirichlet variant loses ~half
        # the release by upstream diffusion into u = 0). Everything
        # else is natural by omission: with the NON-conservative
        # advection form this is exactly du/dn = 0 — free advective
        # outflow on the right, no-flux walls elsewhere (v.n = 0
        # there). Outflow must be neither Dirichlet (artificial
        # boundary layer reflecting the plume) nor zero-total-flux
        # (traps all contaminant).
        boundary_conditions: List[BoundaryConditionRole[_Np]] = [
            self._inlet.boundary_condition(self._basis, vel_shape_params, bkd)
        ]
        parameterization = AdvectionDiffusionParameterization(
            physics, reaction_map=extraction_map, bkd=bkd
        )

        if zone_weight is None:
            zone_weight = ElementAlignedRectangleZone(_ZONE_XLIM, _ZONE_YLIM)
        self._zone_weight: "ZoneWeightProtocol[_Np]" = zone_weight
        zone_mass = zone_weight.assemble_weighted_mass(self._basis, bkd)

        self._transient: "GalerkinTransientSystem[_Np, CSRFormat]" = (
            galerkin_transient_system(
                skfem_discretization(
                    compose_galerkin_system(physics, boundary_conditions)
                ),
                ScipySparseLU(),
                ParameterState(parameterization),
            )
        )
        self._steps = FixedStep(deltat, (0.0, final_time))
        self._init_state = bkd.zeros((physics.nstates(), 1))

        # J = int_0^T u^T W u dt, by the method the plume is solved with,
        # plus the actuation cost (alpha/2)||p||^2, which reads the
        # controls each solve is set to.
        tikhonov = TikhonovTerm(bkd, self._alpha)
        functional: "SummedFunctional[_Np, _Np]" = SummedFunctional(
            [
                time_integrated_weighted_l2(
                    self._transient,
                    self._method,
                    RecomputeAlways(self._transient.solved.solver),
                    zone_mass,
                ),
                FixedTrajectoryFunctional(
                    TrajectoryFunctional.from_terms(bkd, 1, parameter_terms=(tikhonov,))
                ),
            ]
        )
        implicit = TransientImplicitMap(
            self._transient,
            ParameterFanOut(self._transient, [tikhonov.set_param]),
            self._method,
            RecomputeAlways(self._transient.solved.solver),
            self._steps,
            self._init_state,
            functional,
        )
        self._model: "ImplicitMapFunction[Any, _Np, _Np]" = ImplicitMapFunction(
            implicit, ArrayParameters(bkd)
        )

    def bkd(self) -> NumpyBkd:
        """Return the backend."""
        return self._bkd

    def ncontrols(self) -> int:
        """Return the number of actuator amplitudes K."""
        return len(self._actuator_labels)

    def model(self) -> "ImplicitMapFunction[Any, _Np, _Np]":
        """Return the control objective J(p).

        Satisfies ``ObjectiveProtocol``: adjoint gradient and
        second-order-adjoint HVP via ``derivatives()``.
        """
        return self._model

    def trajectory(self, controls: _Np) -> Tuple[_Np, _Np]:
        """Solve the transport at ``controls`` and return every state.

        Parameters
        ----------
        controls : np.ndarray
            Extraction rates. Shape: (ncontrols, 1).

        Returns
        -------
        states : np.ndarray
            The concentration at each time. Shape: (nstates, ntimes).
        times : np.ndarray
            The times. Shape: (ntimes,).
        """
        from pyapprox.implicitfunction.setups import RecomputeAlways
        from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
        from pyapprox.pde.models.galerkin.transient_system import (
            solve_galerkin_transient,
        )

        bkd = self._bkd
        self._transient.set_param(controls)
        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        solve_galerkin_transient(
            self._transient,
            self._method,
            RecomputeAlways(self._transient.solved.solver),
            self._steps,
            self._init_state,
            store,
        )
        states = bkd.concatenate(
            [store.state(n) for n in range(store.ntimes())], axis=1
        )
        return states, bkd.asarray(store.times())

    def bounds(self) -> _Np:
        """Return extraction-rate bounds. Shape: ``(ncontrols, 2)``.

        The lower bound 0 is structural: extraction removes only, so
        the maximum principle is preserved, and low-leverage devices
        pin at zero rather than drifting signed.
        """
        return self._bkd.asarray(
            np.tile([0.0, self._control_bound], (self.ncontrols(), 1))
        )

    def inlet(self) -> InletConditionProtocol:
        """Return the inflow condition."""
        return self._inlet

    def release_center(self) -> Tuple[float, float]:
        """Return the fixed release center (for plotting)."""
        return self._release_center

    def release_field(self) -> np.ndarray:
        """Return the nodal release rate field s(x). Shape: (ndofs,)."""
        return self._release_nodal

    def extraction_field(self, controls: np.ndarray) -> np.ndarray:
        """Return the nodal extraction rate field sum_k p_k q_k(x).

        Parameters
        ----------
        controls : np.ndarray
            Extraction rates. Shape: (ncontrols,).

        Returns
        -------
        np.ndarray
            Nonnegative rate field. Shape: (ndofs,).
        """
        return np.asarray(
            np.asarray(controls, dtype=float) @ self._actuator_fields_nodal()
        )

    def actuator_labels(self) -> Tuple[str, ...]:
        """Return the actuator labels in parameter order."""
        return self._actuator_labels

    def actuator_centers(self) -> np.ndarray:
        """Return actuator centers. Shape: ``(2, ncontrols)``."""
        return self._actuator_centers

    def zone_weight(self) -> "ZoneWeightProtocol[_Np]":
        """Return the protected-zone weighting."""
        return self._zone_weight

    def basis(self) -> "LagrangeBasis[_Np]":
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
