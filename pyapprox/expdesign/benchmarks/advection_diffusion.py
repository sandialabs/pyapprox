"""Obstructed advection-diffusion OED benchmark.

Couples Stokes flow with advection-diffusion-reaction on an obstructed
domain. The Stokes velocity field drives advection, and a KLE-based
log-normal random field provides forcing. Parameters control the KLE
coefficients, inlet velocity shape, and Reynolds number.
"""

# TODO: Should benchmarks be defined here on in benchmarks module. Decide and document rule, and place in benchmarks.CONVENTIONS.md

from typing import Generic, Tuple

import numpy as np

from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def _build_obstructed_mesh(bkd, nrefine):
    """Create the obstructed domain mesh."""
    from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D

    xintervals = np.array([0, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 1.0])
    yintervals = np.linspace(0, 1, 5)
    obstruction_indices = np.array([3, 6, 13], dtype=int)
    return ObstructedMesh2D(
        xintervals,
        yintervals,
        obstruction_indices,
        bkd,
        nrefine=nrefine,
    )


def _solve_stokes(mesh, bkd, reynolds_num, vel_shape_params):
    """Solve Stokes on the obstructed mesh for velocity field."""
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
    from pyapprox.pde.galerkin.time_integration.galerkin_model import (
        GalerkinModel,
    )

    vel_basis = VectorLagrangeBasis(mesh, degree=2)
    pres_basis = LagrangeBasis(mesh, degree=1)

    a, b = vel_shape_params

    def inlet_func(x, time=None):
        """Parabolic-like inlet: y^(a-1) * (1-y)^(b-1)."""
        y = x[1]
        npts = x.shape[1]
        vals = np.zeros((npts, 2))
        vals[:, 0] = y ** (a - 1) * (1 - y) ** (b - 1)
        return vals

    def zero_vel(x, time=None):
        return np.zeros((x.shape[1], 2))

    # No-slip on obs, bottom, top; parabolic inlet on left
    vel_bcs = [
        ("left", inlet_func),
        ("bottom", zero_vel),
        ("top", zero_vel),
        ("obs0", zero_vel),
        ("obs1", zero_vel),
        ("obs2", zero_vel),
    ]

    viscosity = 1.0 / reynolds_num

    stokes = StokesPhysics(
        vel_basis,
        pres_basis,
        bkd,
        navier_stokes=True,
        viscosity=viscosity,
        vel_dirichlet_bcs=vel_bcs,
    )

    model = GalerkinModel(stokes, bkd)
    init_guess = stokes.init_guess(0.0)
    sol = model.solve_steady(init_guess, tol=1e-10, maxiter=50)

    return sol, stokes, vel_basis, pres_basis


def _extract_velocity_callable(
    sol,
    stokes,
    vel_basis,
    pres_basis,
    adr_basis,
    bkd,
):
    """Extract velocity from Stokes solution as callable for ADR.

    Returns a callable that accepts points of shape (2, ...) and returns
    velocity of shape (2, ...), preserving any extra dimensions (e.g.,
    quadrature point structure from skfem).
    """
    vel_ndofs = stokes.vel_ndofs()
    vel_state_np = bkd.to_numpy(sol[:vel_ndofs])

    # skfem ElementVector DOFs are block-ordered:
    # [comp0_dof0, comp0_dof1,..., comp1_dof0,...]
    nscalar = vel_ndofs // 2
    vel_x = vel_state_np[:nscalar]
    vel_y = vel_state_np[nscalar:]

    # Get skfem basis for velocity components (scalar P2)
    vel_skfem = vel_basis.skfem_basis()
    from skfem import Basis

    scalar_elem = vel_skfem.elem.elem
    scalar_vel_basis = Basis(
        vel_basis.mesh().skfem_mesh(),
        scalar_elem,
        intorder=4,
    )

    # Interpolate velocity components onto ADR mesh nodes
    adr_skfem = adr_basis.skfem_basis()
    vel_x_interp = adr_skfem.interpolator(
        scalar_vel_basis.interpolator(vel_x)(adr_skfem.doflocs)
    )
    vel_y_interp = adr_skfem.interpolator(
        scalar_vel_basis.interpolator(vel_y)(adr_skfem.doflocs)
    )

    # Return callable that evaluates velocity at arbitrary points.
    # Must preserve input shape: (2, ...) -> (2, ...)
    # skfem passes w.x with shape (2, nqpts, nelements)
    def velocity_field(x):
        orig_shape = x.shape[1:]
        x_flat = x.reshape(2, -1)
        vx = vel_x_interp(x_flat).reshape(orig_shape)
        vy = vel_y_interp(x_flat).reshape(orig_shape)
        return np.stack([vx, vy], axis=0)

    return velocity_field


def _create_kle_forcing(adr_basis, nkle_terms, bkd):
    """Create KLE forcing on quadrature points."""
    from pyapprox.pde.field_maps.kle_factory import (
        create_lognormal_kle_field_map,
    )

    # Use mesh nodes for KLE construction
    mesh_coords_np = bkd.to_numpy(adr_basis.dof_coordinates())
    mesh_coords = bkd.asarray(mesh_coords_np)
    mean_log = bkd.zeros(mesh_coords.shape[1])

    kle_map = create_lognormal_kle_field_map(
        mesh_coords=mesh_coords,
        mean_log_field=mean_log,
        bkd=bkd,
        correlation_length=0.3,
        num_kle_terms=nkle_terms,
        sigma=0.3,
    )
    return kle_map


class ObstructedAdvectionDiffusionOEDBenchmark(Generic[Array]):
    """Obstructed advection-diffusion OED benchmark.

    Couples Stokes flow with transient advection-diffusion on an
    obstructed domain [0,1]^2 with three rectangular obstacles.

    Parameters (13 total):
        - KLE coefficients (10) for log-normal forcing field
        - Inlet velocity shape parameters (2): U[2, 3]
        - Reynolds number (1): U[5, 20]

    The observation model evaluates concentration at sensor locations
    at the final time. The prediction model integrates concentration
    over a target subdomain (x >= 5/7) at the final time.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nstokes_refine : int
        Number of mesh refinements for Stokes solve.
    nadvec_diff_refine : int
        Number of mesh refinements for ADR solve.
    nkle_terms : int
        Number of KLE terms for forcing field.
    nsensors : int
        Number of sensor locations.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nstokes_refine: int = 3,
        nadvec_diff_refine: int = 3,
        nkle_terms: int = 10,
        nsensors: int = 20,
    ) -> None:
        self._bkd = bkd
        self._nkle_terms = nkle_terms
        self._nsensors = nsensors
        self._nstokes_refine = nstokes_refine
        self._nadvec_diff_refine = nadvec_diff_refine

        # Build prior: [kle_params (10), vel_shape (2), reynolds (1)]
        marginals: list[Any] = []
        for _ in range(nkle_terms):
            marginals.append(GaussianMarginal(0.0, 1.0, bkd))
        marginals.append(UniformMarginal(2.0, 3.0, bkd))
        marginals.append(UniformMarginal(2.0, 3.0, bkd))
        marginals.append(UniformMarginal(5.0, 20.0, bkd))
        self._prior = IndependentJoint(marginals, bkd)
        self._nparams = nkle_terms + 3

        # Build mesh and ADR basis (shared across evaluations)
        self._stokes_mesh = _build_obstructed_mesh(bkd, nstokes_refine)
        self._adr_mesh = _build_obstructed_mesh(bkd, nadvec_diff_refine)

        from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis

        self._adr_basis = LagrangeBasis(self._adr_mesh, degree=1)

        # Create KLE forcing map
        self._kle_map = _create_kle_forcing(
            self._adr_basis,
            nkle_terms,
            bkd,
        )

        # Set up sensor locations (regular grid in non-obstructed region)
        rng = np.random.default_rng(42)
        nodes_np = bkd.to_numpy(self._adr_mesh.nodes())
        indices = rng.choice(nodes_np.shape[1], nsensors, replace=False)
        self._sensor_indices = indices
        self._sensor_locations = bkd.asarray(nodes_np[:, indices])

        # Target subdomain: x >= 5/7
        self._target_threshold = 5.0 / 7.0

        # Time integration config
        self._deltat = 0.25
        self._final_time = 1.5

        # Create FunctionProtocol wrappers
        nqoi_obs = nsensors
        nqoi_pred = 1

        def obs_callable(samples):
            return self._evaluate_observation(samples)

        def pred_callable(samples):
            return self._evaluate_prediction(samples)

        self._obs_model = FunctionFromCallable(
            nqoi_obs,
            self._nparams,
            obs_callable,
            bkd,
        )
        self._pred_model = FunctionFromCallable(
            nqoi_pred,
            self._nparams,
            pred_callable,
            bkd,
        )

    def _solve_forward(self, params_np):
        """Full forward solve for a single parameter vector."""
        from pyapprox.pde.galerkin.boundary.implementations import (
            RobinBC,
        )
        from pyapprox.pde.galerkin.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.pde.galerkin.time_integration.galerkin_model import (
            GalerkinModel,
        )
        from pyapprox.pde.time.config import TimeIntegrationConfig

        bkd = self._bkd

        # Extract parameters
        kle_params = params_np[: self._nkle_terms]
        vel_shape_a = float(params_np[self._nkle_terms])
        vel_shape_b = float(params_np[self._nkle_terms + 1])
        reynolds_num = float(params_np[self._nkle_terms + 2])

        # Solve Stokes for velocity
        sol, stokes, vel_basis, pres_basis = _solve_stokes(
            self._stokes_mesh,
            bkd,
            reynolds_num,
            [vel_shape_a, vel_shape_b],
        )

        # Extract velocity callable
        vel_callable = _extract_velocity_callable(
            sol,
            stokes,
            vel_basis,
            pres_basis,
            self._adr_basis,
            bkd,
        )

        # Evaluate KLE forcing at KLE params (1D input for field map)
        kle_coeffs = bkd.asarray(kle_params.ravel())
        forcing_vals = self._kle_map(kle_coeffs)
        forcing_nodal = bkd.to_numpy(forcing_vals)

        # Create forcing callable from nodal values
        adr_skfem = self._adr_basis.skfem_basis()
        forcing_interp = adr_skfem.interpolator(forcing_nodal)

        def forcing_func(x, time=None):
            return forcing_interp(x)

        # Robin BCs on left/right (alpha=0.1, g=0)
        alpha = 0.1
        robin_bcs = [
            RobinBC(self._adr_basis, "left", alpha, 0.0, bkd),
            RobinBC(self._adr_basis, "right", alpha, 0.0, bkd),
        ]

        # Create ADR physics
        adr = AdvectionDiffusionReaction(
            basis=self._adr_basis,
            diffusivity=0.1,
            bkd=bkd,
            velocity=vel_callable,
            forcing=forcing_func,
            boundary_conditions=robin_bcs,
        )

        # Solve transient
        model = GalerkinModel(adr, bkd)
        ic = bkd.to_numpy(bkd.zeros(adr.nstates()))
        config = TimeIntegrationConfig(
            method="backward_euler",
            final_time=self._final_time,
            deltat=self._deltat,
            newton_tol=1e-8,
            newton_maxiter=5,
        )
        solutions, times = model.solve_transient(
            bkd.asarray(ic),
            config,
        )

        return solutions, times

    def _extract_obs(self, solutions):
        """Extract observations from a forward solve solution."""
        final_sol = self._bkd.to_numpy(solutions[:, -1])
        return final_sol[self._sensor_indices]

    def _extract_pred(self, solutions):
        """Extract predictions from a forward solve solution."""
        nodes_np = self._bkd.to_numpy(self._adr_mesh.nodes())
        target_mask = nodes_np[0, :] >= self._target_threshold
        final_sol = self._bkd.to_numpy(solutions[:, -1])
        return np.array([np.mean(final_sol[target_mask])])

    def _evaluate_observation(self, samples):
        """Evaluate observation model. samples: (nparams, nsamples)."""
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = []

        for ii in range(nsamples):
            solutions, times = self._solve_forward(samples_np[:, ii])
            results.append(self._extract_obs(solutions))

        return bkd.asarray(np.column_stack(results))

    def _evaluate_prediction(self, samples):
        """Evaluate prediction model. samples: (nparams, nsamples)."""
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = []

        for ii in range(nsamples):
            solutions, times = self._solve_forward(samples_np[:, ii])
            results.append(self._extract_pred(solutions))

        return bkd.asarray(np.column_stack(results))

    def evaluate_both(self, samples: Array) -> Tuple[Array, Array]:
        """Evaluate observation and prediction models with a single solve.

        Solves the forward model once per sample and extracts both
        observation and prediction quantities from the same solution.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nparams, nsamples)

        Returns
        -------
        obs_values : Array
            Observation values. Shape: (nobs, nsamples)
        pred_values : Array
            Prediction values. Shape: (npred, nsamples)
        """
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        obs_results = []
        pred_results = []

        for ii in range(nsamples):
            solutions, times = self._solve_forward(samples_np[:, ii])
            obs_results.append(self._extract_obs(solutions))
            pred_results.append(self._extract_pred(solutions))

        return (
            bkd.asarray(np.column_stack(obs_results)),
            bkd.asarray(np.column_stack(pred_results)),
        )

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def prior(self) -> IndependentJoint[Array]:
        """Return the prior distribution."""
        return self._prior

    def observation_model(self):
        """Return the observation model (FunctionProtocol)."""
        return self._obs_model

    def prediction_model(self):
        """Return the prediction model (FunctionProtocol)."""
        return self._pred_model

    def observation_locations(self) -> Array:
        """Return sensor locations. Shape: (2, nsensors)."""
        return self._sensor_locations

    def nobservations(self) -> int:
        """Return number of observation sensors."""
        return self._nsensors

    def nparams(self) -> int:
        """Return number of parameters."""
        return self._nparams


@BenchmarkRegistry.register(
    "obstructed_advection_diffusion_oed",
    category="oed",
    description="Obstructed advection-diffusion OED benchmark with Stokes coupling",
)
def _obstructed_advection_diffusion_oed_factory(
    bkd: Backend[Array],
) -> ObstructedAdvectionDiffusionOEDBenchmark:
    return ObstructedAdvectionDiffusionOEDBenchmark(bkd)
