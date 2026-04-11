"""Advection-diffusion OED problem classes.

Two PDE-specific prediction OED problems:

- :class:`AdvectionDiffusionOEDProblem` — full 13-dim parameter
  space (10 KLE terms + 2 inlet velocity shape + 1 Reynolds).
- :class:`FixedVelocityAdvectionDiffusionOEDProblem` — pins the
  velocity triple at construction, caches the Stokes solve, and
  reduces to a pure KLE prior.

Both subclass :class:`PredictionOEDProblem` and drive a two-step
template (:meth:`_build_pde_substrate` → :meth:`_wire_inference_problem`)
so subclass variants override exactly the step that changes.

Mesh, Stokes, and KLE helper functions live in the sibling private
modules ``_mesh``, ``_stokes``, ``_kle``.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

if TYPE_CHECKING:
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
    from pyapprox.pde.galerkin.protocols.boundary import (
        BoundaryConditionProtocol,
    )

import numpy as np
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.problems.inverse import BayesianInferenceProblem
from pyapprox_benchmarks.problems.oed.advection_diffusion._kle import (
    _create_subdomain_kle_forcing,
)
from pyapprox_benchmarks.problems.oed.advection_diffusion._mesh import (
    _build_obstructed_mesh,
)
from pyapprox_benchmarks.problems.oed.advection_diffusion._stokes import (
    _extract_velocity_callable,
    _solve_stokes,
)
from pyapprox_benchmarks.problems.oed.prediction_problem import (
    PredictionOEDProblem,
)

# ---------------------------------------------------------------------------
# AdvectionDiffusionOEDProblem
# ---------------------------------------------------------------------------


class AdvectionDiffusionOEDProblem(
    PredictionOEDProblem[Array], Generic[Array],
):
    """Prediction OED problem for obstructed advection-diffusion.

    Couples Stokes flow with transient advection-diffusion on an
    obstructed ``[0, 1]^2`` domain with three rectangular obstacles.
    The KLE log-normal random field drives the ADR equation, and the
    Stokes velocity field is parameterised by inlet shape and Reynolds
    number.

    This class is a :class:`PredictionOEDProblem` subclass, so callers
    get ``prior()``, ``obs_map()``, ``qoi_map()``,
    ``design_conditions()``, ``nobs()``, ``nparams()``, ``npred()``,
    ``noise_variances()``, ``weight_bounds()``, and ``bkd()`` for
    free. It additionally exposes PDE-specific methods
    (``evaluate_nodal``, ``evaluate_both``, ``solve_for_plotting``)
    and mesh accessors (``mesh_nodes``, ``nnodes``).

    .. note::

       **Backend restriction.** This problem uses skfem for the
       Stokes and ADR solves, and skfem is NumPy-only. The
       constructor rejects any backend other than :class:`NumpyBkd`.
       Users that need torch tensors at the call site should convert
       at the boundary; gradients will not propagate through the
       forward solve.

    Parameters (13 total):
        - KLE coefficients (10) for log-normal forcing field
        - Inlet velocity shape parameters (2): U[2, 3]
        - Reynolds number (1): U[5, 20]

    Subclass extension points
    -------------------------
    ``__init__`` drives two template methods in order:

    1. :meth:`_build_pde_substrate` — builds mesh, KLE map, sensor
       locations, prior, observation/prediction bound-method models,
       and the :class:`BayesianInferenceProblem`.
    2. :meth:`_wire_inference_problem` — calls
       :meth:`PredictionOEDProblem.__init__` with the substrate built
       in step 1.

    Subclasses override exactly the step that changes (typically
    ``_build_pde_substrate``) and never re-call ``__init__``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        *,
        noise_std: float = 0.1,
        nstokes_refine: int = 3,
        nadvec_diff_refine: int = 3,
        nkle_terms: int = 10,
        nsensors: int = 20,
        diffusivity: float = 0.1,
        final_time: float = 1.5,
        deltat: float = 0.25,
        kle_subdomain: Optional[Tuple[float, float, float, float]] = (
            0.0,
            0.25,
            0.0,
            1.0,
        ),
        kle_correlation_length: float = 0.1,
        kle_sigma: float = 0.3,
        source_mode: Literal["forcing", "initial_condition"] = "forcing",
    ) -> None:
        if source_mode not in ("forcing", "initial_condition"):
            raise ValueError(
                f"source_mode must be 'forcing' or 'initial_condition', "
                f"got {source_mode!r}"
            )
        from pyapprox.util.backends.numpy import NumpyBkd

        if not isinstance(bkd, NumpyBkd):
            raise TypeError(
                "AdvectionDiffusionOEDProblem requires NumpyBkd because "
                "its forward solve uses skfem, which is NumPy-only. Got "
                f"{type(bkd).__name__}. If you need torch tensors at "
                "the call site, construct with NumpyBkd() and convert "
                "at the boundary. Gradients will not propagate through "
                "the forward solve."
            )

        # mypy narrows ``bkd`` to ``NumpyBkd`` (i.e. ``Backend[ndarray]``)
        # after the ``isinstance`` check, which is incompatible with the
        # class's generic ``Backend[Array]``. The runtime check above
        # already guarantees NumpyBkd, so re-widen with ``cast``.
        self._bkd: Backend[Array] = cast("Backend[Array]", bkd)
        self._noise_std = float(noise_std)
        self._nstokes_refine = nstokes_refine
        self._nadvec_diff_refine = nadvec_diff_refine
        self._nkle_terms = nkle_terms
        self._nsensors = nsensors
        self._diffusivity = diffusivity
        self._final_time = final_time
        self._deltat = deltat
        self._kle_subdomain = kle_subdomain
        self._kle_correlation_length = kle_correlation_length
        self._kle_sigma = kle_sigma
        self._source_mode = source_mode

        # Step 1: build mesh, KLE, sensors, prior, obs/pred, inference.
        self._build_pde_substrate()

        # Step 2: wire the PredictionOEDProblem base class.
        self._wire_inference_problem()

    # ---- template methods ------------------------------------------------

    def _build_pde_substrate(self) -> None:
        """Build mesh, KLE, sensors, prior, obs/pred, inference problem.

        Default implementation builds the full 13-dim prior (10 KLE
        terms + 3 uniform velocity marginals) and wires bound-method
        obs/pred callables. Override to install a different prior or
        pre-cache forward-solve state.

        Populates the following private attributes (every one is part
        of the consumer-impact inventory and must remain reachable as
        ``problem._foo`` for existing ``pyapprox-papers`` code to
        work)::

            self._adr_mesh, self._adr_basis, self._kle_map,
            self._stokes_mesh, self._sensor_indices,
            self._sensor_locations, self._target_threshold,
            self._nparams, self._nkle_terms, self._nsensors,
            self._prior, self._obs_model, self._pred_model,
            self._inference_problem

        ``_target_threshold`` is currently a hard-coded constant
        (``5.0 / 7.0``); any change to make it configurable is out
        of scope for this migration.
        """
        bkd = self._bkd

        # Build prior: [kle_params (10), vel_shape (2), reynolds (1)]
        marginals: List[
            Union[GaussianMarginal[Array], UniformMarginal[Array]]
        ] = []
        for _ in range(self._nkle_terms):
            marginals.append(GaussianMarginal(0.0, 1.0, bkd))
        marginals.append(UniformMarginal(2.0, 3.0, bkd))
        marginals.append(UniformMarginal(2.0, 3.0, bkd))
        marginals.append(UniformMarginal(5.0, 20.0, bkd))
        self._prior = IndependentJoint(marginals, bkd)
        self._nparams = self._nkle_terms + 3

        # Build mesh and ADR basis (shared across evaluations). The
        # ADR mesh is built with the KLE subdomain boundaries
        # inserted as grid lines so no element straddles the
        # subdomain edge.
        self._stokes_mesh = _build_obstructed_mesh(bkd, self._nstokes_refine)
        self._adr_mesh = _build_obstructed_mesh(
            bkd,
            self._nadvec_diff_refine,
            kle_subdomain=self._kle_subdomain,
        )

        from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis

        self._adr_basis = LagrangeBasis(self._adr_mesh, degree=1)

        self._kle_map = _create_subdomain_kle_forcing(
            self._adr_basis,
            self._nkle_terms,
            self._kle_correlation_length,
            self._kle_sigma,
            self._kle_subdomain,
            bkd,
        )

        # Set up sensor locations (regular grid in non-obstructed region)
        rng = np.random.default_rng(42)
        nodes_np = bkd.to_numpy(self._adr_mesh.nodes())
        indices = rng.choice(
            nodes_np.shape[1], self._nsensors, replace=False,
        )
        self._sensor_indices = indices
        self._sensor_locations = bkd.asarray(nodes_np[:, indices])

        # Target subdomain: x >= 5/7 (hard-coded)
        self._target_threshold = 5.0 / 7.0

        # Create FunctionProtocol wrappers. Using bound methods here
        # (rather than local closures) is what makes the inference
        # problem and the owning problem picklable.
        nqoi_obs = self._nsensors
        nqoi_pred = 1
        self._obs_model = FunctionFromCallable(
            nqoi_obs,
            self._nparams,
            self._evaluate_observation,
            bkd,
        )
        self._pred_model = FunctionFromCallable(
            nqoi_pred,
            self._nparams,
            self._evaluate_prediction,
            bkd,
        )

        noise_variances = bkd.full(
            (nqoi_obs,), self._noise_std ** 2,
        )
        self._inference_problem = BayesianInferenceProblem(
            obs_map=self._obs_model,
            prior=self._prior,
            noise_variances=noise_variances,
            bkd=bkd,
        )

    def _wire_inference_problem(self) -> None:
        """Hand the substrate to :class:`PredictionOEDProblem`.

        Subclasses should not need to override this. It exists as a
        separate method so subclasses that replace the substrate can
        trigger the base-class wiring via a single call.
        """
        PredictionOEDProblem.__init__(
            self,
            inference_problem=self._inference_problem,
            qoi_map=self._pred_model,
            design_conditions=self._sensor_locations,
            bkd=self._bkd,
        )

    # ---- public config accessors ----------------------------------------

    def noise_std(self) -> float:
        """Standard deviation of the observation noise."""
        return self._noise_std

    def source_mode(self) -> str:
        """KLE source mode: ``"forcing"`` or ``"initial_condition"``."""
        return self._source_mode

    # ---- Stokes solve (cached entry point) ------------------------------

    def _compute_stokes_result(
        self,
        vel_shape_a: float,
        vel_shape_b: float,
        reynolds_num: float,
    ) -> Tuple[
        Array,
        "StokesPhysics[Array]",
        "VectorLagrangeBasis[Array]",
        "LagrangeBasis[Array]",
    ]:
        """Solve Stokes on the problem's Stokes mesh.

        Factored out of :meth:`_solve_forward_full` so callers that
        need to evaluate the problem at many samples with the same
        velocity parameters can cache the returned tuple and pass it
        to :meth:`_solve_adr_transient` instead of re-solving Stokes
        for every sample.
        """
        return _solve_stokes(
            self._stokes_mesh,
            self._bkd,
            reynolds_num,
            [vel_shape_a, vel_shape_b],
        )

    # ---- ADR transient ---------------------------------------------------

    def _solve_adr_transient(
        self,
        kle_params: np.ndarray,
        stokes_result: Tuple[
            Array,
            "StokesPhysics[Array]",
            "VectorLagrangeBasis[Array]",
            "LagrangeBasis[Array]",
        ],
    ) -> Tuple[Array, Array]:
        """Run the ADR transient using a precomputed Stokes result."""
        from pyapprox.ode.config import TimeIntegrationConfig
        from pyapprox.pde.galerkin.boundary.implementations import RobinBC
        from pyapprox.pde.galerkin.physics.advection_diffusion import (
            AdvectionDiffusionReaction,
        )
        from pyapprox.pde.galerkin.time_integration.galerkin_model import (
            GalerkinModel,
        )

        bkd = self._bkd
        sol, stokes, vel_basis, pres_basis = stokes_result

        vel_callable = _extract_velocity_callable(
            sol, stokes, vel_basis, pres_basis, self._adr_basis, bkd,
        )

        kle_coeffs = bkd.asarray(np.asarray(kle_params).ravel())
        kle_vals = self._kle_map(kle_coeffs)
        kle_nodal = bkd.to_numpy(kle_vals)

        adr_skfem = self._adr_basis.skfem_basis()
        forcing_func: Optional[Callable[..., np.ndarray]]
        if self._source_mode == "forcing":
            forcing_interp = adr_skfem.interpolator(kle_nodal)

            def forcing_func(
                x: np.ndarray, time: float = 0.0,
            ) -> np.ndarray:
                return np.asarray(forcing_interp(x))
        else:
            forcing_func = None

        alpha = 0.1
        robin_bcs: List["BoundaryConditionProtocol[Array]"] = [
            RobinBC(self._adr_basis, "left", alpha, 0.0, bkd),
            RobinBC(self._adr_basis, "right", alpha, 0.0, bkd),
        ]

        adr = AdvectionDiffusionReaction(
            basis=self._adr_basis,
            diffusivity=self._diffusivity,
            bkd=bkd,
            velocity=vel_callable,
            forcing=forcing_func,
            boundary_conditions=robin_bcs,
        )

        model = GalerkinModel(adr, bkd)
        if self._source_mode == "initial_condition":
            ic = kle_nodal.astype(np.float64)
        else:
            ic = bkd.to_numpy(bkd.zeros((adr.nstates(),)))
        config = TimeIntegrationConfig(
            method="backward_euler",
            final_time=self._final_time,
            deltat=self._deltat,
            newton_tol=1e-8,
            newton_maxiter=5,
        )
        return model.solve_transient(bkd.asarray(ic), config)

    # ---- forward-solve wrappers -----------------------------------------

    def _solve_forward_full(
        self, params_np: np.ndarray,
    ) -> Dict[str, Any]:
        """Full forward solve returning all intermediates.

        Parent implementation expects a ``(nparams,)`` vector where
        the last three entries are
        ``(vel_shape_a, vel_shape_b, reynolds_num)``. Subclasses that
        fix the velocity override this method.

        Returns a dict with keys ``"adr_solutions"``, ``"adr_times"``,
        ``"stokes_sol"``, ``"stokes_physics"``, ``"vel_basis"``.
        """
        kle_params = params_np[: self._nkle_terms]
        vel_shape_a = float(params_np[self._nkle_terms])
        vel_shape_b = float(params_np[self._nkle_terms + 1])
        reynolds_num = float(params_np[self._nkle_terms + 2])

        stokes_result = self._compute_stokes_result(
            vel_shape_a, vel_shape_b, reynolds_num,
        )
        sol, stokes, vel_basis, _ = stokes_result
        solutions, times = self._solve_adr_transient(
            kle_params, stokes_result,
        )
        return {
            "adr_solutions": solutions,
            "adr_times": times,
            "stokes_sol": sol,
            "stokes_physics": stokes,
            "vel_basis": vel_basis,
        }

    def _solve_forward(
        self, params_np: np.ndarray,
    ) -> Tuple[Array, Array]:
        """Full forward solve for a single parameter vector."""
        result = self._solve_forward_full(params_np)
        return result["adr_solutions"], result["adr_times"]

    # ---- plotting assembly ----------------------------------------------

    def _assemble_plotting_dict(
        self, forward_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Translate a ``_solve_forward_full`` dict into plotting data.

        Split out from :meth:`solve_for_plotting` so subclasses with
        a reduced parameter space (e.g. fixed velocity) can reuse the
        exact same assembly path without round-tripping through a
        padded full-dim vector.
        """
        bkd = self._bkd
        stokes = forward_result["stokes_physics"]
        vel_ndofs = stokes.vel_ndofs()
        stokes_sol_np = bkd.to_numpy(forward_result["stokes_sol"])

        vel_x = stokes_sol_np[:vel_ndofs:2]
        vel_y = stokes_sol_np[1:vel_ndofs:2]
        vel_mag = np.sqrt(vel_x ** 2 + vel_y ** 2)

        adr_solutions = forward_result["adr_solutions"]
        concentration = bkd.to_numpy(adr_solutions[:, -1])

        vel_basis = forward_result["vel_basis"]
        scalar_vel_skfem = vel_basis.scalar_basis().skfem_basis()

        return {
            "vel_x": vel_x,
            "vel_y": vel_y,
            "vel_magnitude": vel_mag,
            "vel_scalar_skfem_basis": scalar_vel_skfem,
            "stokes_skfem_mesh": self._stokes_mesh.skfem_mesh(),
            "concentration": concentration,
            "adr_skfem_basis": self._adr_basis.skfem_basis(),
            "adr_skfem_mesh": self._adr_mesh.skfem_mesh(),
        }

    def solve_for_plotting(
        self, sample: Array,
    ) -> Dict[str, Any]:
        """Solve for a single parameter sample, returning plotting data."""
        params_np = self._bkd.to_numpy(sample).ravel()
        return self._assemble_plotting_dict(
            self._solve_forward_full(params_np),
        )

    # ---- obs/pred extraction --------------------------------------------

    def _extract_obs(self, solutions: Array) -> np.ndarray:
        """Extract observations from a forward solve solution."""
        final_sol = self._bkd.to_numpy(solutions[:, -1])
        return final_sol[self._sensor_indices]

    def _extract_pred(self, solutions: Array) -> np.ndarray:
        """Extract predictions from a forward solve solution."""
        nodes_np = self._bkd.to_numpy(self._adr_mesh.nodes())
        target_mask = nodes_np[0, :] >= self._target_threshold
        final_sol = self._bkd.to_numpy(solutions[:, -1])
        return np.array([np.mean(final_sol[target_mask])])

    def _evaluate_observation(self, samples: Array) -> Array:
        """Evaluate observation model. samples: ``(nparams, nsamples)``."""
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = []
        for ii in range(nsamples):
            solutions, _ = self._solve_forward(samples_np[:, ii])
            results.append(self._extract_obs(solutions))
        return bkd.asarray(np.column_stack(results))

    def _evaluate_prediction(self, samples: Array) -> Array:
        """Evaluate prediction model. samples: ``(nparams, nsamples)``."""
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = []
        for ii in range(nsamples):
            solutions, _ = self._solve_forward(samples_np[:, ii])
            results.append(self._extract_pred(solutions))
        return bkd.asarray(np.column_stack(results))

    # ---- public PDE evaluation API --------------------------------------

    def evaluate_both(
        self, samples: Array,
    ) -> Tuple[Array, Array]:
        """Evaluate observation and prediction maps with a single solve."""
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        obs_results = []
        pred_results = []
        for ii in range(nsamples):
            solutions, _ = self._solve_forward(samples_np[:, ii])
            obs_results.append(self._extract_obs(solutions))
            pred_results.append(self._extract_pred(solutions))
        return (
            bkd.asarray(np.column_stack(obs_results)),
            bkd.asarray(np.column_stack(pred_results)),
        )

    def evaluate_nodal(self, samples: Array) -> Array:
        """Evaluate full nodal concentration at the final time."""
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = []
        for ii in range(nsamples):
            solutions, _ = self._solve_forward(samples_np[:, ii])
            final_sol = bkd.to_numpy(solutions[:, -1])
            results.append(final_sol)
        return bkd.asarray(np.column_stack(results))

    def mesh_nodes(self) -> Array:
        """Return all ADR mesh node coordinates. Shape: ``(2, nnodes)``."""
        return self._adr_mesh.nodes()

    def nnodes(self) -> int:
        """Return number of ADR mesh nodes."""
        return self._adr_mesh.nnodes()


# ---------------------------------------------------------------------------
# FixedVelocityAdvectionDiffusionOEDProblem
# ---------------------------------------------------------------------------


class FixedVelocityAdvectionDiffusionOEDProblem(
    AdvectionDiffusionOEDProblem[Array], Generic[Array],
):
    """Fixed-velocity variant of :class:`AdvectionDiffusionOEDProblem`.

    Pins ``(vel_shape_a, vel_shape_b, reynolds_num)`` at construction,
    solves Stokes once, and reduces the parameter space to the
    ``nkle_terms`` Gaussian KLE marginals (drops the three uniform
    velocity marginals). Overrides only
    :meth:`_build_pde_substrate` — the inherited ``__init__`` still
    drives the two-step template.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        *,
        vel_shape_a: float = 2.5,
        vel_shape_b: float = 2.5,
        reynolds_num: float = 12.5,
        noise_std: float = 0.1,
        **problem_kwargs: Any,
    ) -> None:
        self._vel_shape_a = float(vel_shape_a)
        self._vel_shape_b = float(vel_shape_b)
        self._reynolds_num = float(reynolds_num)
        super().__init__(bkd, noise_std=noise_std, **problem_kwargs)

    def _build_pde_substrate(self) -> None:
        """Override: build reduced-prior substrate with cached Stokes.

        Delegates mesh/KLE/sensor construction to the parent, then
        replaces the prior, reduces ``_nparams``, rebuilds
        obs/pred/inference on the reduced dimensionality, and
        pre-computes the Stokes result.

        .. note::

           The parent's ``_build_pde_substrate`` builds a full 13-dim
           ``_obs_model`` / ``_pred_model`` / ``_inference_problem``
           that we immediately replace. This is intentional and
           cheap: the replaced objects are thin wrappers (no PDE
           solves happen at construction), and keeping the parent
           implementation straightforward is worth the few
           microseconds of redundant allocation. A conditional on
           "am I the parent or a subclass" would couple the parent
           to its subclasses, which the template-method split is
           meant to avoid.
        """
        super()._build_pde_substrate()

        bkd = self._bkd
        nkle = self._nkle_terms

        reduced_marginals = list(self._prior.marginals()[:nkle])
        self._prior = IndependentJoint(reduced_marginals, bkd)
        self._nparams = nkle

        nqoi_obs = self._nsensors
        self._obs_model = FunctionFromCallable(
            nqoi_obs, nkle, self._evaluate_observation, bkd,
        )
        self._pred_model = FunctionFromCallable(
            1, nkle, self._evaluate_prediction, bkd,
        )

        noise_variances = bkd.full(
            (nqoi_obs,), self._noise_std ** 2,
        )
        self._inference_problem = BayesianInferenceProblem(
            obs_map=self._obs_model,
            prior=self._prior,
            noise_variances=noise_variances,
            bkd=bkd,
        )

        self._cached_stokes_result = self._compute_stokes_result(
            self._vel_shape_a, self._vel_shape_b, self._reynolds_num,
        )

    # ---- pinned velocity accessors --------------------------------------

    def vel_shape_a(self) -> float:
        return self._vel_shape_a

    def vel_shape_b(self) -> float:
        return self._vel_shape_b

    def reynolds_num(self) -> float:
        return self._reynolds_num

    # ---- forward solve (reduced dimension) ------------------------------

    def _solve_forward_full(
        self, params_np: np.ndarray,
    ) -> Dict[str, Any]:
        """Override: reuse cached Stokes; expects a pure KLE vector.

        After the reduction done in :meth:`_build_pde_substrate`,
        ``params_np`` has shape exactly ``(nkle_terms,)`` — there
        are no velocity indices in the vector at all.
        """
        assert params_np.shape == (self._nkle_terms,), (
            f"FixedVelocityAdvectionDiffusionOEDProblem expects a "
            f"({self._nkle_terms},) KLE vector, got shape "
            f"{params_np.shape}"
        )
        solutions, times = self._solve_adr_transient(
            params_np, self._cached_stokes_result,
        )
        sol, stokes, vel_basis, _ = self._cached_stokes_result
        return {
            "adr_solutions": solutions,
            "adr_times": times,
            "stokes_sol": sol,
            "stokes_physics": stokes,
            "vel_basis": vel_basis,
        }

    def solve_for_plotting(
        self, sample: Array,
    ) -> Dict[str, Any]:
        """Plot a reduced KLE sample using the cached Stokes result."""
        kle_np = self._bkd.to_numpy(sample).ravel()
        assert kle_np.shape == (self._nkle_terms,), (
            f"solve_for_plotting expects a ({self._nkle_terms},) "
            f"KLE vector, got shape {kle_np.shape}"
        )
        return self._assemble_plotting_dict(
            self._solve_forward_full(kle_np),
        )
