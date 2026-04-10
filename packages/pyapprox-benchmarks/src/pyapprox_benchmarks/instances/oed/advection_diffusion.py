"""Obstructed advection-diffusion OED benchmark.

Couples Stokes flow with advection-diffusion-reaction on an obstructed
domain. The Stokes velocity field drives advection, and a KLE-based
log-normal random field provides forcing. Parameters control the KLE
coefficients, inlet velocity shape, and Reynolds number.
"""

# TODO: Should benchmarks be defined here or in benchmarks module.
# Decide and document rule, and place in benchmarks.CONVENTIONS.md

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
)

if TYPE_CHECKING:
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D
    from pyapprox.pde.galerkin.physics.stokes import StokesPhysics

import numpy as np
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.problems.inverse import BayesianInferenceProblem
from pyapprox_benchmarks.problems.oed import PredictionOEDProblem
from pyapprox_benchmarks.registry import BenchmarkRegistry


def _insert_grid_line(vals: np.ndarray, new_val: float) -> np.ndarray:
    """Insert ``new_val`` into ``vals`` if not already present (within tol)."""
    tol = 1e-12
    if np.any(np.abs(vals - new_val) <= tol):
        return vals.copy()
    merged = np.concatenate([vals, [new_val]])
    return np.sort(merged)


def _recompute_obstruction_indices(
    old_xintervals: np.ndarray,
    old_yintervals: np.ndarray,
    old_obstruction_indices: np.ndarray,
    new_xintervals: np.ndarray,
    new_yintervals: np.ndarray,
) -> np.ndarray:
    """Re-map obstruction cell indices after inserting new grid lines.

    Obstruction indices are row-major with x varying fastest:
        idx = row * (nx - 1) + col
    where nx is the number of x grid lines. Each obstruction cell has
    fixed (xlo, xhi, ylo, yhi) coordinates which we recover from the
    old grid, then locate in the new grid.
    """
    old_ncols = old_xintervals.shape[0] - 1
    new_ncols = new_xintervals.shape[0] - 1
    tol = 1e-12

    new_indices = []
    for idx in old_obstruction_indices:
        row = int(idx) // old_ncols
        col = int(idx) % old_ncols
        xlo = old_xintervals[col]
        ylo = old_yintervals[row]
        # Find the new (row, col) whose lower-left corner matches (xlo, ylo).
        new_col = int(np.argmin(np.abs(new_xintervals[:-1] - xlo)))
        new_row = int(np.argmin(np.abs(new_yintervals[:-1] - ylo)))
        if (
            abs(new_xintervals[new_col] - xlo) > tol
            or abs(new_yintervals[new_row] - ylo) > tol
        ):
            raise RuntimeError(
                f"Could not recover obstruction cell {idx} at "
                f"({xlo}, {ylo}) in the refined grid."
            )
        new_indices.append(new_row * new_ncols + new_col)
    return np.array(new_indices, dtype=int)


def _build_obstructed_mesh(
    bkd: Backend[Array],
    nrefine: int,
    kle_subdomain: Optional[Tuple[float, float, float, float]] = None,
) -> "ObstructedMesh2D[Array]":
    """Create the obstructed domain mesh.

    If ``kle_subdomain`` is provided, its boundary coordinates are
    inserted into the x/y interval grids so that subdomain edges are
    mesh grid lines. This prevents any element from straddling the
    subdomain boundary after uniform refinement (which only halves
    existing cells). Obstruction cell indices are re-mapped
    automatically to account for the inserted grid lines.
    """
    from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D

    xintervals = np.array([0, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 1.0])
    yintervals = np.linspace(0, 1, 5)
    obstruction_indices = np.array([3, 6, 13], dtype=int)

    if kle_subdomain is not None:
        xmin, xmax, ymin, ymax = kle_subdomain
        if not (
            xintervals[0] <= xmin < xmax <= xintervals[-1]
            and yintervals[0] <= ymin < ymax <= yintervals[-1]
        ):
            raise ValueError(
                f"kle_subdomain {kle_subdomain} must lie inside the base "
                f"domain [{xintervals[0]}, {xintervals[-1]}] x "
                f"[{yintervals[0]}, {yintervals[-1]}] with xmin<xmax and "
                f"ymin<ymax."
            )
        new_x = xintervals.copy()
        new_y = yintervals.copy()
        for v in (xmin, xmax):
            new_x = _insert_grid_line(new_x, float(v))
        for v in (ymin, ymax):
            new_y = _insert_grid_line(new_y, float(v))
        obstruction_indices = _recompute_obstruction_indices(
            xintervals, yintervals, obstruction_indices, new_x, new_y,
        )
        xintervals = new_x
        yintervals = new_y

    return ObstructedMesh2D(
        xintervals,
        yintervals,
        obstruction_indices,
        bkd,
        nrefine=nrefine,
    )


def _solve_stokes(
    mesh: "ObstructedMesh2D[Array]",
    bkd: Backend[Array],
    reynolds_num: float,
    vel_shape_params: List[float],
) -> Tuple[
    Array,
    "StokesPhysics[Array]",
    "VectorLagrangeBasis[Array]",
    "LagrangeBasis[Array]",
]:
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

    def inlet_func(x: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Parabolic-like inlet: y^(a-1) * (1-y)^(b-1)."""
        y = x[1]
        npts = x.shape[1]
        vals = np.zeros((npts, 2))
        vals[:, 0] = y ** (a - 1) * (1 - y) ** (b - 1)
        return vals

    def zero_vel(x: np.ndarray, time: float = 0.0) -> np.ndarray:
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
    sol: Array,
    stokes: "StokesPhysics[Array]",
    vel_basis: "VectorLagrangeBasis[Array]",
    pres_basis: "LagrangeBasis[Array]",
    adr_basis: "LagrangeBasis[Array]",
    bkd: Backend[Array],
) -> Callable[[np.ndarray], np.ndarray]:
    """Extract velocity from Stokes solution as callable for ADR.

    Returns a callable that accepts points of shape (2, ...) and returns
    velocity of shape (2, ...), preserving any extra dimensions (e.g.,
    quadrature point structure from skfem).
    """
    vel_ndofs = stokes.vel_ndofs()
    vel_state_np = bkd.to_numpy(sol[:vel_ndofs])

    # skfem ElementVector DOFs are interleaved:
    # [comp0_dof0, comp1_dof0, comp0_dof1, comp1_dof1, ...]
    vel_x = vel_state_np[0::2]
    vel_y = vel_state_np[1::2]

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
    def velocity_field(x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape[1:]
        x_flat = x.reshape(2, -1)
        vx = vel_x_interp(x_flat).reshape(orig_shape)
        vy = vel_y_interp(x_flat).reshape(orig_shape)
        return np.stack([vx, vy], axis=0)

    return velocity_field


class _ScatteredFieldMap(Generic[Array]):
    """Scatter a subdomain field into a full-length ADR nodal vector.

    Wraps an inner ``TransformedFieldMap`` that evaluates on a subset of
    ADR mesh nodes (the KLE subdomain) and produces a length-``ntotal``
    output in which entries **outside** the subdomain are exactly zero.

    Because the ADR benchmark is NumPy-only (skfem dependency), this
    class uses direct NumPy indexed assignment and then ``bkd.asarray``
    back. No sparse/dense scatter matrix is needed.
    """

    def __init__(
        self,
        inner: TransformedFieldMap[Array],
        global_node_indices: np.ndarray,
        ntotal: int,
        bkd: Backend[Array],
    ) -> None:
        self._inner = inner
        self._bkd = bkd
        self._ntotal = ntotal
        self._global_node_indices = np.asarray(global_node_indices, dtype=np.int64)

    def nvars(self) -> int:
        return self._inner.nvars()

    def __call__(self, params_1d: Array) -> Array:
        sub_vals = self._bkd.to_numpy(self._inner(params_1d))
        full = np.zeros(self._ntotal, dtype=np.float64)
        full[self._global_node_indices] = sub_vals
        return self._bkd.asarray(full)


def _compute_lumped_mass_weights(
    skfem_mesh: Any,
    elem_idx: np.ndarray,
    global_nodes: np.ndarray,
) -> np.ndarray:
    """Lumped mass weights on subdomain nodes via shoelace element area.

    Distributes each element's area uniformly across its ``nverts``
    vertices. Matches the pattern used in ``cantilever_beam.py``.
    """
    nverts = skfem_mesh.t.shape[0]
    n_nodes = global_nodes.shape[0]
    node_to_local = {int(n): i for i, n in enumerate(global_nodes)}
    sub_w = np.zeros(n_nodes)
    for e in elem_idx:
        verts = skfem_mesh.p[:, skfem_mesh.t[:, e]]
        x, y = verts[0], verts[1]
        area = 0.5 * abs(
            sum(
                x[i] * y[(i + 1) % nverts] - x[(i + 1) % nverts] * y[i]
                for i in range(nverts)
            )
        )
        for n in skfem_mesh.t[:, e]:
            sub_w[node_to_local[int(n)]] += area / nverts
    return sub_w


def _create_subdomain_kle_forcing(
    adr_basis: "LagrangeBasis[Array]",
    nkle_terms: int,
    correlation_length: float,
    sigma: float,
    subdomain: Optional[Tuple[float, float, float, float]],
    bkd: Backend[Array],
) -> "_ScatteredFieldMap[Array]":
    """Create lognormal KLE forcing localized to a rectangular subdomain.

    Builds a lognormal KLE on the ADR mesh nodes that fall inside the
    given rectangle (via element-centroid predicate), using lumped-mass
    quadrature weights on the subdomain nodes. Outside the subdomain,
    the forcing is exactly zero at every ADR node.

    If ``subdomain`` is ``None``, the KLE is constructed on the full
    ADR mesh (backward-compatible behavior).
    """
    from pyapprox.pde.field_maps.kle_factory import (
        create_lognormal_kle_field_map,
    )

    skfem_mesh = adr_basis.mesh().skfem_mesh()
    ntotal = int(skfem_mesh.p.shape[1])

    if subdomain is None:
        elem_idx = np.arange(skfem_mesh.t.shape[1])
        global_nodes = np.arange(ntotal)
    else:
        xmin, xmax, ymin, ymax = subdomain
        # Select elements whose centroid lies inside the rectangle.
        # NOTE: The caller is responsible for ensuring subdomain bounds
        # coincide with mesh grid lines (see _build_obstructed_mesh's
        # kle_subdomain-aware xintervals/yintervals), so no elements
        # straddle the subdomain boundary.
        centroids = skfem_mesh.p[:, skfem_mesh.t].mean(axis=1)
        mask = (
            (centroids[0] >= xmin)
            & (centroids[0] <= xmax)
            & (centroids[1] >= ymin)
            & (centroids[1] <= ymax)
        )
        elem_idx = np.where(mask)[0]
        if elem_idx.size == 0:
            raise ValueError(
                f"KLE subdomain {subdomain} contains no mesh elements. "
                f"Check bounds against the domain extents."
            )
        global_nodes = np.unique(skfem_mesh.t[:, elem_idx].ravel())

    nsub = int(global_nodes.shape[0])
    if nsub < nkle_terms:
        raise ValueError(
            f"KLE subdomain has only {nsub} mesh nodes, but nkle_terms="
            f"{nkle_terms} was requested. Reduce nkle_terms or enlarge "
            f"the subdomain / increase mesh refinement."
        )

    coords_sub = skfem_mesh.p[:, global_nodes]
    sub_w = _compute_lumped_mass_weights(skfem_mesh, elem_idx, global_nodes)

    mean_log = bkd.zeros((nsub,))
    inner = create_lognormal_kle_field_map(
        mesh_coords=bkd.asarray(coords_sub),
        mean_log_field=mean_log,
        bkd=bkd,
        correlation_length=correlation_length,
        num_kle_terms=nkle_terms,
        sigma=sigma,
        quad_weights=bkd.asarray(sub_w),
    )

    return _ScatteredFieldMap(inner, global_nodes, ntotal, bkd)


class ObstructedAdvectionDiffusionOEDBenchmark(Generic[Array]):
    """Obstructed advection-diffusion OED benchmark.

    Couples Stokes flow with transient advection-diffusion on an
    obstructed domain [0,1]^2 with three rectangular obstacles.

    .. note::

       **Backend restriction.** This benchmark uses skfem for the
       Stokes and ADR solves, and skfem is NumPy-only. The constructor
       therefore rejects any backend other than :class:`NumpyBkd`. If
       you need torch tensors at the call site, convert at the
       boundary; gradients will not propagate through the forward
       solve because skfem does not participate in ``torch.autograd``.

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
    noise_std : float
        Observation noise standard deviation. Default 0.1.
    nstokes_refine : int
        Number of mesh refinements for Stokes solve.
    nadvec_diff_refine : int
        Number of mesh refinements for ADR solve.
    nkle_terms : int
        Number of KLE terms for forcing field.
    nsensors : int
        Number of sensor locations.
    diffusivity : float
        Diffusion coefficient for the ADR equation. Default 0.1.
    final_time : float
        Final time for the transient solve. Default 1.5.
    deltat : float
        Time step for the transient solve. Default 0.25.
    kle_subdomain : tuple of floats, optional
        Rectangular subdomain ``(xmin, xmax, ymin, ymax)`` in which the
        log-normal KLE forcing lives. Forcing is exactly zero at every
        ADR mesh node outside the rectangle. Elements that straddle the
        rectangle boundary interpolate linearly from the subdomain edge
        value down to zero (standard P1 FE interpretation). Default
        ``(0.0, 0.25, 0.0, 1.0)``, placing the source in the leftmost
        L/4 strip so the inference problem is genuinely informative at
        downstream sensors. Pass ``None`` to recover a full-domain KLE.
    kle_correlation_length : float
        Isotropic correlation length of the squared-exponential KLE
        kernel. Default 0.1.
    kle_sigma : float
        Standard deviation of the log-field. Default 0.3.
    source_mode : {"forcing", "initial_condition"}
        How the KLE nodal field drives the ADR transient:

        - ``"forcing"`` (default): the KLE field acts as a time-constant
          right-hand-side forcing, with zero initial concentration. This
          is the standard dye-release-style release.
        - ``"initial_condition"``: the KLE field is used as the initial
          concentration ``u(x, 0)``, and the right-hand-side forcing is
          zero. The plume then relaxes under advection-diffusion alone.

        Everything else (parameters, prior, sensors, QoI) is identical
        across the two modes, so they can be A/B compared under matched
        KLE draws and physics.
    """

    def __init__(
        self,
        bkd: Backend[Array],
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
        # skfem is NumPy-only, so every forward solve in this benchmark
        # runs on NumPy regardless of the caller's backend. Rather than
        # silently converting at the call boundary (which would make the
        # torch autograd graph through ``obs_map()`` / ``qoi_map()``
        # silently return zeros), require NumpyBkd explicitly. Users who
        # need to call this benchmark from torch code should convert at
        # their own boundary and accept that gradients do not propagate
        # through the forward solve. See the class docstring for the
        # recommended pattern.
        from pyapprox.util.backends.numpy import NumpyBkd

        if not isinstance(bkd, NumpyBkd):
            raise TypeError(
                "ObstructedAdvectionDiffusionOEDBenchmark requires "
                "NumpyBkd because its forward solve uses skfem, which "
                "is NumPy-only. Got "
                f"{type(bkd).__name__}. If you need torch tensors at "
                "the call site, construct the benchmark with NumpyBkd() "
                "and convert at the boundary (see the class docstring "
                "for the recommended pattern). Gradients will not "
                "propagate through the forward solve."
            )
        self._bkd = bkd
        self._nkle_terms = nkle_terms
        self._nsensors = nsensors
        self._nstokes_refine = nstokes_refine
        self._nadvec_diff_refine = nadvec_diff_refine
        self._diffusivity = diffusivity
        self._kle_subdomain = kle_subdomain
        self._kle_correlation_length = kle_correlation_length
        self._kle_sigma = kle_sigma
        self._source_mode = source_mode

        # Build prior: [kle_params (10), vel_shape (2), reynolds (1)]
        marginals: List[Union[GaussianMarginal[Array], UniformMarginal[Array]]] = []
        for _ in range(nkle_terms):
            marginals.append(GaussianMarginal(0.0, 1.0, bkd))
        marginals.append(UniformMarginal(2.0, 3.0, bkd))
        marginals.append(UniformMarginal(2.0, 3.0, bkd))
        marginals.append(UniformMarginal(5.0, 20.0, bkd))
        self._prior = IndependentJoint(marginals, bkd)
        self._nparams = nkle_terms + 3

        # Build mesh and ADR basis (shared across evaluations). The ADR
        # mesh is built with the KLE subdomain boundaries inserted as
        # grid lines so no element straddles the subdomain edge.
        self._stokes_mesh = _build_obstructed_mesh(bkd, nstokes_refine)
        self._adr_mesh = _build_obstructed_mesh(
            bkd, nadvec_diff_refine, kle_subdomain=kle_subdomain,
        )

        from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis

        self._adr_basis = LagrangeBasis(self._adr_mesh, degree=1)

        # Create subdomain-localized KLE forcing map
        self._kle_map = _create_subdomain_kle_forcing(
            self._adr_basis,
            nkle_terms,
            kle_correlation_length,
            kle_sigma,
            kle_subdomain,
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
        self._deltat = deltat
        self._final_time = final_time

        # Create FunctionProtocol wrappers
        nqoi_obs = nsensors
        nqoi_pred = 1

        def obs_callable(samples: Array) -> Array:
            return self._evaluate_observation(samples)

        def pred_callable(samples: Array) -> Array:
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

        # Create BayesianInferenceProblem
        noise_variances = bkd.full((nqoi_obs,), noise_std**2)
        self._inference_problem = BayesianInferenceProblem(
            obs_map=self._obs_model,
            prior=self._prior,
            noise_variances=noise_variances,
            bkd=bkd,
        )

        # Compose into PredictionOEDProblem
        self._problem = PredictionOEDProblem(
            inference_problem=self._inference_problem,
            qoi_map=self._pred_model,
            design_conditions=self._sensor_locations,
            bkd=bkd,
        )

    def problem(self) -> PredictionOEDProblem[Array]:
        """Get the prediction OED problem."""
        return self._problem

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
        """Solve Stokes on the benchmark's Stokes mesh.

        Factored out of :meth:`_solve_forward_full` so callers that need
        to evaluate the benchmark at many samples with the same velocity
        parameters (Reynolds number and inlet shape) can cache the
        returned tuple and pass it to :meth:`_solve_adr_transient`
        instead of re-solving Stokes for every sample.
        """
        return _solve_stokes(
            self._stokes_mesh,
            self._bkd,
            reynolds_num,
            [vel_shape_a, vel_shape_b],
        )

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
        """Run the ADR transient using a precomputed Stokes result.

        Given a ``stokes_result`` tuple produced by
        :meth:`_compute_stokes_result` (or ``_solve_stokes`` directly),
        extracts the velocity callable on the benchmark's ADR basis,
        assembles the KLE forcing / initial condition according to
        ``self._source_mode``, and integrates
        :class:`AdvectionDiffusionReaction` to ``self._final_time`` with
        backward Euler at step size ``self._deltat``.

        Parameters
        ----------
        kle_params
            1D array of the first ``nkle_terms`` parameters (the KLE
            coefficients). Shape ``(nkle_terms,)``.
        stokes_result
            ``(sol, stokes, vel_basis, pres_basis)``.

        Returns
        -------
        (solutions, times)
            ``solutions`` has shape ``(nstates, ntimes)`` and ``times``
            has shape ``(ntimes,)``.
        """
        from pyapprox.ode.config import TimeIntegrationConfig
        from pyapprox.pde.galerkin.boundary.implementations import (
            RobinBC,
        )
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

        # Evaluate KLE nodal field (same call for both source modes).
        kle_coeffs = bkd.asarray(np.asarray(kle_params).ravel())
        kle_vals = self._kle_map(kle_coeffs)
        kle_nodal = bkd.to_numpy(kle_vals)

        # In forcing mode the KLE field interpolates onto ADR quadrature
        # points as a right-hand-side source; in initial-condition mode
        # the ADR equation has no forcing and the KLE field seeds u(x,0).
        adr_skfem = self._adr_basis.skfem_basis()
        forcing_func: Optional[Callable[..., np.ndarray]]
        if self._source_mode == "forcing":
            forcing_interp = adr_skfem.interpolator(kle_nodal)

            def forcing_func(
                x: np.ndarray, time: float = 0.0,
            ) -> np.ndarray:
                return forcing_interp(x)
        else:
            forcing_func = None

        # Robin BCs on left/right (alpha=0.1, g=0)
        alpha = 0.1
        robin_bcs = [
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

        # Solve transient. In initial-condition mode the KLE field seeds
        # u(x, 0); in forcing mode the initial concentration is zero.
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

    def _solve_forward_full(
        self, params_np: np.ndarray,
    ) -> Dict[str, Any]:
        """Full forward solve returning all intermediates.

        Thin wrapper that solves Stokes with
        :meth:`_compute_stokes_result` and then runs the ADR transient
        with :meth:`_solve_adr_transient`. External callers that want
        to amortize the Stokes solve across many samples (e.g. a
        fixed-velocity sweep) should call those two methods directly.

        Returns a dict with keys:
            - "adr_solutions": Array, shape (ndofs, ntimes)
            - "adr_times": Array, shape (ntimes,)
            - "stokes_sol": Array, Stokes solution vector
            - "stokes_physics": StokesPhysics object
            - "vel_basis": VectorLagrangeBasis for velocity
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

    def _solve_forward(self, params_np: np.ndarray) -> Tuple[Array, Array]:
        """Full forward solve for a single parameter vector."""
        result = self._solve_forward_full(params_np)
        return result["adr_solutions"], result["adr_times"]

    def solve_for_plotting(
        self, sample: Array,
    ) -> Dict[str, Any]:
        """Solve for a single parameter sample, returning data for plotting.

        Parameters
        ----------
        sample : Array
            Parameter vector. Shape: (nparams,) or (nparams, 1)

        Returns
        -------
        dict with keys:
            - "stokes_sol": np.ndarray, Stokes solution DOF vector
            - "vel_basis": VectorLagrangeBasis for velocity
            - "stokes_mesh": skfem mesh for Stokes
            - "concentration": np.ndarray, final-time concentration (nodal)
            - "adr_basis": LagrangeBasis for ADR
            - "adr_mesh": skfem mesh for ADR
        """
        bkd = self._bkd
        sample_np = bkd.to_numpy(sample).ravel()
        result = self._solve_forward_full(sample_np)

        stokes = result["stokes_physics"]
        vel_ndofs = stokes.vel_ndofs()
        stokes_sol_np = bkd.to_numpy(result["stokes_sol"])

        # Extract velocity components (interleaved DOF ordering)
        vel_x = stokes_sol_np[:vel_ndofs:2]
        vel_y = stokes_sol_np[1:vel_ndofs:2]
        vel_mag = np.sqrt(vel_x**2 + vel_y**2)

        adr_solutions = result["adr_solutions"]
        concentration = bkd.to_numpy(adr_solutions[:, -1])

        vel_basis = result["vel_basis"]
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
        """Evaluate observation model. samples: (nparams, nsamples)."""
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = []

        for ii in range(nsamples):
            solutions, times = self._solve_forward(samples_np[:, ii])
            results.append(self._extract_obs(solutions))

        return bkd.asarray(np.column_stack(results))

    def _evaluate_prediction(self, samples: Array) -> Array:
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

    def obs_map(self) -> FunctionFromCallable[Array]:
        """Return the observation map (FunctionProtocol)."""
        return self._obs_model

    def qoi_map(self) -> FunctionFromCallable[Array]:
        """Return the QoI map (FunctionProtocol)."""
        return self._pred_model

    def design_conditions(self) -> Array:
        """Return sensor locations. Shape: (2, nsensors)."""
        return self._sensor_locations

    def evaluate_nodal(self, samples: Array) -> Array:
        """Evaluate full nodal concentration at final time.

        Solves the forward PDE for each sample and returns the
        complete final-time concentration vector at all mesh nodes.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nparams, nsamples).

        Returns
        -------
        Array
            Nodal concentration values. Shape: (nnodes, nsamples).
        """
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = []

        for ii in range(nsamples):
            solutions, times = self._solve_forward(samples_np[:, ii])
            final_sol = bkd.to_numpy(solutions[:, -1])
            results.append(final_sol)

        return bkd.asarray(np.column_stack(results))

    def mesh_nodes(self) -> Array:
        """Return all ADR mesh node coordinates. Shape: (2, nnodes)."""
        return self._adr_mesh.nodes()

    def nnodes(self) -> int:
        """Return number of ADR mesh nodes."""
        return self._adr_mesh.nnodes()

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
) -> ObstructedAdvectionDiffusionOEDBenchmark[Array]:
    return ObstructedAdvectionDiffusionOEDBenchmark(bkd, noise_std=0.1)
