"""Cantilever beam benchmark instances for UQ workflows.

Wraps Galerkin FEM models (1D Euler-Bernoulli beam, 2D composite linear
elasticity, 2D composite Neo-Hookean hyperelasticity) into benchmark
instances with per-subdomain KLE priors for Young's modulus.

Each subdomain gets its own lognormal KLE field map. The benchmark input
parameters are the concatenated KLE coefficients (standard normal) from
all subdomains. A 1-term KLE recovers the constant-parameter case.

Model hierarchy:
    1D Euler-Bernoulli: EI(x) varies via KLE on bending stiffness
    2D linear elastic: E(x,y) varies per subdomain via KLE
    2D Neo-Hookean: E(x,y) varies per subdomain via KLE
"""

import os
from typing import Callable, Dict, Generic, List, Optional, Tuple

import numpy as np

# Default mesh path relative to this package
_DATA_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "data",
)
_DEFAULT_MESH_PATH = os.path.normpath(
    os.path.join(_DATA_DIR, "cantilever_beam_2d_with_holes.json")
)

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.typing.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.probability.univariate.gaussian import GaussianMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint
from pyapprox.typing.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.typing.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.typing.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.typing.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.typing.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
    create_spde_lognormal_kle_field_map,
)
from pyapprox.typing.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.typing.benchmarks.instances.pde.elastic_bar import (
    PDEBenchmarkWrapper,
)

try:
    from skfem.models.elasticity import lame_parameters
except ImportError:
    raise ImportError(
        "scikit-fem is required for cantilever beam benchmarks. "
        "Install with: pip install scikit-fem"
    )


# =========================================================================
# Utilities
# =========================================================================


def _create_subdomain_kle_field_maps(
    skfem_mesh,
    subdomain_elements: Dict[str, np.ndarray],
    subdomain_names: List[str],
    bkd,
    num_kle_terms: int,
    sigma: float,
    correlation_length: float,
    mean_log_E: float,
):
    """Create a lognormal KLE field map per subdomain at mesh nodes.

    Uses the Nystrom method at FEM mesh nodes (not quadrature points)
    to avoid allocating dense kernel matrices proportional to
    ``(nelems * nquad)^2``. For large meshes the quadrature-point
    approach can require tens of gigabytes; the nodal approach reduces
    memory by a factor of ``nquad^2`` (typically 9-16x for 2D quads).

    Eigenvectors live at the subdomain's unique mesh nodes. To obtain
    values at quadrature points, use ``skfem_basis.interpolate()``
    after scattering subdomain node values into a global nodal array.

    Parameters
    ----------
    skfem_mesh : skfem Mesh
        The FEM mesh (provides node coordinates and connectivity).
    subdomain_elements : dict
        Mapping from subdomain name to element index arrays.
    subdomain_names : list of str
        Ordered list of subdomain names.
    bkd : Backend
        Computational backend.
    num_kle_terms : int
        Number of KLE terms per subdomain.
    sigma : float
        Standard deviation of the log-field.
    correlation_length : float
        Correlation length for the KLE kernel.
    mean_log_E : float
        Mean of the log-field (log(E_mean) at KLE params=0).

    Returns
    -------
    field_maps : list of TransformedFieldMap
        One per subdomain. Each maps ``(nterms,)`` -> ``(n_sub_nodes,)``.
    subdomain_node_indices : list of np.ndarray
        Global node indices for each subdomain (for scattering into
        a global nodal array before interpolation).
    """
    ndim = skfem_mesh.p.shape[0]
    lenscale = bkd.full((ndim,), correlation_length)
    kernel = SquaredExponentialKernel(
        lenscale, (0.01, 10.0), ndim, bkd,
    )

    field_maps = []
    subdomain_node_indices = []
    for name in subdomain_names:
        elem_idx = subdomain_elements[name]
        # Unique global node indices in this subdomain
        global_nodes = np.unique(skfem_mesh.t[:, elem_idx].ravel())
        subdomain_node_indices.append(global_nodes)

        # Node coordinates for this subdomain
        coords_sub = skfem_mesh.p[:, global_nodes]  # (ndim, n_nodes)
        n_nodes = len(global_nodes)

        # Lumped mass weights: distribute element areas to nodes
        nverts = skfem_mesh.t.shape[0]
        sub_w = np.zeros(n_nodes)
        node_to_local = {int(n): i for i, n in enumerate(global_nodes)}
        for e in elem_idx:
            verts = skfem_mesh.p[:, skfem_mesh.t[:, e]]
            x, y = verts[0], verts[1]
            area = 0.5 * abs(sum(
                x[i] * y[(i + 1) % nverts] - x[(i + 1) % nverts] * y[i]
                for i in range(nverts)
            ))
            for n in skfem_mesh.t[:, e]:
                sub_w[node_to_local[int(n)]] += area / nverts

        mean_log = bkd.full((n_nodes,), mean_log_E)

        mesh_kle = MeshKLE(
            bkd.asarray(coords_sub), kernel,
            sigma=sigma, mean_field=0.0,
            nterms=num_kle_terms,
            quad_weights=bkd.asarray(sub_w),
            bkd=bkd,
        )

        inner = MeshKLEFieldMap(
            bkd, mean_log, mesh_kle.weighted_eigenvectors(),
        )
        fm = TransformedFieldMap(
            inner,
            transform=lambda x: bkd.exp(x),
            transform_deriv=lambda x: bkd.exp(x),
            bkd=bkd,
            transform_deriv2=lambda x: bkd.exp(x),
        )
        field_maps.append(fm)
    return field_maps, subdomain_node_indices


class _SkfemSubmesh(Generic[Array]):
    """Lightweight mesh wrapper for a skfem submesh.

    Wraps the result of ``skfem_mesh.restrict(elem_indices)`` so it
    satisfies ``GalerkinMeshProtocol`` and can be passed to
    ``LagrangeBasis``.
    """

    def __init__(self, skfem_submesh, bkd: Backend[Array]):
        self._skfem_mesh = skfem_submesh
        self._bkd = bkd
        self._nodes = bkd.asarray(
            skfem_submesh.p.astype(np.float64),
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return self._skfem_mesh.p.shape[0]

    def nelements(self) -> int:
        return self._skfem_mesh.nelements

    def nnodes(self) -> int:
        return self._skfem_mesh.nvertices

    def nodes(self) -> Array:
        return self._nodes

    def elements(self) -> Array:
        return self._bkd.asarray(
            self._skfem_mesh.t.astype(np.int64),
        )

    def skfem_mesh(self):
        return self._skfem_mesh

    def boundary_nodes(self, boundary_id: str) -> Array:
        bndries = self._skfem_mesh.boundaries or {}
        if boundary_id not in bndries:
            raise ValueError(
                f"Unknown boundary '{boundary_id}'. "
                f"Available: {list(bndries.keys())}"
            )
        facet_idx = bndries[boundary_id]
        node_idx = np.unique(self._skfem_mesh.facets[:, facet_idx])
        return self._bkd.asarray(node_idx.astype(np.int64))


def _create_subdomain_spde_kle_field_maps(
    skfem_mesh,
    subdomain_elements: Dict[str, np.ndarray],
    subdomain_names: List[str],
    bkd,
    num_kle_terms: int,
    sigma: float,
    correlation_length: float,
    mean_log_E: float,
):
    """Create a lognormal SPDE KLE field map per subdomain.

    For each subdomain, extracts a submesh via
    ``skfem_mesh.restrict(elem_indices)`` and runs the SPDE eigensolve
    on the submesh.  Uses sparse matrices and a partial eigensolve,
    giving O(N) memory per subdomain instead of the O(N^2) dense kernel
    matrices used by ``_create_subdomain_kle_field_maps``.

    The SPDE parameters ``gamma`` and ``delta`` are computed from the
    desired correlation length: ``gamma = correlation_length^2``,
    ``delta = 1``, giving ``l_c = sqrt(gamma/delta) = correlation_length``.

    Parameters
    ----------
    skfem_mesh : skfem Mesh
        The FEM mesh (provides node coordinates and connectivity).
    subdomain_elements : dict
        Mapping from subdomain name to element index arrays.
    subdomain_names : list of str
        Ordered list of subdomain names.
    bkd : Backend
        Computational backend.
    num_kle_terms : int
        Number of KLE terms per subdomain.
    sigma : float
        Standard deviation of the log-field.
    correlation_length : float
        Correlation length for the SPDE Matern field.
    mean_log_E : float
        Mean of the log-field (log(E_mean) at KLE params=0).

    Returns
    -------
    field_maps : list of TransformedFieldMap
        One per subdomain. Each maps ``(nterms,)`` -> ``(n_sub_nodes,)``.
    subdomain_node_indices : list of np.ndarray
        Global node indices for each subdomain (for scattering into
        a global nodal array before interpolation).
    """
    # SPDE params: l_c = sqrt(gamma/delta)
    gamma = correlation_length ** 2
    delta = 1.0

    field_maps = []
    subdomain_node_indices = []
    for name in subdomain_names:
        elem_idx = subdomain_elements[name]
        # Global node indices for this subdomain
        global_nodes = np.unique(skfem_mesh.t[:, elem_idx].ravel())
        subdomain_node_indices.append(global_nodes)
        n_nodes = len(global_nodes)

        # Build submesh and basis
        submesh_skfem = skfem_mesh.restrict(elem_idx)
        submesh = _SkfemSubmesh(submesh_skfem, bkd)
        basis = LagrangeBasis(submesh, degree=1)

        mean_log = bkd.full((n_nodes,), mean_log_E)

        fm = create_spde_lognormal_kle_field_map(
            basis, mean_log, bkd,
            n_modes=num_kle_terms, gamma=gamma, delta=delta,
            sigma=sigma,
        )
        field_maps.append(fm)
    return field_maps, subdomain_node_indices


def _kle_to_lame_arrays(
    kle_params_1d: np.ndarray,
    field_maps,
    subdomain_node_indices: List[np.ndarray],
    subdomain_elements: Dict[str, np.ndarray],
    subdomain_names: List[str],
    poisson_ratios: Dict[str, float],
    scalar_skfem_basis,
    num_kle_terms: int,
    bkd,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map concatenated KLE coefficients to per-element Lame parameter arrays.

    Field maps produce Young's modulus values at subdomain mesh nodes.
    These are scattered into a global nodal array and interpolated to
    quadrature points via ``scalar_skfem_basis.interpolate()``, then
    converted to Lame parameters using per-subdomain Poisson ratios.

    Parameters
    ----------
    kle_params_1d : np.ndarray
        Concatenated KLE coefficients. Shape: (n_subdomains * num_kle_terms,)
    field_maps : list
        Per-subdomain KLE field maps (values at subdomain nodes).
    subdomain_node_indices : list of np.ndarray
        Global node indices for each subdomain.
    subdomain_elements : dict
        Name to element index arrays.
    subdomain_names : list of str
        Ordered subdomain names.
    poisson_ratios : dict
        Name to Poisson ratio per subdomain.
    scalar_skfem_basis : skfem CellBasis
        Scalar FEM basis for interpolation from nodes to quad points.
    num_kle_terms : int
        KLE terms per subdomain.
    bkd : Backend

    Returns
    -------
    lam_arr : np.ndarray, shape (nelems, nquad)
    mu_arr : np.ndarray, shape (nelems, nquad)
    """
    nnodes = scalar_skfem_basis.mesh.p.shape[1]
    E_nodal = np.zeros(nnodes)

    offset = 0
    for i, name in enumerate(subdomain_names):
        xi = kle_params_1d[offset:offset + num_kle_terms]
        E_sub = bkd.to_numpy(field_maps[i](bkd.asarray(xi)))
        E_nodal[subdomain_node_indices[i]] = E_sub
        offset += num_kle_terms

    # Interpolate nodal E to quadrature points: (nelems, nquad)
    E_at_quads = scalar_skfem_basis.interpolate(E_nodal).value

    nelems = scalar_skfem_basis.mesh.nelements
    nquad = scalar_skfem_basis.dx.shape[1]
    lam_arr = np.zeros((nelems, nquad))
    mu_arr = np.zeros((nelems, nquad))

    for name in subdomain_names:
        elem_idx = subdomain_elements[name]
        nu = poisson_ratios[name]
        E_sub = E_at_quads[elem_idx, :]
        mu_arr[elem_idx, :] = E_sub / (2.0 * (1.0 + nu))
        lam_arr[elem_idx, :] = E_sub * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    return lam_arr, mu_arr


# =========================================================================
# 2D forward model (linear and hyperelastic)
# =========================================================================


class CantileverBeam2DForwardModel(Generic[Array]):
    """Forward model: KLE coefficients -> tip displacement for 2D beam.

    Wraps CompositeLinearElasticity or CompositeHyperelasticityPhysics
    with per-subdomain KLE field maps for Young's modulus.
    """

    def __init__(
        self,
        physics,
        solver,
        field_maps,
        subdomain_node_indices: List[np.ndarray],
        subdomain_elements: Dict[str, np.ndarray],
        subdomain_names: List[str],
        poisson_ratios: Dict[str, float],
        num_kle_terms: int,
        scalar_skfem_basis,
        tip_dof_index: int,
        bkd: Backend[Array],
    ):
        self._physics = physics
        self._solver = solver
        self._field_maps = field_maps
        self._subdomain_node_indices = subdomain_node_indices
        self._subdomain_elements = subdomain_elements
        self._subdomain_names = subdomain_names
        self._poisson_ratios = poisson_ratios
        self._num_kle_terms = num_kle_terms
        self._scalar_skfem_basis = scalar_skfem_basis
        self._tip_dof_index = tip_dof_index
        self._bkd = bkd
        self._nvars = len(subdomain_names) * num_kle_terms

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate: KLE coefficients -> tip displacement.

        Parameters
        ----------
        samples : Array
            KLE coefficients. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Tip displacement. Shape: (1, nsamples)
        """
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = np.zeros((1, nsamples))

        for ii in range(nsamples):
            xi = samples_np[:, ii]
            lam_arr, mu_arr = _kle_to_lame_arrays(
                xi, self._field_maps,
                self._subdomain_node_indices,
                self._subdomain_elements,
                self._subdomain_names,
                self._poisson_ratios,
                self._scalar_skfem_basis,
                self._num_kle_terms,
                bkd,
            )
            self._physics.set_lame_parameters(lam_arr, mu_arr)
            init = bkd.asarray(np.zeros(self._physics.nstates()))
            result = self._solver.solve(init)
            sol_np = bkd.to_numpy(result.solution)
            results[0, ii] = sol_np[self._tip_dof_index]

        return bkd.asarray(results)


# =========================================================================
# 1D forward model (Euler-Bernoulli beam)
# =========================================================================


class CantileverBeam1DForwardModel(Generic[Array]):
    """Forward model: KLE coefficients -> tip displacement for 1D beam.

    Wraps EulerBernoulliBeamFEM with a KLE field map for bending stiffness
    EI(x). The KLE parameterizes log(EI) on the 1D mesh.
    """

    def __init__(
        self,
        nx: int,
        length: float,
        EI_mean: float,
        load_func: Callable,
        field_map,
        bkd: Backend[Array],
    ):
        self._nx = nx
        self._length = length
        self._EI_mean = EI_mean
        self._load_func = load_func
        self._field_map = field_map
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._field_map.nvars()

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate: KLE coefficients -> tip displacement.

        Parameters
        ----------
        samples : Array
            KLE coefficients. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Tip displacement. Shape: (1, nsamples)
        """
        from pyapprox.typing.pde.galerkin.physics.euler_bernoulli import (
            EulerBernoulliBeamFEM,
        )
        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = np.zeros((1, nsamples))

        for ii in range(nsamples):
            xi = samples_np[:, ii]
            # EI field at element midpoints
            EI_field = bkd.to_numpy(
                self._field_map(bkd.asarray(xi))
            )
            # Use mean EI for now (1-term KLE gives constant)
            EI_avg = float(np.mean(EI_field))

            beam = EulerBernoulliBeamFEM(
                nx=self._nx,
                length=self._length,
                EI=EI_avg,
                load_func=self._load_func,
                bkd=bkd,
            )
            results[0, ii] = beam.tip_deflection()

        return bkd.asarray(results)


# =========================================================================
# Factory functions
# =========================================================================


def cantilever_beam_1d(
    bkd: Backend[Array],
    nx: int = 40,
    length: float = 100.0,
    EI_mean: float = 1e6,
    q0: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> PDEBenchmarkWrapper:
    """Create a 1D Euler-Bernoulli cantilever beam benchmark.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    nx : int
        Number of beam elements.
    length : float
        Beam length.
    EI_mean : float
        Mean bending stiffness.
    q0 : float
        Load magnitude for linearly increasing load q(x) = q0*x/L.
    num_kle_terms : int
        Number of KLE terms for EI(x).
    sigma : float
        Standard deviation of log(EI) field.
    correlation_length : float
        Correlation length for KLE kernel.
    """
    # Create KLE field map on element midpoints
    midpoints = np.linspace(
        length / (2 * nx), length - length / (2 * nx), nx,
    )
    mesh_coords = bkd.asarray(midpoints[np.newaxis, :])  # (1, nx)
    # Normalize to [0, 1] for KLE
    mesh_coords_norm = mesh_coords / length
    mean_log = bkd.full((nx,), float(np.log(EI_mean)))

    elem_lengths = bkd.full((nx,), 1.0 / nx)  # normalized element lengths
    field_map = create_lognormal_kle_field_map(
        mesh_coords=mesh_coords_norm,
        mean_log_field=mean_log,
        bkd=bkd,
        num_kle_terms=num_kle_terms,
        sigma=sigma,
        correlation_length=correlation_length,
        quad_weights=elem_lengths,
    )

    load_func = lambda x: q0 * x / length  # noqa: E731

    fwd = CantileverBeam1DForwardModel(
        nx=nx, length=length, EI_mean=EI_mean,
        load_func=load_func, field_map=field_map, bkd=bkd,
    )

    prior = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(num_kle_terms)],
        bkd,
    )
    bounds = bkd.array([[-4.0, 4.0]] * num_kle_terms)
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    inner = BenchmarkWithPrior(
        _name=f"cantilever_beam_1d_tip_displacement",
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            f"1D Euler-Bernoulli cantilever beam, nx={nx}, "
            f"{num_kle_terms} KLE terms for EI(x)"
        ),
    )
    return PDEBenchmarkWrapper(inner, estimated_cost=1e-03)


def _find_tip_dof(basis, length, height, bkd):
    """Find the vertical displacement DOF closest to the beam tip.

    The tip is at (x=length, y=height/2). For interleaved DOFs
    [ux_0, uy_0, ux_1, uy_1, ...], the y-displacement DOF is 2*node+1.
    """
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    ndim = basis.ncomponents()
    n_dofs = basis.ndofs()
    tip_x, tip_y = length, height / 2.0

    best_dof = -1
    best_dist = np.inf
    for i in range(n_dofs):
        comp = i % ndim
        if comp != 1:  # only y-displacement
            continue
        node_idx = i // ndim
        x_coord = dof_coords[0, i]
        y_coord = dof_coords[1, i]
        dist = (x_coord - tip_x)**2 + (y_coord - tip_y)**2
        if dist < best_dist:
            best_dist = dist
            best_dof = i
    return best_dof


def cantilever_beam_2d_linear(
    bkd: Backend[Array],
    mesh_path: str = _DEFAULT_MESH_PATH,
    length: float = 100.0,
    height: float = 30.0,
    E_mean: float = 1e4,
    poisson_ratio: float = 0.3,
    q0: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> PDEBenchmarkWrapper:
    """Create a 2D linear elastic cantilever beam benchmark.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_path : str
        Path to JSON mesh file with boundaries and subdomains.
    length : float
        Beam length (after rescaling).
    height : float
        Beam height (after rescaling).
    E_mean : float
        Mean Young's modulus for all subdomains.
    poisson_ratio : float
        Poisson ratio (same for all subdomains).
    q0 : float
        Load magnitude for linearly increasing traction on top surface.
    num_kle_terms : int
        Number of KLE terms per subdomain.
    sigma : float
        Standard deviation of log(E) field.
    correlation_length : float
        Correlation length for KLE kernel (in normalized coordinates).
    """
    from skfem import Basis as SkfemBasis
    from pyapprox.typing.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.typing.pde.galerkin.physics import CompositeLinearElasticity
    from pyapprox.typing.pde.galerkin.boundary.implementations import (
        DirichletBC, NeumannBC,
    )
    from pyapprox.typing.pde.galerkin.solvers.steady_state import (
        SteadyStateSolver,
    )

    mesh = UnstructuredMesh2D(mesh_path, bkd, rescale_origin=(0.0, 0.0))
    basis = VectorLagrangeBasis(mesh, degree=1)
    skfem_mesh = mesh.skfem_mesh()
    subdomain_names = mesh.subdomain_names()
    subdomain_elements = {
        name: mesh.subdomain_elements(name) for name in subdomain_names
    }

    # Scalar basis for quadrature point access (KLE)
    scalar_element = basis.scalar_basis().skfem_basis().elem
    scalar_skfem_basis = SkfemBasis(skfem_mesh, scalar_element)

    # Poisson ratio per subdomain (same for all)
    poisson_ratios = {name: poisson_ratio for name in subdomain_names}

    # KLE field maps per subdomain (at mesh nodes)
    mean_log_E = float(np.log(E_mean))
    field_maps, subdomain_node_indices = _create_subdomain_kle_field_maps(
        skfem_mesh, subdomain_elements, subdomain_names, bkd,
        num_kle_terms, sigma, correlation_length, mean_log_E,
    )

    # Boundary conditions: clamped left, traction on top
    def zero_dirichlet(coords, time=0.0):
        return np.zeros(coords.shape[1])

    bc_left = DirichletBC(basis, "left_edge", zero_dirichlet, bkd)

    def top_traction(coords, time=0.0):
        x = coords[0]
        npts = coords.shape[1]
        traction = np.zeros((2, npts))
        traction[1, :] = -q0 * x / length  # downward, linearly increasing
        return traction

    bc_top = NeumannBC(basis, "top_edge", top_traction, bkd)

    # Initial material: uniform E_mean
    material_map = {name: (E_mean, poisson_ratio) for name in subdomain_names}

    physics = CompositeLinearElasticity(
        basis=basis,
        material_map=material_map,
        element_materials=subdomain_elements,
        bkd=bkd,
        boundary_conditions=[bc_left, bc_top],
    )

    solver = SteadyStateSolver(physics, tol=1e-10, max_iter=1)
    tip_dof = _find_tip_dof(basis, length, height, bkd)

    nvars = len(subdomain_names) * num_kle_terms
    fwd = CantileverBeam2DForwardModel(
        physics=physics,
        solver=solver,
        field_maps=field_maps,
        subdomain_node_indices=subdomain_node_indices,
        subdomain_elements=subdomain_elements,
        subdomain_names=subdomain_names,
        poisson_ratios=poisson_ratios,
        num_kle_terms=num_kle_terms,
        scalar_skfem_basis=scalar_skfem_basis,
        tip_dof_index=tip_dof,
        bkd=bkd,
    )

    prior = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
        bkd,
    )
    bounds = bkd.array([[-4.0, 4.0]] * nvars)
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    inner = BenchmarkWithPrior(
        _name="cantilever_beam_2d_linear_tip_displacement",
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            f"2D linear elastic cantilever beam, "
            f"{num_kle_terms} KLE terms per subdomain, "
            f"{len(subdomain_names)} subdomains"
        ),
    )
    return PDEBenchmarkWrapper(inner, estimated_cost=1.0)


def cantilever_beam_2d_neohookean(
    bkd: Backend[Array],
    mesh_path: str = _DEFAULT_MESH_PATH,
    length: float = 100.0,
    height: float = 30.0,
    E_mean: float = 1e4,
    poisson_ratio: float = 0.3,
    q0: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> PDEBenchmarkWrapper:
    """Create a 2D Neo-Hookean cantilever beam benchmark.

    Same setup as the linear benchmark but with nonlinear hyperelastic
    constitutive model (Neo-Hookean).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_path : str
        Path to JSON mesh file with boundaries and subdomains.
    length : float
        Beam length (after rescaling).
    height : float
        Beam height (after rescaling).
    E_mean : float
        Mean Young's modulus for all subdomains.
    poisson_ratio : float
        Poisson ratio (same for all subdomains).
    q0 : float
        Load magnitude for linearly increasing traction on top surface.
    num_kle_terms : int
        Number of KLE terms per subdomain.
    sigma : float
        Standard deviation of log(E) field.
    correlation_length : float
        Correlation length for KLE kernel (in normalized coordinates).
    """
    from skfem import Basis as SkfemBasis
    from pyapprox.typing.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.typing.pde.galerkin.physics import (
        CompositeHyperelasticityPhysics,
    )
    from pyapprox.typing.pde.galerkin.boundary.implementations import (
        DirichletBC, NeumannBC,
    )
    from pyapprox.typing.pde.galerkin.solvers.steady_state import (
        SteadyStateSolver,
    )

    mesh = UnstructuredMesh2D(mesh_path, bkd, rescale_origin=(0.0, 0.0))
    basis = VectorLagrangeBasis(mesh, degree=1)
    skfem_mesh = mesh.skfem_mesh()
    subdomain_names = mesh.subdomain_names()
    subdomain_elements = {
        name: mesh.subdomain_elements(name) for name in subdomain_names
    }

    # Scalar basis for quadrature point access (KLE)
    scalar_element = basis.scalar_basis().skfem_basis().elem
    scalar_skfem_basis = SkfemBasis(skfem_mesh, scalar_element)

    poisson_ratios = {name: poisson_ratio for name in subdomain_names}

    mean_log_E = float(np.log(E_mean))
    field_maps, subdomain_node_indices = _create_subdomain_kle_field_maps(
        skfem_mesh, subdomain_elements, subdomain_names, bkd,
        num_kle_terms, sigma, correlation_length, mean_log_E,
    )

    def zero_dirichlet(coords, time=0.0):
        return np.zeros(coords.shape[1])

    bc_left = DirichletBC(basis, "left_edge", zero_dirichlet, bkd)

    def top_traction(coords, time=0.0):
        x = coords[0]
        npts = coords.shape[1]
        traction = np.zeros((2, npts))
        traction[1, :] = -q0 * x / length
        return traction

    bc_top = NeumannBC(basis, "top_edge", top_traction, bkd)

    material_map = {name: (E_mean, poisson_ratio) for name in subdomain_names}

    physics = CompositeHyperelasticityPhysics(
        basis=basis,
        material_map=material_map,
        element_materials=subdomain_elements,
        bkd=bkd,
        boundary_conditions=[bc_left, bc_top],
    )

    solver = SteadyStateSolver(
        physics, tol=1e-10, max_iter=50, line_search=True,
    )
    tip_dof = _find_tip_dof(basis, length, height, bkd)

    nvars = len(subdomain_names) * num_kle_terms
    fwd = CantileverBeam2DForwardModel(
        physics=physics,
        solver=solver,
        field_maps=field_maps,
        subdomain_node_indices=subdomain_node_indices,
        subdomain_elements=subdomain_elements,
        subdomain_names=subdomain_names,
        poisson_ratios=poisson_ratios,
        num_kle_terms=num_kle_terms,
        scalar_skfem_basis=scalar_skfem_basis,
        tip_dof_index=tip_dof,
        bkd=bkd,
    )

    prior = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
        bkd,
    )
    bounds = bkd.array([[-4.0, 4.0]] * nvars)
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    inner = BenchmarkWithPrior(
        _name="cantilever_beam_2d_neohookean_tip_displacement",
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            f"2D Neo-Hookean cantilever beam, "
            f"{num_kle_terms} KLE terms per subdomain, "
            f"{len(subdomain_names)} subdomains"
        ),
    )
    return PDEBenchmarkWrapper(inner, estimated_cost=5.0)


# =========================================================================
# SPDE-based variants (Matern KLE via sparse eigensolve)
# =========================================================================


def cantilever_beam_1d_spde(
    bkd: Backend[Array],
    nx: int = 40,
    length: float = 100.0,
    EI_mean: float = 1e6,
    q0: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> PDEBenchmarkWrapper:
    """Create a 1D Euler-Bernoulli cantilever beam benchmark (SPDE KLE).

    Same as :func:`cantilever_beam_1d` but uses the sparse SPDE-based
    Matern KLE instead of the dense Nystrom squared-exponential KLE.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    nx : int
        Number of beam elements.
    length : float
        Beam length.
    EI_mean : float
        Mean bending stiffness.
    q0 : float
        Load magnitude for linearly increasing load q(x) = q0*x/L.
    num_kle_terms : int
        Number of KLE terms for EI(x).
    sigma : float
        Standard deviation of log(EI) field.
    correlation_length : float
        Correlation length for the SPDE Matern field.
    """
    from pyapprox.typing.pde.galerkin.mesh.structured import StructuredMesh1D

    mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)

    gamma = correlation_length ** 2
    delta = 1.0
    mean_log = bkd.full((basis.ndofs(),), float(np.log(EI_mean)))

    field_map = create_spde_lognormal_kle_field_map(
        basis, mean_log, bkd,
        n_modes=num_kle_terms, gamma=gamma, delta=delta, sigma=sigma,
    )

    load_func = lambda x: q0 * x / length  # noqa: E731

    fwd = CantileverBeam1DForwardModel(
        nx=nx, length=length, EI_mean=EI_mean,
        load_func=load_func, field_map=field_map, bkd=bkd,
    )

    prior = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(num_kle_terms)],
        bkd,
    )
    bounds = bkd.array([[-4.0, 4.0]] * num_kle_terms)
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    inner = BenchmarkWithPrior(
        _name="cantilever_beam_1d_spde_tip_displacement",
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            f"1D Euler-Bernoulli cantilever beam (SPDE Matern KLE), "
            f"nx={nx}, {num_kle_terms} KLE terms for EI(x)"
        ),
    )
    return PDEBenchmarkWrapper(inner, estimated_cost=1e-03)


def cantilever_beam_2d_linear_spde(
    bkd: Backend[Array],
    mesh_path: str = _DEFAULT_MESH_PATH,
    length: float = 100.0,
    height: float = 30.0,
    E_mean: float = 1e4,
    poisson_ratio: float = 0.3,
    q0: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> PDEBenchmarkWrapper:
    """Create a 2D linear elastic cantilever beam benchmark (SPDE KLE).

    Same as :func:`cantilever_beam_2d_linear` but uses the sparse
    SPDE-based Matern KLE instead of the dense Nystrom
    squared-exponential KLE.  Per-subdomain KLEs are built on submeshes
    extracted via ``skfem_mesh.restrict()``.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_path : str
        Path to JSON mesh file with boundaries and subdomains.
    length : float
        Beam length (after rescaling).
    height : float
        Beam height (after rescaling).
    E_mean : float
        Mean Young's modulus for all subdomains.
    poisson_ratio : float
        Poisson ratio (same for all subdomains).
    q0 : float
        Load magnitude for linearly increasing traction on top surface.
    num_kle_terms : int
        Number of KLE terms per subdomain.
    sigma : float
        Standard deviation of log(E) field.
    correlation_length : float
        Correlation length for the SPDE Matern field.
    """
    from skfem import Basis as SkfemBasis
    from pyapprox.typing.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.typing.pde.galerkin.physics import CompositeLinearElasticity
    from pyapprox.typing.pde.galerkin.boundary.implementations import (
        DirichletBC, NeumannBC,
    )
    from pyapprox.typing.pde.galerkin.solvers.steady_state import (
        SteadyStateSolver,
    )

    mesh = UnstructuredMesh2D(mesh_path, bkd, rescale_origin=(0.0, 0.0))
    basis = VectorLagrangeBasis(mesh, degree=1)
    skfem_mesh = mesh.skfem_mesh()
    subdomain_names = mesh.subdomain_names()
    subdomain_elements = {
        name: mesh.subdomain_elements(name) for name in subdomain_names
    }

    scalar_element = basis.scalar_basis().skfem_basis().elem
    scalar_skfem_basis = SkfemBasis(skfem_mesh, scalar_element)

    poisson_ratios = {name: poisson_ratio for name in subdomain_names}

    mean_log_E = float(np.log(E_mean))
    field_maps, subdomain_node_indices = _create_subdomain_spde_kle_field_maps(
        skfem_mesh, subdomain_elements, subdomain_names, bkd,
        num_kle_terms, sigma, correlation_length, mean_log_E,
    )

    def zero_dirichlet(coords, time=0.0):
        return np.zeros(coords.shape[1])

    bc_left = DirichletBC(basis, "left_edge", zero_dirichlet, bkd)

    def top_traction(coords, time=0.0):
        x = coords[0]
        npts = coords.shape[1]
        traction = np.zeros((2, npts))
        traction[1, :] = -q0 * x / length
        return traction

    bc_top = NeumannBC(basis, "top_edge", top_traction, bkd)

    material_map = {name: (E_mean, poisson_ratio) for name in subdomain_names}

    physics = CompositeLinearElasticity(
        basis=basis,
        material_map=material_map,
        element_materials=subdomain_elements,
        bkd=bkd,
        boundary_conditions=[bc_left, bc_top],
    )

    solver = SteadyStateSolver(physics, tol=1e-10, max_iter=1)
    tip_dof = _find_tip_dof(basis, length, height, bkd)

    nvars = len(subdomain_names) * num_kle_terms
    fwd = CantileverBeam2DForwardModel(
        physics=physics,
        solver=solver,
        field_maps=field_maps,
        subdomain_node_indices=subdomain_node_indices,
        subdomain_elements=subdomain_elements,
        subdomain_names=subdomain_names,
        poisson_ratios=poisson_ratios,
        num_kle_terms=num_kle_terms,
        scalar_skfem_basis=scalar_skfem_basis,
        tip_dof_index=tip_dof,
        bkd=bkd,
    )

    prior = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
        bkd,
    )
    bounds = bkd.array([[-4.0, 4.0]] * nvars)
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    inner = BenchmarkWithPrior(
        _name="cantilever_beam_2d_linear_spde_tip_displacement",
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            f"2D linear elastic cantilever beam (SPDE Matern KLE), "
            f"{num_kle_terms} KLE terms per subdomain, "
            f"{len(subdomain_names)} subdomains"
        ),
    )
    return PDEBenchmarkWrapper(inner, estimated_cost=1.0)


def cantilever_beam_2d_neohookean_spde(
    bkd: Backend[Array],
    mesh_path: str = _DEFAULT_MESH_PATH,
    length: float = 100.0,
    height: float = 30.0,
    E_mean: float = 1e4,
    poisson_ratio: float = 0.3,
    q0: float = 1.0,
    num_kle_terms: int = 2,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> PDEBenchmarkWrapper:
    """Create a 2D Neo-Hookean cantilever beam benchmark (SPDE KLE).

    Same as :func:`cantilever_beam_2d_neohookean` but uses the sparse
    SPDE-based Matern KLE instead of the dense Nystrom
    squared-exponential KLE.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_path : str
        Path to JSON mesh file with boundaries and subdomains.
    length : float
        Beam length (after rescaling).
    height : float
        Beam height (after rescaling).
    E_mean : float
        Mean Young's modulus for all subdomains.
    poisson_ratio : float
        Poisson ratio (same for all subdomains).
    q0 : float
        Load magnitude for linearly increasing traction on top surface.
    num_kle_terms : int
        Number of KLE terms per subdomain.
    sigma : float
        Standard deviation of log(E) field.
    correlation_length : float
        Correlation length for the SPDE Matern field.
    """
    from skfem import Basis as SkfemBasis
    from pyapprox.typing.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.typing.pde.galerkin.physics import (
        CompositeHyperelasticityPhysics,
    )
    from pyapprox.typing.pde.galerkin.boundary.implementations import (
        DirichletBC, NeumannBC,
    )
    from pyapprox.typing.pde.galerkin.solvers.steady_state import (
        SteadyStateSolver,
    )

    mesh = UnstructuredMesh2D(mesh_path, bkd, rescale_origin=(0.0, 0.0))
    basis = VectorLagrangeBasis(mesh, degree=1)
    skfem_mesh = mesh.skfem_mesh()
    subdomain_names = mesh.subdomain_names()
    subdomain_elements = {
        name: mesh.subdomain_elements(name) for name in subdomain_names
    }

    scalar_element = basis.scalar_basis().skfem_basis().elem
    scalar_skfem_basis = SkfemBasis(skfem_mesh, scalar_element)

    poisson_ratios = {name: poisson_ratio for name in subdomain_names}

    mean_log_E = float(np.log(E_mean))
    field_maps, subdomain_node_indices = _create_subdomain_spde_kle_field_maps(
        skfem_mesh, subdomain_elements, subdomain_names, bkd,
        num_kle_terms, sigma, correlation_length, mean_log_E,
    )

    def zero_dirichlet(coords, time=0.0):
        return np.zeros(coords.shape[1])

    bc_left = DirichletBC(basis, "left_edge", zero_dirichlet, bkd)

    def top_traction(coords, time=0.0):
        x = coords[0]
        npts = coords.shape[1]
        traction = np.zeros((2, npts))
        traction[1, :] = -q0 * x / length
        return traction

    bc_top = NeumannBC(basis, "top_edge", top_traction, bkd)

    material_map = {name: (E_mean, poisson_ratio) for name in subdomain_names}

    physics = CompositeHyperelasticityPhysics(
        basis=basis,
        material_map=material_map,
        element_materials=subdomain_elements,
        bkd=bkd,
        boundary_conditions=[bc_left, bc_top],
    )

    solver = SteadyStateSolver(
        physics, tol=1e-10, max_iter=50, line_search=True,
    )
    tip_dof = _find_tip_dof(basis, length, height, bkd)

    nvars = len(subdomain_names) * num_kle_terms
    fwd = CantileverBeam2DForwardModel(
        physics=physics,
        solver=solver,
        field_maps=field_maps,
        subdomain_node_indices=subdomain_node_indices,
        subdomain_elements=subdomain_elements,
        subdomain_names=subdomain_names,
        poisson_ratios=poisson_ratios,
        num_kle_terms=num_kle_terms,
        scalar_skfem_basis=scalar_skfem_basis,
        tip_dof_index=tip_dof,
        bkd=bkd,
    )

    prior = IndependentJoint(
        [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nvars)],
        bkd,
    )
    bounds = bkd.array([[-4.0, 4.0]] * nvars)
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    inner = BenchmarkWithPrior(
        _name="cantilever_beam_2d_neohookean_spde_tip_displacement",
        _function=fwd,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            f"2D Neo-Hookean cantilever beam (SPDE Matern KLE), "
            f"{num_kle_terms} KLE terms per subdomain, "
            f"{len(subdomain_names)} subdomains"
        ),
    )
    return PDEBenchmarkWrapper(inner, estimated_cost=5.0)


# =========================================================================
# Registry
# =========================================================================


@BenchmarkRegistry.register(
    "cantilever_beam_1d",
    category="pde",
    description="1D Euler-Bernoulli cantilever beam with KLE bending stiffness",
)
def _cantilever_beam_1d_factory(bkd: Backend[Array]) -> PDEBenchmarkWrapper:
    return cantilever_beam_1d(bkd)


@BenchmarkRegistry.register(
    "cantilever_beam_2d_linear",
    category="pde",
    description=(
        "2D linear elastic cantilever beam with per-subdomain KLE "
        "Young's modulus"
    ),
)
def _cantilever_beam_2d_linear_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return cantilever_beam_2d_linear(bkd)


@BenchmarkRegistry.register(
    "cantilever_beam_2d_neohookean",
    category="pde",
    description=(
        "2D Neo-Hookean cantilever beam with per-subdomain KLE "
        "Young's modulus"
    ),
)
def _cantilever_beam_2d_neohookean_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return cantilever_beam_2d_neohookean(bkd)


@BenchmarkRegistry.register(
    "cantilever_beam_1d_spde",
    category="pde",
    description=(
        "1D Euler-Bernoulli cantilever beam with SPDE Matern KLE "
        "bending stiffness"
    ),
)
def _cantilever_beam_1d_spde_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return cantilever_beam_1d_spde(bkd)


@BenchmarkRegistry.register(
    "cantilever_beam_2d_linear_spde",
    category="pde",
    description=(
        "2D linear elastic cantilever beam with per-subdomain SPDE "
        "Matern KLE Young's modulus"
    ),
)
def _cantilever_beam_2d_linear_spde_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return cantilever_beam_2d_linear_spde(bkd)


@BenchmarkRegistry.register(
    "cantilever_beam_2d_neohookean_spde",
    category="pde",
    description=(
        "2D Neo-Hookean cantilever beam with per-subdomain SPDE "
        "Matern KLE Young's modulus"
    ),
)
def _cantilever_beam_2d_neohookean_spde_factory(
    bkd: Backend[Array],
) -> PDEBenchmarkWrapper:
    return cantilever_beam_2d_neohookean_spde(bkd)
