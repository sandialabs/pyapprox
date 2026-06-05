"""Shared-field cantilever beam models for multi-fidelity ensembles.

Provides ``SharedFieldBeamModel``, a ``FunctionProtocol``-compatible
forward model that consumes ``(xi_1..xi_d, v)`` — KLE coefficients for
a spatially varying Young's modulus field plus a random wind speed —
and returns ``(tip_deflection, integrated_von_mises_stress)``.

The load is parameterized as dynamic pressure: ``q0 = v^2``, where
``v`` is the wind speed.  This nonlinear input transformation ensures
all models (including linear elasticity) exhibit meaningful nonlinearity
in the input-to-output map.

Unlike the per-subdomain builders in ``cantilever_beam.py``, each model
here uses a **single KLE on the full mesh** (no subdomain splitting).
Different meshes get their own ``MeshKLE`` instance with identical kernel
parameters, so the same ``xi`` produces correlated E fields on different
grids.  The factory ``build_shared_field_beam`` assembles physics, solver,
KLE, BCs, and QoI extraction into a single callable.
"""

import os
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
)

if TYPE_CHECKING:
    import skfem
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )

import numpy as np
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.util.backends.protocols import Array, Backend

_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "data",
)

MESH_PATHS = {
    0.5: os.path.normpath(
        os.path.join(_DATA_DIR, "cantilever_beam_2d_with_holes_h_0_5.json")
    ),
    1: os.path.normpath(
        os.path.join(_DATA_DIR, "cantilever_beam_2d_with_holes_h_1.json")
    ),
    2: os.path.normpath(
        os.path.join(_DATA_DIR, "cantilever_beam_2d_with_holes_h_2.json")
    ),
    4: os.path.normpath(
        os.path.join(_DATA_DIR, "cantilever_beam_2d_with_holes_h_4.json")
    ),
}


def _lumped_mass_weights(
    skfem_mesh: "skfem.Mesh",
) -> np.ndarray:
    """Compute lumped mass quadrature weights by distributing element areas."""
    nnodes = skfem_mesh.nvertices
    nverts_per_elem = skfem_mesh.t.shape[0]
    weights = np.zeros(nnodes)
    for e in range(skfem_mesh.nelements):
        verts = skfem_mesh.p[:, skfem_mesh.t[:, e]]
        x, y = verts[0], verts[1]
        area = 0.5 * abs(
            sum(
                x[i] * y[(i + 1) % nverts_per_elem]
                - x[(i + 1) % nverts_per_elem] * y[i]
                for i in range(nverts_per_elem)
            )
        )
        for n in skfem_mesh.t[:, e]:
            weights[int(n)] += area / nverts_per_elem
    return weights


def _build_kle_field_map(
    skfem_mesh: "skfem.Mesh",
    bkd: Backend[Array],
    num_kle_terms: int,
    sigma: float,
    correlation_length: float,
    E_mean: float,
) -> TransformedFieldMap[Array]:
    """Build a lognormal KLE field map on the full mesh (no subdomains)."""
    ndim = skfem_mesh.p.shape[0]
    lenscale = bkd.full((ndim,), correlation_length)
    kernel = SquaredExponentialKernel(lenscale, (0.01, 10.0), ndim, bkd)

    coords = bkd.asarray(skfem_mesh.p.astype(np.float64))
    quad_weights = bkd.asarray(_lumped_mass_weights(skfem_mesh))
    mean_log_E = float(np.log(E_mean))

    mesh_kle = MeshKLE(
        coords,
        kernel,
        sigma=sigma,
        mean_field=0.0,
        nterms=num_kle_terms,
        quad_weights=quad_weights,
        bkd=bkd,
    )

    nnodes = skfem_mesh.nvertices
    mean_log = bkd.full((nnodes,), mean_log_E)
    inner = MeshKLEFieldMap(bkd, mean_log, mesh_kle.weighted_eigenvectors())
    return TransformedFieldMap(
        inner,
        transform=lambda x: bkd.exp(x),
        transform_deriv=lambda x: bkd.exp(x),
        bkd=bkd,
        transform_deriv2=lambda x: bkd.exp(x),
    )


def _find_tip_dof(
    basis: "VectorLagrangeBasis[Array]",
    length: float,
    height: float,
    bkd: Backend[Array],
) -> int:
    """Find the y-displacement DOF closest to the beam tip (L, H/2)."""
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    ndim = basis.ncomponents()
    n_dofs = basis.ndofs()
    best_dof = -1
    best_dist = np.inf
    for i in range(n_dofs):
        if i % ndim != 1:
            continue
        dist = (dof_coords[0, i] - length) ** 2 + (
            dof_coords[1, i] - height / 2.0
        ) ** 2
        if dist < best_dist:
            best_dist = dist
            best_dof = i
    return best_dof


class SharedFieldBeamModel(Generic[Array]):
    """Forward model: (xi, v) -> (tip_deflection, integrated_VM_stress).

    Consumes ``(num_kle_terms + 1)`` input variables: ``xi`` for the
    shared KLE field and ``v`` for the wind speed.  The traction load
    is ``q0 = v^2`` (dynamic pressure).  Each evaluation sets the E
    field from the KLE, computes the load from the wind speed, and
    solves the FEM problem.
    """

    def __init__(
        self,
        physics: object,
        solver: object,
        neumann_bc: object,
        scalar_skfem_basis: "skfem.CellBasis",
        kle_field_map: TransformedFieldMap[Array],
        poisson_ratio: float,
        tip_dof_index: int,
        length: float,
        bkd: Backend[Array],
        num_kle_terms: int,
    ) -> None:
        self._physics = physics
        self._solver = solver
        self._neumann_bc = neumann_bc
        self._scalar_skfem_basis = scalar_skfem_basis
        self._kle_field_map = kle_field_map
        self._poisson_ratio = poisson_ratio
        self._tip_dof_index = tip_dof_index
        self._length = length
        self._bkd = bkd
        self._num_kle_terms = num_kle_terms

        skfem_mesh = scalar_skfem_basis.mesh
        self._coordx = skfem_mesh.p[0]
        self._coordy = skfem_mesh.p[1]
        self._connectivity = skfem_mesh.t.T

        nelems = skfem_mesh.nelements
        areas = np.empty(nelems)
        for ie in range(nelems):
            nodes = self._connectivity[ie]
            xe, ye = self._coordx[nodes], self._coordy[nodes]
            n = len(nodes)
            a = 0.0
            for j in range(n):
                a += xe[j] * ye[(j + 1) % n] - xe[(j + 1) % n] * ye[j]
            areas[ie] = abs(a) / 2.0
        self._element_areas = areas

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._num_kle_terms + 1

    def nqoi(self) -> int:
        return 2

    def __call__(self, samples: Array) -> Array:
        from pyapprox.pde.galerkin.postprocessing import von_mises_stress_2d

        bkd = self._bkd
        samples_np = bkd.to_numpy(samples)
        nsamples = samples_np.shape[1]
        results = np.zeros((2, nsamples))
        d = self._num_kle_terms
        nu = self._poisson_ratio
        L = self._length

        for ii in range(nsamples):
            xi = samples_np[:d, ii]
            v = float(samples_np[d, ii])
            q0 = v ** 2

            E_nodal = bkd.to_numpy(
                self._kle_field_map(bkd.asarray(xi))
            )

            E_at_quads = self._scalar_skfem_basis.interpolate(
                E_nodal
            ).value
            mu_arr = E_at_quads / (2.0 * (1.0 + nu))
            lam_arr = E_at_quads * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self._physics.set_lame_parameters(lam_arr, mu_arr)

            self._neumann_bc._flux_func = (
                lambda coords, time=0.0, _q0=q0, _L=L: np.column_stack([
                    np.zeros(coords.shape[1]),
                    -_q0 * coords[0] / _L,
                ]).T
            )

            init = bkd.asarray(np.zeros(self._physics.nstates()))
            result = self._solver.solve(init)
            sol_np = bkd.to_numpy(result.solution)

            results[0, ii] = sol_np[self._tip_dof_index]

            ux = sol_np[0::2]
            uy = sol_np[1::2]
            lam_elem = np.mean(lam_arr, axis=1)
            mu_elem = np.mean(mu_arr, axis=1)
            vm = von_mises_stress_2d(
                self._coordx,
                self._coordy,
                self._connectivity,
                ux,
                uy,
                lam_elem,
                mu_elem,
            )
            results[1, ii] = np.sum(vm * self._element_areas)

        return bkd.asarray(results)


def build_shared_field_beam(
    bkd: Backend[Array],
    mesh_path: str,
    physics_type: Literal["linear", "neohookean"],
    num_kle_terms: int = 5,
    length: float = 100.0,
    height: float = 30.0,
    E_mean: float = 1e4,
    poisson_ratio: float = 0.3,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> SharedFieldBeamModel[Array]:
    """Build a shared-field beam model for use in an ensemble.

    Unlike ``build_cantilever_beam_2d_linear``/``neohookean``, this
    factory uses a single KLE on the full mesh (no subdomain splitting)
    and leaves the load magnitude as an input parameter (not fixed).

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_path : str
        Path to JSON mesh file.
    physics_type : {"linear", "neohookean"}
        Constitutive model.
    num_kle_terms : int
        Number of KLE terms for the E field.
    length, height : float
        Beam dimensions (after rescaling).
    E_mean : float
        Mean Young's modulus.
    poisson_ratio : float
        Poisson ratio (constant).
    sigma : float
        Standard deviation of log(E) field.
    correlation_length : float
        Correlation length for KLE kernel.

    Returns
    -------
    SharedFieldBeamModel
        Callable model: ``(num_kle_terms+1, nsamples) -> (2, nsamples)``.
    """
    from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.pde.galerkin.boundary.implementations import (
        DirichletBC,
        NeumannBC,
    )
    from pyapprox.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
    from skfem import Basis as SkfemBasis

    mesh = UnstructuredMesh2D(mesh_path, bkd, rescale_origin=(0.0, 0.0))
    basis = VectorLagrangeBasis(mesh, degree=1)
    skfem_mesh = mesh.skfem_mesh()

    scalar_element = basis.scalar_basis().skfem_basis().elem
    scalar_skfem_basis = SkfemBasis(skfem_mesh, scalar_element)

    kle_field_map = _build_kle_field_map(
        skfem_mesh, bkd, num_kle_terms, sigma, correlation_length, E_mean,
    )

    def zero_dirichlet(coords: np.ndarray, time: float = 0.0) -> np.ndarray:
        return np.zeros(coords.shape[1])

    bc_left = DirichletBC(basis, "left_edge", zero_dirichlet, bkd)

    def top_traction(coords: np.ndarray, time: float = 0.0) -> np.ndarray:
        x = coords[0]
        npts = coords.shape[1]
        traction = np.zeros((2, npts))
        traction[1, :] = -1.0 * x / length
        return traction

    bc_top = NeumannBC(basis, "top_edge", top_traction, bkd)

    subdomain_names = mesh.subdomain_names()
    subdomain_elements = {
        name: mesh.subdomain_elements(name) for name in subdomain_names
    }
    material_map = {
        name: (E_mean, poisson_ratio) for name in subdomain_names
    }

    if physics_type == "linear":
        from pyapprox.pde.galerkin.physics import CompositeLinearElasticity

        physics = CompositeLinearElasticity(
            basis=basis,
            material_map=material_map,
            element_materials=subdomain_elements,
            bkd=bkd,
            boundary_conditions=[bc_left, bc_top],
        )
        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=1)
    elif physics_type == "neohookean":
        from pyapprox.pde.galerkin.physics import (
            CompositeHyperelasticityPhysics,
        )

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
    else:
        raise ValueError(
            f"Unknown physics_type '{physics_type}'; "
            "expected 'linear' or 'neohookean'"
        )

    tip_dof = _find_tip_dof(basis, length, height, bkd)

    return SharedFieldBeamModel(
        physics=physics,
        solver=solver,
        neumann_bc=bc_top,
        scalar_skfem_basis=scalar_skfem_basis,
        kle_field_map=kle_field_map,
        poisson_ratio=poisson_ratio,
        tip_dof_index=tip_dof,
        length=length,
        bkd=bkd,
        num_kle_terms=num_kle_terms,
    )
