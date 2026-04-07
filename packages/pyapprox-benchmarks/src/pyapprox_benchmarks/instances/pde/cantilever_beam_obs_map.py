"""FEM-based observation map for cantilever beam load identification.

Builds the design matrix A by solving two unit-load FEM problems
(constant traction, slope traction) and extracting y-displacements
at sensor locations. Wraps A as a FunctionFromCallable.

This is the analog of expdesign/benchmarks/functions/linear_gaussian.py
for a PDE-derived forward model.

Migration note: When pyapprox/benchmarks/ adopts the full
functions/problems/instances/ directory structure, this file moves to
pyapprox/benchmarks/functions/pde/cantilever_beam.py.
"""

from typing import Optional, Tuple

import numpy as np

# Reuse _find_dof from the main cantilever beam module
from pyapprox.benchmarks.instances.pde.cantilever_beam import (
    _DEFAULT_MESH_PATH,
    _find_dof,
)
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend


def build_cantilever_beam_design_matrix(
    bkd: Backend[Array],
    mesh_path: str = _DEFAULT_MESH_PATH,
    length: float = 100.0,
    height: float = 30.0,
    E_mean: float = 1e4,
    poisson_ratio: float = 0.3,
    sensor_xs: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Build design matrix A from FEM unit-load solutions.

    Solves two FEM problems (constant and slope traction) and extracts
    y-displacements at sensor locations to form the design matrix.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    mesh_path : str
        Path to JSON mesh file.
    length : float
        Beam length.
    height : float
        Beam height.
    E_mean : float
        Young's modulus.
    poisson_ratio : float
        Poisson ratio.
    sensor_xs : Array or None
        Sensor x-coordinates. Shape (nobs,).
        Default: 5 equally spaced in [length/5, length].

    Returns
    -------
    design_matrix : Array
        Design matrix A. Shape: (nobs, 2).
    sensor_xs : Array
        Sensor x-coordinates. Shape: (nobs,).
    """
    from typing import Callable

    from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.pde.galerkin.boundary.implementations import (
        DirichletBC,
        NeumannBC,
    )
    from pyapprox.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.pde.galerkin.physics import CompositeLinearElasticity
    from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver

    # ---- Build FEM ----
    mesh = UnstructuredMesh2D(mesh_path, bkd, rescale_origin=(0.0, 0.0))
    basis = VectorLagrangeBasis(mesh, degree=1)
    mesh.skfem_mesh()
    subdomain_names = mesh.subdomain_names()
    subdomain_elements = {
        name: mesh.subdomain_elements(name) for name in subdomain_names
    }

    def zero_dirichlet(
        coords: np.ndarray, time: float = 0.0,
    ) -> np.ndarray:
        return np.zeros(coords.shape[1])

    bc_left = DirichletBC(basis, "left_edge", zero_dirichlet, bkd)
    material_map = {
        name: (E_mean, poisson_ratio) for name in subdomain_names
    }

    # ---- Solve unit load cases ----
    def _solve_unit_load(
        traction_func: Callable[[np.ndarray, float], np.ndarray],
    ) -> np.ndarray:
        bc_top = NeumannBC(basis, "top_edge", traction_func, bkd)
        physics = CompositeLinearElasticity(
            basis=basis,
            material_map=material_map,
            element_materials=subdomain_elements,
            bkd=bkd,
            boundary_conditions=[bc_left, bc_top],
        )
        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=1)
        init = bkd.asarray(np.zeros(physics.nstates()))
        result = solver.solve(init)
        return bkd.to_numpy(result.solution)

    def const_traction(
        coords: np.ndarray, time: float = 0.0,
    ) -> np.ndarray:
        npts = coords.shape[1]
        traction = np.zeros((2, npts))
        traction[1, :] = -1.0
        return traction

    def slope_traction(
        coords: np.ndarray, time: float = 0.0,
    ) -> np.ndarray:
        x = coords[0]
        npts = coords.shape[1]
        traction = np.zeros((2, npts))
        traction[1, :] = -x / length
        return traction

    sol_const = _solve_unit_load(const_traction)
    sol_slope = _solve_unit_load(slope_traction)

    # ---- Sensor locations ----
    if sensor_xs is None:
        sensor_xs = bkd.linspace(length / 5.0, length, 5)
    nobs = sensor_xs.shape[0]
    sensor_xs_np = bkd.to_numpy(sensor_xs)

    # ---- Build design matrix A (nobs, 2) ----
    A_np = np.zeros((nobs, 2))
    for i, sx in enumerate(sensor_xs_np):
        dof_idx = _find_dof(basis, float(sx), height / 2.0, 1, bkd)
        A_np[i, 0] = sol_const[dof_idx]
        A_np[i, 1] = sol_slope[dof_idx]

    return bkd.asarray(A_np), sensor_xs


def build_cantilever_beam_obs_map(
    design_matrix: Array,
    bkd: Backend[Array],
) -> FunctionProtocol[Array]:
    """Wrap a design matrix as a FunctionProtocol.

    Parameters
    ----------
    design_matrix : Array
        Design matrix A. Shape: (nobs, nparams).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    FunctionProtocol[Array]
        Observation model: y = A @ theta.
    """
    nobs, nparams = design_matrix.shape
    A = design_matrix

    def _obs_fun(samples: Array) -> Array:
        return bkd.dot(A, samples)

    return FunctionFromCallable(nobs, nparams, _obs_fun, bkd)
