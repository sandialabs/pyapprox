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
    Any,
    Literal,
    Union,
)

if TYPE_CHECKING:
    import skfem

    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.boundary.implementations import NeumannBC
    from pyapprox.pde.galerkin.physics.composite_hyperelasticity import (
        CompositeHyperelasticityPhysics,
    )
    from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
        CompositeLinearElasticity,
    )
    from pyapprox.pde.galerkin.system import GalerkinSystem
    from pyapprox.util.rootfinding.nonlinear import Newton

import numpy as np
from numpy.typing import NDArray

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

# The skfem models are bound to NumPy: skfem assembles SciPy matrices.
_Np = NDArray[Any]

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


class _TopTraction:
    """Top-edge traction (0, -q0 x / L), time-independent; a class, not
    a closure, so the model pickles. Declared time-independent where it is
    handed to the boundary condition."""

    def __init__(self, q0: float, length: float) -> None:
        self._q0 = q0
        self._length = length

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.zeros(coords.shape[1]), -self._q0 * coords[0] / self._length]
        )


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


class SharedFieldBeamModel:
    """Forward model: (xi, v) -> (tip_deflection, integrated_VM_stress).

    Consumes ``(num_kle_terms + 1)`` input variables: ``xi`` for the
    shared KLE field and ``v`` for the wind speed.  The traction load
    is ``q0 = v^2`` (dynamic pressure).  Each evaluation sets the E
    field from the KLE, computes the load from the wind speed, and
    solves the FEM problem.

    The Lame fields are applied through a ``ParameterState`` over
    ``LameFieldApplyOnlyParameterization`` and the load by replacing the
    top traction; the steady system of ``system`` (built once) is solved
    by ``newton`` with sparse LU from zero. Kept setups are valid for one
    parameter value, so the setup strategy is cleared after each change.
    The model is bound to NumPy: skfem assembles SciPy matrices.
    """

    def __init__(
        self,
        physics: Union[
            "CompositeLinearElasticity[_Np]",
            "CompositeHyperelasticityPhysics[_Np]",
        ],
        system: "GalerkinSystem[_Np]",
        newton: "Newton[_Np, _Np]",
        neumann_bc: "NeumannBC[_Np]",
        scalar_skfem_basis: "skfem.CellBasis",
        kle_field_map: TransformedFieldMap[_Np],
        poisson_ratio: float,
        tip_dof_index: int,
        length: float,
        bkd: NumpyBkd,
        num_kle_terms: int,
        vector_basis: "VectorLagrangeBasis[_Np]",
    ) -> None:
        from pyapprox.implicitfunction.setups import RecomputeAlways
        from pyapprox.implicitfunction.steady_system import SteadySystem
        from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
            skfem_discretization,
        )
        from pyapprox.pde.models.galerkin.steady_residual import (
            galerkin_steady_residual,
        )
        from pyapprox.pde.parameterizations.galerkin_lame import (
            LameFieldApplyOnlyParameterization,
        )
        from pyapprox.pde.parameterizations.parameter_state import ParameterState
        from pyapprox.util.linalg.operators import InnerProductSpaceProtocol
        from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU

        self._physics = physics
        field_shape = (
            int(scalar_skfem_basis.mesh.nelements),
            int(scalar_skfem_basis.dx.shape[1]),
        )
        self._parameters = ParameterState(
            LameFieldApplyOnlyParameterization(physics, field_shape, bkd)
        )
        discretization = skfem_discretization(system)
        space = discretization.space()
        if not isinstance(space, InnerProductSpaceProtocol):
            raise TypeError("the beam's state space has no inner product")
        self._space = space
        self._setups: RecomputeAlways[_Np, CSRFormat] = RecomputeAlways(
            ScipySparseLU()
        )
        self._steady = SteadySystem(
            galerkin_steady_residual(discretization, self._parameters),
            space,
            self._setups,
        )
        self._newton = newton
        self._neumann_bc = neumann_bc
        self._scalar_skfem_basis = scalar_skfem_basis
        self._kle_field_map = kle_field_map
        self._poisson_ratio = poisson_ratio
        self._tip_dof_index = tip_dof_index
        self._length = length
        self._bkd = bkd
        self._num_kle_terms = num_kle_terms

        # Vector basis for von Mises post-processing
        self._vector_basis = vector_basis

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nvars(self) -> int:
        return self._num_kle_terms + 1

    def nqoi(self) -> int:
        return 2

    def derivatives(self) -> Derivatives[_Np]:
        return Derivatives.none()

    def __call__(self, samples: _Np) -> _Np:
        from pyapprox.pde.constitutive.coefficient_functions import (
            TimeIndependent,
        )
        from pyapprox.pde.galerkin.postprocessing import (
            integrate,
            von_mises_stress,
        )

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
            self._parameters.set_param(
                np.concatenate([lam_arr.ravel(), mu_arr.ravel()])
            )
            self._neumann_bc.set_flux_func(TimeIndependent(_TopTraction(q0, L)))
            self._setups.clear()

            point = self._steady.solve(self._newton, self._space.zeros())
            if not point.converged():
                raise RuntimeError(
                    f"the beam solve did not converge: {point.info().message}"
                )
            sol_np = np.array(point.solution()[:, 0])

            results[0, ii] = sol_np[self._tip_dof_index]

            # plane_strain matches the constitutive law the 2D physics
            # assembles (sigma = lam*tr(eps)*I + 2*mu*eps)
            vm = von_mises_stress(
                self._vector_basis,
                sol_np,
                lam_arr,
                mu_arr,
                assumption="plane_strain",
            )
            results[1, ii] = integrate(self._vector_basis, vm)

        return bkd.asarray(results)


def build_shared_field_beam(
    bkd: NumpyBkd,
    mesh_path: str,
    physics_type: Literal["linear", "neohookean"],
    num_kle_terms: int = 5,
    length: float = 100.0,
    height: float = 30.0,
    E_mean: float = 1e4,
    poisson_ratio: float = 0.3,
    sigma: float = 0.3,
    correlation_length: float = 0.3,
) -> SharedFieldBeamModel:
    """Build a shared-field beam model for use in an ensemble.

    Unlike ``build_cantilever_beam_2d_linear``/``neohookean``, this
    factory uses a single KLE on the full mesh (no subdomain splitting)
    and leaves the load magnitude as an input parameter (not fixed).

    Parameters
    ----------
    bkd : NumpyBkd
        NumPy backend: the model is bound to skfem's SciPy matrices.
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
    from skfem import Basis as SkfemBasis

    from pyapprox.pde.constitutive.coefficient_functions import (
        TimeIndependent,
    )
    from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.pde.galerkin.boundary.implementations import (
        DirichletBC,
        NeumannBC,
    )
    from pyapprox.pde.galerkin.compose import compose_galerkin_system
    from pyapprox.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.pde.galerkin.system import GalerkinSystem
    from pyapprox.util.rootfinding.nonlinear import Newton

    mesh = UnstructuredMesh2D(mesh_path, bkd, rescale_origin=(0.0, 0.0))
    basis = VectorLagrangeBasis(mesh, degree=1)
    skfem_mesh = mesh.skfem_mesh()

    scalar_element = basis.scalar_basis().skfem_basis().elem
    scalar_skfem_basis = SkfemBasis(skfem_mesh, scalar_element)

    kle_field_map = _build_kle_field_map(
        skfem_mesh, bkd, num_kle_terms, sigma, correlation_length, E_mean,
    )

    bc_left = DirichletBC(basis, "left_edge", 0.0, bkd)

    def top_traction(coords: np.ndarray) -> np.ndarray:
        x = coords[0]
        npts = coords.shape[1]
        traction = np.zeros((2, npts))
        traction[1, :] = -1.0 * x / length
        return traction

    bc_top = NeumannBC(basis, "top_edge", TimeIndependent(top_traction), bkd)

    subdomain_names = mesh.subdomain_names()
    subdomain_elements = {
        name: mesh.subdomain_elements(name) for name in subdomain_names
    }
    material_map = {
        name: (E_mean, poisson_ratio) for name in subdomain_names
    }

    physics: Union[
        "CompositeLinearElasticity[_Np]", "CompositeHyperelasticityPhysics[_Np]"
    ]
    system: GalerkinSystem[_Np]
    newton: Newton[_Np, _Np]
    if physics_type == "linear":
        from pyapprox.pde.galerkin.physics import CompositeLinearElasticity

        linear = CompositeLinearElasticity(
            basis=basis,
            material_map=material_map,
            element_materials=subdomain_elements,
            bkd=bkd,
        )
        physics = linear
        system = compose_galerkin_system(linear, [bc_left, bc_top])
        newton = Newton(
            atol=1e-10, rtol=0.0, maxiters=1, line_search=True, linesearch_maxiters=10
        )
    elif physics_type == "neohookean":
        from pyapprox.pde.galerkin.physics import (
            CompositeHyperelasticityPhysics,
        )

        neohookean = CompositeHyperelasticityPhysics(
            basis=basis,
            material_map=material_map,
            element_materials=subdomain_elements,
            bkd=bkd,
        )
        physics = neohookean
        system = compose_galerkin_system(neohookean, [bc_left, bc_top])
        newton = Newton(
            atol=1e-10, rtol=0.0, maxiters=50, line_search=True, linesearch_maxiters=10
        )
    else:
        raise ValueError(
            f"Unknown physics_type '{physics_type}'; "
            "expected 'linear' or 'neohookean'"
        )

    tip_dof = _find_tip_dof(basis, length, height, bkd)

    return SharedFieldBeamModel(
        physics=physics,
        system=system,
        newton=newton,
        neumann_bc=bc_top,
        scalar_skfem_basis=scalar_skfem_basis,
        kle_field_map=kle_field_map,
        poisson_ratio=poisson_ratio,
        tip_dof_index=tip_dof,
        length=length,
        bkd=bkd,
        num_kle_terms=num_kle_terms,
        vector_basis=basis,
    )
