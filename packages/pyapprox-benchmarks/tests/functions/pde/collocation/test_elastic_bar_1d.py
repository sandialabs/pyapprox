r"""Tests of the 1D linear elastic bar builder.

Each solve uses ``collocation_steady_function`` defaults. With a
constant modulus (a one-term ``BasisExpansion`` at :math:`p = 0`):

- the manufactured :math:`u = x(2 - x)` on :math:`[0, 2]` (:math:`E = 3`)
  is recovered to 1e-10, and the closed residual vanishes at the
  manufactured :math:`u = x(1 - x)` on :math:`[0, 1]` (:math:`E = 2`) to
  1e-10;
- :math:`u = f x (2L - x)/(2E) + T x/E` (:math:`L = 1.5`, :math:`E = 4`,
  :math:`f = 2`, :math:`T = 3`) is recovered to 1e-10, and with
  ``dirichlet_left = 0.25`` the same plus 0.25.

With a two-term lognormal KLE modulus the Jacobian passes central finite
differences at :math:`p = (0.1, -0.1)` (error ratio at most 1e-5).
"""

from typing import Generic

from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols import (
    FunctionProtocol,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import AffineTransform1D, TransformedMesh1D
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.manufactured import ManufacturedAdvectionDiffusionReaction
from pyapprox.pde.models.collocation.steady_function import (
    ParameterizedCollocationSystem,
    collocation_steady_function,
)
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.functions.pde.collocation.elastic_bar_1d import (
    build_linear_elastic_bar_1d,
)


def _mesh(bkd: Backend[Array], npts: int, length: float) -> TransformedMesh1D[Array]:
    return TransformedMesh1D(npts, bkd, AffineTransform1D((0.0, length), bkd))


def _make_kle_field_map(
    bkd: Backend[Array], mesh: TransformedMesh1D[Array], num_kle_terms: int = 2
) -> TransformedFieldMap[Array]:
    """Helper: create lognormal KLE field map on mesh nodes."""
    physical_pts = mesh.points()  # shape (1, npts)
    npts = physical_pts.shape[1]
    # Map [0, L] -> [0, 1] for correlation kernel
    x_min = physical_pts[0, 0]
    x_max = physical_pts[0, -1]
    length = x_max - x_min
    mesh_coords = (physical_pts - x_min) / length  # (1, npts)
    mean_log = bkd.zeros((npts,))
    return create_lognormal_kle_field_map(
        mesh_coords,
        mean_log,
        bkd,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
    )


def _constant_modulus(
    bkd: Backend[Array], npts: int, modulus: float
) -> BasisExpansion[Array]:
    """E = modulus + p_0, so ``modulus`` at p = 0."""
    return BasisExpansion(bkd, modulus, [bkd.ones((npts,))])


class _ConstantLoad(Generic[Array]):
    """A uniform body force."""

    def __init__(self, bkd: Backend[Array], value: float) -> None:
        self._bkd, self._value = bkd, value

    def __call__(self, coords: Array) -> Array:
        return self._value * self._bkd.ones((coords.shape[1],))


class _ManufacturedForcing(Generic[Array]):
    """The forcing of a manufactured solution at the given coordinates."""

    def __init__(self, solution: ManufacturedAdvectionDiffusionReaction[Array]) -> None:
        self._solution = solution

    def __call__(self, coords: Array) -> Array:
        return self._solution.functions["forcing"](coords)


def _every_state(bkd: Backend[Array], nstates: int) -> StateFunctional[Array, Array]:
    return StateFunctional.from_terms(
        bkd,
        nstates,
        state_terms=(ObservedValue(StateEntries(bkd, nstates, list(range(nstates)))),),
    )


def _solve_at_zero(
    configured: ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]],
    bkd: Backend[Array],
    npts: int,
) -> Array:
    """Every state at p = 0, shape (npts,)."""
    function = collocation_steady_function(
        configured.system,
        ParameterState(configured.parameterization),
        _every_state(bkd, npts),
    )
    nvars = configured.parameterization.nparams()
    return function(bkd.zeros((nvars, 1)))[:, 0]


def _manufactured(
    bkd: Backend[Array], sol_str: str, modulus: float
) -> ManufacturedAdvectionDiffusionReaction[Array]:
    return ManufacturedAdvectionDiffusionReaction(
        sol_str=sol_str,
        nvars=1,
        diff_str=str(modulus),
        react_str="0",
        vel_strs=["0"],
        bkd=bkd,
        oned=True,
    )


def _manufactured_bar(
    bkd: Backend[Array],
    man_sol: ManufacturedAdvectionDiffusionReaction[Array],
    mesh: TransformedMesh1D[Array],
    modulus: float,
) -> ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]]:
    """The bar of a manufactured solution vanishing at the left end, its
    traction E u'(b) = -(flux . n) read from the generator."""
    right_idx = mesh.boundary_indices(1)
    flux_normal = man_sol.neumann_values(
        mesh.points()[:, right_idx], mesh.boundary_normals(1), convention="flux"
    )
    return build_linear_elastic_bar_1d(
        mesh,
        TimeIndependent(_ManufacturedForcing(man_sol)),
        _constant_modulus(bkd, mesh.npts(), modulus),
        traction=-bkd.to_float(flux_normal[0]),
    )


class TestElasticBar1D:
    def test_manufactured_solution(self, bkd: Backend[Array]) -> None:
        """Manufactured solution: solve and compare to exact."""
        npts = 20
        E_val = 3.0
        # Manufactured solution: u(x) = x*(2-x), vanishes at x=0
        man_sol = _manufactured(bkd, "x*(2-x)", E_val)
        mesh = _mesh(bkd, npts, 2.0)
        u_exact = man_sol.functions["solution"](mesh.points())
        configured = _manufactured_bar(bkd, man_sol, mesh, E_val)
        u_num = _solve_at_zero(configured, bkd, npts)
        bkd.assert_allclose(u_num, u_exact, atol=1e-10)

    def test_residual_at_exact(self, bkd: Backend[Array]) -> None:
        """Residual is near zero at the exact manufactured solution."""
        npts = 15
        E_val = 2.0
        man_sol = _manufactured(bkd, "x*(1-x)", E_val)
        mesh = _mesh(bkd, npts, 1.0)
        u_exact = man_sol.functions["solution"](mesh.points())
        configured = _manufactured_bar(bkd, man_sol, mesh, E_val)
        parameters = ParameterState(configured.parameterization)
        residual = collocation_steady_residual(configured.system, parameters)
        parameters.set_param(bkd.zeros((1,)))
        bkd.assert_allclose(
            residual(bkd.reshape(u_exact, (npts, 1))),
            bkd.zeros((npts, 1)),
            atol=1e-10,
        )

    def test_constant_E_analytical(self, bkd: Backend[Array]) -> None:
        """Constant E: u(x) = f*x*(2L-x)/(2E) + T*x/E (analytical), and
        shifted by a prescribed left displacement."""
        npts = 20
        length = 1.5
        E_val = 4.0
        f_val = 2.0
        T_val = 3.0

        mesh = _mesh(bkd, npts, length)
        x = mesh.points()[0, :]  # physical coords, shape (npts,)

        # Analytical: u(x) = f*x*(2L-x)/(2E) + T*x/E
        u_exact = f_val * x * (2.0 * length - x) / (2.0 * E_val) + T_val * x / E_val

        for shift in (0.0, 0.25):
            configured = build_linear_elastic_bar_1d(
                mesh,
                TimeIndependent(_ConstantLoad(bkd, f_val)),
                _constant_modulus(bkd, npts, E_val),
                traction=T_val,
                dirichlet_left=shift,
            )
            u_num = _solve_at_zero(configured, bkd, npts)
            bkd.assert_allclose(u_num, u_exact + shift, atol=1e-10)

    def test_param_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on KLE-parameterized elastic bar."""
        npts = 20
        mesh = _mesh(bkd, npts, 1.0)
        configured = build_linear_elastic_bar_1d(
            mesh,
            TimeIndependent(_ConstantLoad(bkd, 1.0)),
            _make_kle_field_map(bkd, mesh),
            traction=1.0,
        )
        fwd = collocation_steady_function(
            configured.system,
            ParameterState(configured.parameterization),
            _every_state(bkd, npts),
        )

        assert isinstance(fwd, FunctionProtocol)
        assert fwd.derivatives().jacobian is not None

        checker = DerivativeChecker(fwd)
        sample = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = bkd.to_float(bkd.min(errors) / bkd.max(errors))
        assert ratio <= 1e-5
