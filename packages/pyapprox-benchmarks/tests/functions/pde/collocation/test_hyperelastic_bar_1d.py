r"""Tests of the 1D hyperelastic bar builder.

Each solve uses ``collocation_steady_function`` defaults (Newton with a
line search). With a constant modulus (a one-term ``BasisExpansion`` at
:math:`p = 0`) set from the generator's Lame parameters:

- the manufactured :math:`u = 0.1 x(1 - x)` (:math:`\lambda = 2`,
  :math:`\mu = 1`, 25 nodes) is recovered to 1e-9, and the closed
  residual vanishes at the manufactured :math:`u = 0.05 x(1 - x)`
  (:math:`\lambda = 3`, :math:`\mu = 2`, 20 nodes) to 1e-10.

With a two-term lognormal KLE modulus (:math:`E = 1` at :math:`p = 0`):

- the Jacobian passes central finite differences at
  :math:`p = (0.1, -0.1)` (error ratio at most 1e-5);
- at traction 0.01 and :math:`\nu = 0` the displacement matches the
  linear bar's to 1e-2 relative;
- at traction 0.5 the first Piola-Kirchhoff stress at the right end
  equals the traction to 1e-8 relative.
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
from pyapprox.pde.constitutive import NeoHookeanStress
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.manufactured.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
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
from pyapprox_benchmarks.functions.pde.collocation.hyperelastic_bar_1d import (
    build_hyperelastic_bar_1d,
)


def _mesh(bkd: Backend[Array], npts: int, length: float) -> TransformedMesh1D[Array]:
    return TransformedMesh1D(npts, bkd, AffineTransform1D((0.0, length), bkd))


class _ConstantLoad(Generic[Array]):
    """A uniform body force."""

    def __init__(self, bkd: Backend[Array], value: float) -> None:
        self._bkd, self._value = bkd, value

    def __call__(self, coords: Array) -> Array:
        return self._value * self._bkd.ones((coords.shape[1],))


class _ManufacturedForcing(Generic[Array]):
    """The forcing of a manufactured solution at the given coordinates."""

    def __init__(self, solution: ManufacturedHyperelasticityEquations[Array]) -> None:
        self._solution = solution

    def __call__(self, coords: Array) -> Array:
        # oned=True + VectorSolutionMixin => (npts, ncomponents)
        return self._solution.functions["forcing"](coords)[:, 0]


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


def _manufactured_bar(
    bkd: Backend[Array],
    sol_str: str,
    npts: int,
    lamda_val: float,
    mu_val: float,
) -> tuple[
    ManufacturedHyperelasticityEquations[Array],
    TransformedMesh1D[Array],
    ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]],
]:
    """The bar on [0, 1] of a manufactured solution vanishing at x=0,
    with the constant E and nu of the Lame parameters, and the traction
    (P . n at x=1) read from the generator."""
    stress_model: NeoHookeanStress[Array] = NeoHookeanStress(
        lamda=lamda_val, mu=mu_val
    )
    man_sol = ManufacturedHyperelasticityEquations(
        sol_strs=[sol_str],
        nvars=1,
        stress_model=stress_model,
        bkd=bkd,
        oned=True,
    )
    mesh = _mesh(bkd, npts, 1.0)
    right_idx = mesh.boundary_indices(1)
    right_normals = mesh.boundary_normals(1)
    # flux shape: (1, npts, 1) for 1D with oned=True
    flux = man_sol.functions["flux"](mesh.points())
    traction_val = flux[0, right_idx, 0] * right_normals[:, 0]
    poisson_ratio = lamda_val / (2.0 * (lamda_val + mu_val))
    youngs = mu_val * (3.0 * lamda_val + 2.0 * mu_val) / (lamda_val + mu_val)
    configured = build_hyperelastic_bar_1d(
        mesh,
        TimeIndependent(_ManufacturedForcing(man_sol)),
        BasisExpansion(bkd, youngs, [bkd.ones((npts,))]),
        poisson_ratio,
        traction=bkd.to_float(traction_val[0]),
    )
    return man_sol, mesh, configured


def _make_kle_field_map(
    bkd: Backend[Array], mesh: TransformedMesh1D[Array], num_kle_terms: int = 2
) -> TransformedFieldMap[Array]:
    """Helper: create lognormal KLE field map on mesh nodes."""
    physical_pts = mesh.points()  # shape (1, npts)
    npts = physical_pts.shape[1]
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


class TestHyperelasticBar1D:
    def test_manufactured_solution(self, bkd: Backend[Array]) -> None:
        """Manufactured solution: solve and compare to exact."""
        npts = 25
        # u(x) = 0.1*x*(1-x), vanishes at x=0
        man_sol, mesh, configured = _manufactured_bar(
            bkd, "0.1*x*(1-x)", npts, lamda_val=2.0, mu_val=1.0
        )
        # oned=True + VectorSolutionMixin => (npts, ncomponents)
        u_exact = man_sol.functions["solution"](mesh.points())[:, 0]
        u_num = _solve_at_zero(configured, bkd, npts)
        bkd.assert_allclose(u_num, u_exact, atol=1e-9)

    def test_residual_at_exact(self, bkd: Backend[Array]) -> None:
        """Residual is near zero at the exact manufactured solution."""
        npts = 20
        man_sol, mesh, configured = _manufactured_bar(
            bkd, "0.05*x*(1-x)", npts, lamda_val=3.0, mu_val=2.0
        )
        u_exact = man_sol.functions["solution"](mesh.points())[:, 0]
        parameters = ParameterState(configured.parameterization)
        residual = collocation_steady_residual(configured.system, parameters)
        parameters.set_param(bkd.zeros((1,)))
        bkd.assert_allclose(
            residual(bkd.reshape(u_exact, (npts, 1))),
            bkd.zeros((npts, 1)),
            atol=1e-10,
        )

    def test_param_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on KLE-parameterized hyperelastic bar."""
        npts = 20
        nu = 0.3
        mesh = _mesh(bkd, npts, 1.0)
        configured = build_hyperelastic_bar_1d(
            mesh,
            TimeIndependent(_ConstantLoad(bkd, 1.0)),
            _make_kle_field_map(bkd, mesh),
            nu,
            traction=0.5,
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

    def test_small_strain_matches_linear(self, bkd: Backend[Array]) -> None:
        """Small traction: hyperelastic solution ~ linear elastic solution.

        Uses nu=0 so the 1D Neo-Hookean linearized modulus (2*mu + lambda)
        equals E. With nu>0, the 3D Lame-based effective 1D modulus differs
        from the plane-stress E used by the linear bar.
        """
        npts = 25
        nu = 0.0  # ensures linearized Neo-Hookean modulus = E
        small_traction = 0.01

        mesh = _mesh(bkd, npts, 1.0)
        field_map = _make_kle_field_map(bkd, mesh)
        no_load = TimeIndependent(_ConstantLoad(bkd, 0.0))

        # Evaluate at KLE coefficients = 0 (mean field)
        sol_hyper = _solve_at_zero(
            build_hyperelastic_bar_1d(
                mesh, no_load, field_map, nu, traction=small_traction
            ),
            bkd,
            npts,
        )
        sol_linear = _solve_at_zero(
            build_linear_elastic_bar_1d(
                mesh, no_load, field_map, traction=small_traction
            ),
            bkd,
            npts,
        )

        # Should match to ~O(traction^2) relative error
        # atol handles the Dirichlet node where both solutions are ~0
        bkd.assert_allclose(sol_hyper, sol_linear, rtol=1e-2, atol=1e-12)

    def test_flux_neumann_bc(self, bkd: Backend[Array]) -> None:
        """After solve, PK1 stress at right boundary matches traction."""
        npts = 25
        nu = 0.3
        T_val = 0.5

        mesh = _mesh(bkd, npts, 1.0)
        configured = build_hyperelastic_bar_1d(
            mesh,
            TimeIndependent(_ConstantLoad(bkd, 0.0)),
            _make_kle_field_map(bkd, mesh),
            nu,
            traction=T_val,
        )
        basis = configured.basis

        # Solve at mean (KLE coeffs = 0): KLE gives E=exp(0)=1.0
        u = _solve_at_zero(configured, bkd, npts)

        # Compute PK1 stress at right boundary using E from KLE
        Dx = basis.derivative_matrix(1, 0)
        F = 1.0 + Dx @ u

        # KLE at zero coeffs: E = exp(mean_log=0) = 1.0
        E_actual = 1.0
        dmu_dE = 1.0 / (2.0 * (1.0 + nu))
        dlam_dE = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu_val = E_actual * dmu_dE
        lam_val = E_actual * dlam_dE
        stress_model: NeoHookeanStress[Array] = NeoHookeanStress(
            lamda=lam_val, mu=mu_val
        )
        P = stress_model.compute_stress_1d(F, bkd)

        # At x=L with n=+1: flux.n = P, should equal prescribed traction
        right_idx = mesh.boundary_indices(1)
        bkd.assert_allclose(
            P[right_idx],
            T_val * bkd.ones((1,)),
            rtol=1e-8,
        )
