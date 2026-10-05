"""Robin boundary conditions on CompositeLinearElasticity.

Problem (2D, constant Lame values lam, mu):

    -div(sigma(u)) = f                  in Omega = [0, 1]^2
    sigma(u) = lam * tr(eps(u)) * I + 2 * mu * eps(u)
    sigma(u) n + a * u = g              on the Robin edge, a >= 0

The Robin condition is an elastic support: a spring of stiffness ``a``
pulling the boundary toward ``g / a``. Its weak form adds
``a * <u, v>`` to the stiffness and ``<g, v>`` to the load, so the spatial
residual is ``F = b - (K + K_Gamma) u``.

Two kinds of test. The manufactured-solution tests check convergence and use
the existing manufactured infrastructure. The remaining tests need no
derivation at all: they compare Robin with the already-validated Dirichlet
and Neumann conditions, or check a physical property, so they pin down what
``a`` and ``g`` mean independently of any manufactured data.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinHyperelasticityAdapter,
    create_elasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.util.backends.numpy import NumpyBkd
from scipy.sparse import issparse

_Arr = NDArray[np.floating[Any]]

_LAM = 1.5
_MU = 0.8
_ROBIN_A = 2.0
# Non-polynomial, coupled, and nonzero in both components on the Robin edge
# x = 1, so the spring term a * <u, v> acts on every Robin-edge DOF.
_SOL_STRS = [
    "0.1*sin(x + 0.5*y) + 0.05",
    "0.05*cos(0.3*x + y) + 0.02",
]
# BC order for the adapter: [left, right, bottom, top].
_BC_TYPES = ["D", "R", "D", "D"]

# Spring rest position and body force for the derivation-free tests.
_U0 = np.array([0.05, -0.02])
_BODY_FORCE = np.array([0.0, -0.1])


def _dense(mat: Any) -> _Arr:
    return np.asarray(mat.toarray() if issparse(mat) else mat)


def _set_exact_lame(physics: CompositeLinearElasticity[_Arr]) -> None:
    nelems = physics.basis().skfem_basis().mesh.nelements
    physics.set_lame_parameters(np.full(nelems, _LAM), np.full(nelems, _MU))


def _solve(system: GalerkinSystem[_Arr]) -> _Arr:
    bkd = system.bkd()
    solver = SteadyStateSolver(
        system.steady(), tol=1e-10, max_iter=5, line_search=False
    )
    result = solver.solve(bkd.asarray(np.zeros(system.nstates())))
    assert result.converged
    return np.asarray(bkd.to_numpy(result.solution))


# ---------------------------------------------------------------------------
# Manufactured solution
# ---------------------------------------------------------------------------


class _RobinDataAsNeumann:
    """Neumann data equal to the Robin data ``g = a u + sigma n`` on x = 1.

    Used as the wrong model: a Neumann condition with this data is the Robin
    condition with its spring term ``a u`` dropped.
    """

    def __init__(self, functions: Any, alpha: float) -> None:
        self._solution = functions["solution"]
        self._flux = functions["flux"]
        self._alpha = alpha

    def __call__(self, coords: _Arr) -> _Arr:
        u = np.asarray(self._solution(coords)).T  # (ndim, npts)
        stress = np.asarray(self._flux(coords))  # (ndim, npts, ndim)
        traction = stress[:, :, 0]  # sigma n with n = (1, 0)
        return np.asarray(self._alpha * u + traction)


def _mms_physics(nx: int, wrong_model: bool) -> Tuple[
    CompositeLinearElasticity[_Arr],
    GalerkinSystem[_Arr],
    VectorLagrangeBasis[_Arr],
    Any,
]:
    bkd = NumpyBkd()
    functions, _ = create_elasticity_manufactured_test(
        bounds=[0.0, 1.0, 0.0, 1.0],
        sol_strs=_SOL_STRS,
        lambda_str=str(_LAM),
        mu_str=str(_MU),
        bkd=bkd,
    )
    mesh = StructuredMesh2D(nx=nx, ny=nx, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
    basis = VectorLagrangeBasis(mesh, degree=1)
    adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
    bc_set = adapter.create_boundary_conditions(_BC_TYPES, robin_alpha=_ROBIN_A)
    bcs: List[Any] = bc_set.all_conditions()
    if wrong_model:
        bcs = list(bc_set.essential_bcs())
        bcs.append(
            NeumannBC(
                basis,
                "right",
                TimeIndependent(_RobinDataAsNeumann(functions, _ROBIN_A)),
                bkd,
            )
        )
    physics = CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=1.0,  # replaced by the manufactured Lame values
        poisson_ratio=0.3,
        bkd=bkd,
        body_force=adapter.forcing_for_galerkin(),
    )
    system = compose_galerkin_system(physics, bcs)
    _set_exact_lame(physics)
    return physics, system, basis, functions


def _mms_l2_error(nx: int, wrong_model: bool = False) -> float:
    physics, system, basis, functions = _mms_physics(nx, wrong_model)
    u_h = _solve(system)
    dof_coords = np.asarray(physics.bkd().to_numpy(basis.dof_coordinates()))
    exact_vals = np.asarray(functions["solution"](dof_coords))  # (ndofs, 2)
    ndofs = basis.ndofs()
    exact = exact_vals[np.arange(ndofs), np.arange(ndofs) % 2]
    err = u_h - exact
    mass = _dense(physics.mass_matrix())
    return float(np.sqrt(err @ mass @ err))


class TestElasticityRobinManufactured:
    """Manufactured-solution convergence with a Robin edge."""

    _NXS = (4, 8, 16)

    def test_robin_converges_at_element_order(self) -> None:
        """P1 L2 error falls at O(h^2) with a Robin edge."""
        errors = [_mms_l2_error(nx) for nx in self._NXS]
        rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        assert np.all(rates > 1.8), (errors, rates)

    def test_robin_without_spring_term_does_not_converge(self) -> None:
        """Wrong-model control: the same data with ``a u`` dropped (Neumann
        data g) converges to a different solution, so its error stalls.
        This shows the convergence test can see the spring term."""
        wrong = [_mms_l2_error(nx, wrong_model=True) for nx in self._NXS]
        right = [_mms_l2_error(nx) for nx in self._NXS]
        assert wrong[-1] / wrong[0] > 0.5, wrong
        assert wrong[-1] > 50.0 * right[-1], (wrong, right)


# ---------------------------------------------------------------------------
# Derivation-free checks: a clamped plate held by a spring edge
# ---------------------------------------------------------------------------


class _ConstantVector:
    """Spatially constant vector field, returned as ``(ndim, npts)``."""

    def __init__(self, value: _Arr) -> None:
        self._value = np.asarray(value, dtype=float)

    def __call__(self, coords: _Arr) -> _Arr:
        return np.outer(self._value, np.ones(coords.shape[1]))


def _plate(right_bc: str, a: float = 0.0, clamp: bool = True, nx: int = 6) -> Tuple[
    CompositeLinearElasticity[_Arr],
    GalerkinSystem[_Arr],
    VectorLagrangeBasis[_Arr],
]:
    """Plate clamped on the left, loaded by a body force, and held on the
    right by ``right_bc``: ``"robin"`` (spring of stiffness ``a`` toward
    ``_U0``), ``"dirichlet"`` (``u = _U0``), or ``"neumann"`` (traction
    ``a * _U0``, the Robin data with the spring removed)."""
    bkd = NumpyBkd()
    mesh = StructuredMesh2D(nx=nx, ny=nx, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
    basis = VectorLagrangeBasis(mesh, degree=1)
    bcs: List[Any] = []
    if clamp:
        bcs.append(DirichletBC(basis, "left", 0.0, bkd))
    if right_bc == "robin":
        data = TimeIndependent(_ConstantVector(a * _U0))
        bcs.append(RobinBC(basis, "right", a, data, bkd))
    elif right_bc == "dirichlet":
        bcs.append(
            DirichletBC(basis, "right", TimeIndependent(_ConstantVector(_U0)), bkd)
        )
    elif right_bc == "neumann":
        data = TimeIndependent(_ConstantVector(a * _U0))
        bcs.append(NeumannBC(basis, "right", data, bkd))
    else:
        raise ValueError(f"unknown right_bc {right_bc!r}")
    physics = CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        bkd=bkd,
        body_force=TimeIndependent(_ConstantVector(_BODY_FORCE)),
    )
    system = compose_galerkin_system(physics, bcs)
    _set_exact_lame(physics)
    return physics, system, basis


def _right_edge_gap(u: _Arr, basis: VectorLagrangeBasis[_Arr]) -> float:
    """Max distance of the right-edge displacement from the rest position."""
    dofs = np.asarray(basis.bkd().to_numpy(basis.get_dofs("right"))).astype(int)
    return float(np.max(np.abs(u[dofs] - _U0[dofs % 2])))


class TestElasticityRobinLimits:
    """Robin checked against Dirichlet, Neumann and physical properties."""

    def test_large_stiffness_approaches_dirichlet(self) -> None:
        """As a grows, the spring becomes a support: u -> Dirichlet u = g/a,
        with the difference falling like 1/a."""
        u_dir = _solve(_plate("dirichlet")[1])
        diffs = []
        for a in (1e2, 1e3, 1e4):
            u_rob = _solve(_plate("robin", a)[1])
            diffs.append(float(np.max(np.abs(u_rob - u_dir))))
        ratios = np.array(diffs[:-1]) / np.array(diffs[1:])
        assert np.all(ratios > 5.0), (diffs, ratios)
        assert diffs[-1] < 1e-3 * np.max(np.abs(u_dir)), diffs

    def test_zero_stiffness_equals_neumann(self) -> None:
        """At a = 0 the Robin data is a traction, entering with the same sign
        as Neumann data."""
        a = 0.0
        traction = 0.3
        bkd = NumpyBkd()
        mesh = StructuredMesh2D(nx=4, ny=4, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        data = TimeIndependent(_ConstantVector(traction * _U0))
        solutions = []
        for bc in (
            RobinBC(basis, "right", a, data, bkd),
            NeumannBC(basis, "right", data, bkd),
        ):
            physics = CompositeLinearElasticity.from_uniform(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=0.3,
                bkd=bkd,
            )
            system = compose_galerkin_system(
                physics, [DirichletBC(basis, "left", 0.0, bkd), bc]
            )
            solutions.append(_solve(system))
        bkd.assert_allclose(solutions[0], solutions[1], rtol=1e-10)

    def test_stiffer_spring_pulls_edge_toward_rest_position(self) -> None:
        """The right edge moves monotonically toward u0 as a increases."""
        gaps = []
        for a in (0.1, 1.0, 10.0, 100.0):
            _, system, basis = _plate("robin", a)
            gaps.append(_right_edge_gap(_solve(system), basis))
        assert np.all(np.diff(gaps) < 0.0), gaps

    def test_spring_makes_unclamped_operator_positive_definite(self) -> None:
        """With no clamp, K has rigid-body modes, so it is singular. A spring
        on one edge removes them: K + K_Gamma is symmetric positive
        definite. A wrong sign or a missing K_Gamma fails this."""
        physics, system, _ = _plate("robin", a=1.0, clamp=False, nx=3)
        stiffness = -_dense(system.spatial_operator().spatial_jacobian(
            physics.bkd().asarray(np.zeros(physics.nstates())), 0.0
        ))
        physics.bkd().assert_allclose(stiffness, stiffness.T, atol=1e-12)
        interior = _dense(physics.stiffness_matrix())
        assert np.min(np.linalg.eigvalsh(interior)) < 1e-10
        assert np.min(np.linalg.eigvalsh(stiffness)) > 1e-6


# ---------------------------------------------------------------------------
# Derivative regression checks (pass with or without the Robin stiffness)
# ---------------------------------------------------------------------------


class TestElasticityRobinDerivatives:
    """Jacobians stay consistent with the residual when Robin is present."""

    def test_state_jacobian(self) -> None:
        physics, system, _ = _plate("robin", a=_ROBIN_A, nx=3)
        operator = system.spatial_operator()
        bkd = physics.bkd()
        nstates = physics.nstates()

        def residual(samples: _Arr) -> _Arr:
            return bkd.stack(
                [
                    operator.spatial_residual(samples[:, ii], 0.0)
                    for ii in range(samples.shape[1])
                ],
                axis=1,
            )

        def jacobian(sample: _Arr) -> _Arr:
            return bkd.asarray(_dense(operator.spatial_jacobian(sample[:, 0], 0.0)))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates, nvars=nstates, fun=residual, jacobian=jacobian, bkd=bkd
        )
        rng = np.random.default_rng(7)
        sample = bkd.asarray(rng.normal(0.0, 0.1, (nstates, 1)))
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, relative=True)[0]
        assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 1e-6

    def test_lame_jacobian(self) -> None:
        """The Robin term does not depend on the Lame values."""
        physics, system, _ = _plate("robin", a=_ROBIN_A, nx=3)
        operator = system.spatial_operator()
        bkd = physics.bkd()
        nstates = physics.nstates()
        base = np.array([_LAM, _MU])
        rng = np.random.default_rng(11)
        state = bkd.asarray(rng.normal(0.0, 0.1, nstates))

        def residual(samples: _Arr) -> _Arr:
            cols = []
            for ii in range(samples.shape[1]):
                physics.set_lame_material_values(bkd.to_numpy(samples[:, ii]))
                cols.append(bkd.to_numpy(operator.spatial_residual(state, 0.0)))
            physics.set_lame_material_values(base)
            return bkd.asarray(np.stack(cols, axis=1))

        analytic = bkd.to_numpy(physics.residual_lame_jacobian(state))

        def jacobian(sample: _Arr) -> _Arr:
            return bkd.asarray(analytic)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates, nvars=2, fun=residual, jacobian=jacobian, bkd=bkd
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(bkd.asarray(base)[:, None], relative=True)[0]
        assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 1e-6
