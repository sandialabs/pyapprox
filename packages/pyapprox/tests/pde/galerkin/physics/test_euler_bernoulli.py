"""Tests for Euler-Bernoulli beam physics.

Tests the analytical and FEM implementations using:
- Manufactured solutions with polynomial loads (exactly representable)
- Residual-is-zero-at-exact-solution verification
- Jacobian consistency via finite differences
- Convergence studies for non-polynomial loads
- Tip deflection agreement between analytical and FEM
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


from typing import Callable, Union

import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.galerkin.basis import CubicHermiteBasis, LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirectDirichletBC,
    DirichletBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.euler_bernoulli import (
    EulerBernoulliBeamAnalytical,
    EulerBernoulliBeamFEM,
    element_curvatures,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.galerkin_model import (
    GalerkinModel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


def _to_dense(mat):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(mat):
        return mat.toarray()
    if hasattr(mat, 'detach'):
        return mat.detach().numpy()
    return np.asarray(mat)


class TestEulerBernoulliAnalytical:
    """Tests for the analytical Euler-Bernoulli beam solutions."""
    def test_uniform_load_tip_deflection(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="uniform"
        )
        bkd.assert_allclose(
            bkd.asarray([beam.tip_deflection()]),
            bkd.asarray([q0 * L**4 / (8.0 * EI)]),
            rtol=1e-14,
        )

    def test_linear_load_tip_deflection(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        L, EI, q0 = 2.0, 3.0, 5.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="linear"
        )
        bkd.assert_allclose(
            bkd.asarray([beam.tip_deflection()]),
            bkd.asarray([11.0 * q0 * L**4 / (120.0 * EI)]),
            rtol=1e-14,
        )

    def test_uniform_load_boundary_conditions(self, numpy_bkd) -> None:
        """w(0)=0, w'(0)=0 for cantilever."""
        bkd = numpy_bkd
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="uniform"
        )
        x = bkd.asarray(np.array([0.0]))
        bkd.assert_allclose(
            beam.deflection(x), bkd.asarray([0.0]), atol=1e-15
        )
        bkd.assert_allclose(
            beam.slope(x), bkd.asarray([0.0]), atol=1e-15
        )

    def test_linear_load_boundary_conditions(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="linear"
        )
        x = bkd.asarray(np.array([0.0]))
        bkd.assert_allclose(
            beam.deflection(x), bkd.asarray([0.0]), atol=1e-15
        )
        bkd.assert_allclose(
            beam.slope(x), bkd.asarray([0.0]), atol=1e-15
        )

    def test_uniform_load_deflection_values(self, numpy_bkd) -> None:
        """Check w(x) at interior points against formula."""
        bkd = numpy_bkd
        L, EI, q0 = 1.0, 1.0, 1.0
        beam = EulerBernoulliBeamAnalytical(
            length=L, EI=EI, q0=q0, bkd=bkd, load_type="uniform"
        )
        x = bkd.asarray(np.array([0.5]))
        # w(0.5) = 1/24 * 0.25 * (6 - 2 + 0.25) = 4.25/96
        expected = bkd.asarray([4.25 / 96.0])
        bkd.assert_allclose(beam.deflection(x), expected, rtol=1e-14)

    def test_invalid_load_type_raises(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError):
            EulerBernoulliBeamAnalytical(
                length=1.0, EI=1.0, q0=1.0, bkd=bkd,
                load_type="quadratic"
            )


    @pytest.mark.parametrize("load_type", ["uniform", "linear"])
    def test_slope_is_derivative_of_deflection(
        self, numpy_bkd: NumpyBkd, load_type: str
    ) -> None:
        bkd = numpy_bkd
        beam = EulerBernoulliBeamAnalytical(
            length=2.0, EI=3.0, q0=1.5, bkd=bkd, load_type=load_type
        )
        x, h = np.linspace(0.1, 1.9, 7), 1e-6
        central = (beam.deflection(x + h) - beam.deflection(x - h)) / (2 * h)
        bkd.assert_allclose(beam.slope(x), central, rtol=1e-8, atol=1e-10)


def _uniform(q0: float) -> Callable[[np.ndarray], np.ndarray]:
    def load(x: np.ndarray) -> np.ndarray:
        return q0 * np.ones_like(x)

    return load


def _beam(
    bkd: Backend[Array],
    nx: int,
    length: float,
    EI: Union[float, np.ndarray],
    load_func: Callable[[np.ndarray], np.ndarray],
) -> EulerBernoulliBeamFEM[Array]:
    basis = CubicHermiteBasis(StructuredMesh1D(nx, (0.0, length), bkd))
    return EulerBernoulliBeamFEM(basis, EI, load_func, bkd)


def _cantilever(beam: EulerBernoulliBeamFEM[Array]) -> GalerkinSystem[Array]:
    """Clamped at the left end, free at the right."""
    bkd = beam.bkd()
    return compose_galerkin_system(
        beam, [DirichletBC(beam.basis(), "left", 0.0, bkd)]
    )


def _solve(system: GalerkinSystem[Array]) -> Array:
    bkd = system.bkd()
    return GalerkinModel(system, bkd).solve_steady(
        bkd.zeros((system.nstates(),))
    )


def _tip(beam: EulerBernoulliBeamFEM[Array], coeffs: Array) -> float:
    bkd = beam.bkd()
    length = float(bkd.to_numpy(beam.basis().mesh().nodes())[0, -1])
    return float(
        bkd.to_numpy(beam.basis().evaluate(coeffs, bkd.asarray([[length]])))[
            0
        ]
    )


def _nodes(beam: EulerBernoulliBeamFEM[Array]) -> Array:
    return beam.basis().mesh().nodes()[0]


class TestEulerBernoulliFEM:
    """Tests for the FEM Euler-Bernoulli beam."""

    # ------------------------------------------------------------------
    # Basic structure tests
    # ------------------------------------------------------------------

    def test_ndofs(self, numpy_bkd: NumpyBkd) -> None:
        nx = 5
        beam = _beam(numpy_bkd, nx, 1.0, 1.0, _uniform(1.0))
        assert beam.nstates() == 2 * (nx + 1)

    def test_rejects_non_hermite_basis(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        basis = LagrangeBasis(StructuredMesh1D(4, (0.0, 1.0), bkd), degree=2)
        with pytest.raises(TypeError, match="CubicHermiteBasis"):
            EulerBernoulliBeamFEM(basis, 1.0, _uniform(1.0), bkd)

    def test_rejects_wrong_EI_shape(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(ValueError, match="one value per element"):
            _beam(numpy_bkd, 4, 1.0, np.ones(5), _uniform(1.0))

    def test_stiffness_symmetric(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        K_np = _to_dense(_beam(bkd, 4, 1.0, 1.0, _uniform(1.0)).stiffness_matrix())
        bkd.assert_allclose(
            bkd.asarray(K_np), bkd.asarray(K_np.T), atol=1e-14
        )

    def test_mass_symmetric_positive(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        M_np = _to_dense(_beam(bkd, 4, 1.0, 1.0, _uniform(1.0)).mass_matrix())
        bkd.assert_allclose(
            bkd.asarray(M_np), bkd.asarray(M_np.T), atol=1e-14
        )
        assert np.all(np.linalg.eigvalsh(M_np) > -1e-14)

    # ------------------------------------------------------------------
    # Solutions of the composed cantilever
    # ------------------------------------------------------------------

    def test_residual_zero_at_solution(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system = _cantilever(_beam(bkd, 8, 2.0, 1.5, _uniform(1.0)))
        sol = _solve(system)
        bkd.assert_allclose(
            system.steady().steady_residual(sol),
            bkd.zeros((system.nstates(),)),
            atol=1e-10,
        )

    def test_simply_supported_beam(self, numpy_bkd: NumpyBkd) -> None:
        """Deflection pinned at both ends (value component only), uniform
        load q: w = q x (L^3 - 2 L x^2 + x^3) / (24 EI) at the nodes.

        Hermite cubics are nodally exact for beams; clamping instead of
        pinning gives a different beam and fails this.
        """
        bkd = numpy_bkd
        EI_val, L, nx, q = 1.5, 2.0, 8, 1.0
        beam = _beam(bkd, nx, L, EI_val, _uniform(q))
        basis = beam.basis()
        system = compose_galerkin_system(
            beam,
            [
                DirichletBC(basis, "left", 0.0, bkd, components=(0,)),
                DirichletBC(basis, "right", 0.0, bkd, components=(0,)),
            ],
        )
        sol = _solve(system)
        x = bkd.to_numpy(_nodes(beam))
        w_exact = q * x * (L**3 - 2 * L * x**2 + x**3) / (24 * EI_val)
        bkd.assert_allclose(
            basis.values(sol), bkd.asarray(w_exact), rtol=1e-10, atol=1e-12
        )

    def test_all_dofs_prescribed_residual_zero(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """With every DOF prescribed to the interpolant of w = x^2, the
        residual vanishes there: each row is ``state[dof] - g``."""
        bkd = numpy_bkd
        beam = _beam(bkd, 4, 1.0, 1.0, lambda x: np.zeros_like(x))
        basis = beam.basis()

        def exact(coords: np.ndarray) -> np.ndarray:
            return np.stack([coords[0] ** 2, 2.0 * coords[0]])

        state = basis.interpolate(exact)
        system = compose_galerkin_system(
            beam, [DirectDirichletBC(bkd.arange(beam.nstates()), state, bkd)]
        )
        bkd.assert_allclose(
            system.steady().steady_residual(state),
            bkd.zeros((beam.nstates(),)),
            atol=1e-12,
        )

    def test_jacobian_fd_consistency(self, numpy_bkd: NumpyBkd) -> None:
        """Jacobian matches finite differences of the residual."""
        bkd = numpy_bkd
        steady = _cantilever(_beam(bkd, 4, 1.0, 2.0, _uniform(1.0))).steady()
        ndofs = steady.nstates()
        state_np = np.random.default_rng(42).normal(size=ndofs) * 0.01
        jac_np = _to_dense(steady.steady_jacobian(bkd.asarray(state_np)))
        eps = 1e-7
        jac_fd = np.zeros((ndofs, ndofs))
        for j in range(ndofs):
            step = np.zeros(ndofs)
            step[j] = eps
            res_p = bkd.to_numpy(steady.steady_residual(bkd.asarray(state_np + step)))
            res_m = bkd.to_numpy(steady.steady_residual(bkd.asarray(state_np - step)))
            jac_fd[:, j] = (res_p - res_m) / (2.0 * eps)
        bkd.assert_allclose(bkd.asarray(jac_np), bkd.asarray(jac_fd), atol=1e-6)

    def test_clamp_constrains_value_and_slope(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The clamp constrains both left DOFs; their Jacobian rows are
        identity rows."""
        bkd = numpy_bkd
        system = _cantilever(_beam(bkd, 4, 1.0, 1.0, _uniform(1.0)))
        dofs = bkd.to_numpy(system.constraint_set().dofs())
        assert sorted(dofs.tolist()) == [0, 1]
        jac_np = _to_dense(
            system.steady().steady_jacobian(bkd.zeros((system.nstates(),)))
        )
        for dof in dofs:
            expected_row = np.zeros(system.nstates())
            expected_row[dof] = 1.0
            bkd.assert_allclose(
                bkd.asarray(jac_np[dof]), bkd.asarray(expected_row), atol=1e-14
            )

    # ------------------------------------------------------------------
    # FEM vs analytical agreement (cubic Hermite is nodally exact)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("load_type", ["uniform", "linear"])
    def test_nodal_values_and_slopes_match_analytical(
        self, numpy_bkd: NumpyBkd, load_type: str
    ) -> None:
        bkd = numpy_bkd
        L, EI_val, q0 = 2.0, 3.0, 1.5
        load = _uniform(q0) if load_type == "uniform" else (lambda x: q0 * x / L)
        beam = _beam(bkd, 8, L, EI_val, load)
        sol = _solve(_cantilever(beam))
        analytical = EulerBernoulliBeamAnalytical(
            length=L, EI=EI_val, q0=q0, bkd=bkd, load_type=load_type
        )
        x = _nodes(beam)
        basis = beam.basis()
        # The clamped DOFs are exactly 0; the solve returns them to
        # round-off (~2e-13 against deflections of order 1).
        bkd.assert_allclose(
            basis.values(sol), analytical.deflection(x), rtol=1e-10, atol=1e-11
        )
        bkd.assert_allclose(
            basis.slopes(sol), analytical.slope(x), rtol=1e-10, atol=1e-11
        )
        bkd.assert_allclose(
            bkd.asarray([_tip(beam, sol)]),
            bkd.asarray([analytical.tip_deflection()]),
            rtol=1e-10,
        )

    # ------------------------------------------------------------------
    # Scaling tests
    # ------------------------------------------------------------------

    def test_different_EI_values(self, numpy_bkd: NumpyBkd) -> None:
        """Tip deflection scales as 1/EI for uniform load."""
        bkd = numpy_bkd
        EI_values = [1.0, 2.0, 4.0]
        products = []
        for EI_val in EI_values:
            beam = _beam(bkd, 10, 1.0, EI_val, _uniform(1.0))
            products.append(_tip(beam, _solve(_cantilever(beam))) * EI_val)
        bkd.assert_allclose(
            bkd.asarray(products[1:]), bkd.asarray(products[:1] * 2), rtol=1e-10
        )

    def test_different_lengths(self, numpy_bkd: NumpyBkd) -> None:
        """Tip deflection scales as L^4 for uniform load."""
        bkd = numpy_bkd
        tips = []
        for L in (1.0, 2.0):
            beam = _beam(bkd, 20, L, 1.0, _uniform(1.0))
            tips.append(_tip(beam, _solve(_cantilever(beam))))
        bkd.assert_allclose(
            bkd.asarray([tips[1] / tips[0]]), bkd.asarray([16.0]), rtol=1e-8
        )


def _smooth_EI_field(x_nodes: np.ndarray, EI_mean: float) -> np.ndarray:
    """Smooth spatially-varying EI: EI(x) = EI_mean * (1 + 0.5*sin(pi*x/L)).

    Returns per-element EI evaluated at element midpoints.
    """
    midpoints = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    L = x_nodes[-1]
    return EI_mean * (1.0 + 0.5 * np.sin(np.pi * midpoints / L))


class TestEulerBernoulliVaryingEI:
    """Tests for Euler-Bernoulli beam with spatially-varying EI(x)."""

    def _make_beam(
        self, bkd: Backend[Array], nx: int = 10, L: float = 1.0,
        EI_mean: float = 1.0,
    ) -> EulerBernoulliBeamFEM[Array]:
        """Create beam with spatially-varying EI from smooth field."""
        beam = _beam(bkd, nx, L, 1.0, _uniform(1.0))
        beam.set_EI(_smooth_EI_field(bkd.to_numpy(_nodes(beam)), EI_mean))
        return beam

    def test_residual_zero_at_solution(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        system = _cantilever(self._make_beam(bkd, nx=10, EI_mean=2.0))
        sol = _solve(system)
        bkd.assert_allclose(
            system.steady().steady_residual(sol),
            bkd.zeros((system.nstates(),)),
            atol=1e-10,
        )

    def test_jacobian_derivative_checker(self, numpy_bkd: NumpyBkd) -> None:
        """Verify Jacobian of residual using DerivativeChecker."""
        bkd = numpy_bkd
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (  # noqa: E501
            DerivativeChecker,
        )
        from pyapprox.interface.functions.fromcallable.jacobian import (
            FunctionWithJacobianFromCallable,
        )

        steady = _cantilever(self._make_beam(bkd, nx=6, EI_mean=2.0)).steady()
        ndofs = steady.nstates()

        def residual_func(samples: Array) -> Array:
            return bkd.reshape(steady.steady_residual(samples[:, 0]), (ndofs, 1))

        def jacobian_func(sample: Array) -> Array:
            return bkd.asarray(_to_dense(steady.steady_jacobian(sample[:, 0])))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=ndofs, nvars=ndofs,
            fun=residual_func, jacobian=jacobian_func,
            bkd=bkd,
        )
        sample = bkd.asarray(
            np.random.default_rng(42).normal(size=(ndofs, 1)) * 0.01
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(sample, relative=True)[0]
        assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 1e-6

    def test_stiffness_symmetric_varying_EI(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        K_np = _to_dense(self._make_beam(bkd, nx=8, EI_mean=3.0).stiffness_matrix())
        bkd.assert_allclose(bkd.asarray(K_np), bkd.asarray(K_np.T), atol=1e-12)

    def test_convergence_varying_EI(self, numpy_bkd: NumpyBkd) -> None:
        """Tip deflection converges under refinement: successive
        differences shrink (piecewise-constant EI converges as O(h^2))."""
        bkd = numpy_bkd
        tip_vals = []
        for nx in [8, 16, 32, 64, 128, 256]:
            beam = self._make_beam(bkd, nx=nx, EI_mean=2.0)
            tip_vals.append(_tip(beam, _solve(_cantilever(beam))))
        diffs = np.abs(np.diff(tip_vals))
        assert np.all(diffs[1:] < diffs[:-1])
        assert diffs[-1] < 5e-6

    def test_constant_field_matches_scalar(self, numpy_bkd: NumpyBkd) -> None:
        """An element-wise EI array with a constant value matches a scalar."""
        bkd = numpy_bkd
        nx, EI_val = 10, 3.0
        sols = [
            _solve(_cantilever(_beam(bkd, nx, 1.0, EI, _uniform(1.0))))
            for EI in (EI_val, np.full(nx, EI_val))
        ]
        bkd.assert_allclose(sols[1], sols[0], rtol=1e-12)

    def test_curvature_positive_and_scales_with_EI(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Curvature is positive under load, and doubling EI halves it."""
        bkd = numpy_bkd
        curvatures = []
        for EI_val in (1.0, 2.0):
            beam = _beam(bkd, 20, 1.0, EI_val, _uniform(1.0))
            sol = _solve(_cantilever(beam))
            curvatures.append(element_curvatures(beam.basis(), sol).max())
        assert curvatures[0] > 0.0
        bkd.assert_allclose(
            bkd.asarray([curvatures[0]]),
            bkd.asarray([2.0 * curvatures[1]]),
            rtol=1e-8,
        )
