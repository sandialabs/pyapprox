"""RobinRows: Robin closure rows alpha u + beta N(u) - g.

2D Chebyshev mesh, 5 x 5 nodes on [-1, 1]^2, four sides with the mesh's
outward normals. The normal term is the gradient operator (grad u . n) or
the flux operator of an ADR physics with D = 1.5 + 0.5 x y. The old path
(``RobinBC.apply_to_*`` in side order, from zero) is the reference.
"""

import pickle
from typing import Callable, Generic, List, Sequence, Tuple, Union

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsProtocol,
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import MappedBoundaryData
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    HyperelasticTractionNormalOperator,
    hyperelastic_traction_robin_bc,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.robin import RobinBC
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField, NodalSupplier
from pyapprox.pde.collocation.physics.hyperelasticity import HyperelasticityPhysics
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.util.backends.protocols import Array, Backend

_N1D = 5
_NPTS = _N1D * _N1D
_NPARAMS = 2
_SIDES = (0, 1, 2, 3)


def _mesh(bkd: Backend[Array]) -> TransformedMesh2D[Array]:
    return TransformedMesh2D(_N1D, _N1D, bkd)


def _normal_term(
    bkd: Backend[Array], side: int, kind: str
) -> Union[GradientNormalOperator[Array], FluxNormalOperator[Array]]:
    mesh = _mesh(bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    indices, normals = mesh.boundary_indices(side), mesh.boundary_normals(side)
    if kind == "gradient":
        derivatives = [basis.derivative_matrix(1, 0), basis.derivative_matrix(1, 1)]
        return GradientNormalOperator(bkd, indices, normals, derivatives)
    x, y = mesh.points()[0], mesh.points()[1]
    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=NodalField(1.5 + 0.5 * x * y)
    )
    return FluxNormalOperator(bkd, indices, normals, physics)


def _side_coordinate(bkd: Backend[Array], side: int) -> Array:
    """The coordinate varying along a side."""
    mesh = _mesh(bkd)
    return mesh.points()[1 if side < 2 else 0, mesh.boundary_indices(side)]


def _growing(coords: Array, time: float) -> Array:
    """A positive diffusion varying in space and time."""
    return (1.0 + time) * (2.0 + coords[0] ** 2)


def _state(bkd: Backend[Array]) -> Array:
    rng = np.random.default_rng(7)
    return bkd.asarray(rng.standard_normal(_NPTS))


def _constant_parts(bkd: Backend[Array], kind: str) -> List[RobinPart[Array]]:
    mesh = _mesh(bkd)
    return [
        RobinPart(
            mesh.boundary_indices(side),
            _normal_term(bkd, side, kind),
            2.0 + side,
            0.5,
            NodalField(_side_coordinate(bkd, side) ** 2 - 0.3 * side),
        )
        for side in _SIDES
    ]


class TestRobinRows:
    @pytest.mark.parametrize("kind", ["gradient", "flux"])
    def test_equals_old_robin_rows(self, bkd: Backend[Array], kind: str) -> None:
        """c and c_u equal the old RobinBC rows applied in side order."""
        mesh = _mesh(bkd)
        state = _state(bkd)
        old_residual, old_jacobian = bkd.zeros((_NPTS,)), bkd.zeros((_NPTS, _NPTS))
        for side in _SIDES:
            old = RobinBC(
                bkd,
                mesh.boundary_indices(side),
                _normal_term(bkd, side, kind),
                2.0 + side,
                0.5,
                _side_coordinate(bkd, side) ** 2 - 0.3 * side,
            )
            old_residual = old.apply_to_residual(old_residual, state, 0.0)
            old_jacobian = old.apply_to_jacobian(old_jacobian, state, 0.0)
        rows = RobinRows(bkd, StateLayout.uniform(_NPTS, 1), _constant_parts(bkd, kind))
        assert isinstance(rows, ClosureRowsProtocol)
        bkd.assert_allclose(
            rows.residual(state, 0.0), old_residual, rtol=1e-14, atol=1e-14
        )
        bkd.assert_allclose(
            rows.jacobian(state, 0.0), old_jacobian, rtol=1e-14, atol=1e-14
        )

    @pytest.mark.parametrize("kind", ["gradient", "flux"])
    def test_state_jacobian(self, bkd: Backend[Array], kind: str) -> None:
        """c_u passes DerivativeChecker."""
        rows = RobinRows(bkd, StateLayout.uniform(_NPTS, 1), _constant_parts(bkd, kind))
        function = _VectorFunction(
            bkd,
            _NPTS,
            _NPTS,
            lambda u: rows.residual(u, 0.0),
            lambda u: rows.jacobian(u, 0.0),
        )
        assert _check(function, _state(bkd), bkd) <= 1e-6

    def test_neumann_and_dirichlet_limits(self, bkd: Backend[Array]) -> None:
        """alpha = 0 reads N(u) - g exactly; beta = 0, alpha = 1 reads
        u - g with identity Jacobian rows."""
        mesh = _mesh(bkd)
        indices = mesh.boundary_indices(0)
        normal = _normal_term(bkd, 0, "flux")
        g = NodalField(_side_coordinate(bkd, 0))
        state = _state(bkd)
        neumann = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [RobinPart(indices, normal, 0.0, 1.0, g)],
        )
        bkd.assert_allclose(
            neumann.residual(state, 0.0)[indices],
            normal(state, 0.0) - g.values(0.0),
            rtol=0.0,
            atol=0.0,
        )
        dirichlet = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [RobinPart(indices, normal, 1.0, 0.0, g)],
        )
        bkd.assert_allclose(
            dirichlet.residual(state, 0.0)[indices],
            state[indices] - g.values(0.0),
            rtol=0.0,
            atol=0.0,
        )
        bkd.assert_allclose(
            dirichlet.jacobian(state, 0.0)[indices],
            bkd.eye(_NPTS)[indices],
            rtol=0.0,
            atol=0.0,
        )

    def test_exact_solution_closes_the_rows(self, bkd: Backend[Array]) -> None:
        """u = x y + x^2 on side 0 (x = -1, n = (-1, 0)): grad u . n =
        2 - y, so g = alpha (1 - y) + beta (2 - y) gives c = 0; alpha of
        the opposite sign does not."""
        mesh = _mesh(bkd)
        x, y = mesh.points()[0], mesh.points()[1]
        indices = mesh.boundary_indices(0)
        y_side = y[indices]
        alpha, beta = 2.0, 0.7
        g = NodalField(alpha * (1.0 - y_side) + beta * (2.0 - y_side))
        normal = _normal_term(bkd, 0, "gradient")
        state = x * y + x**2
        right = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [RobinPart(indices, normal, alpha, beta, g)],
        )
        wrong = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [RobinPart(indices, normal, -alpha, beta, g)],
        )
        assert bkd.to_float(bkd.max(bkd.abs(right.residual(state, 0.0)))) < 1e-12
        assert bkd.to_float(bkd.max(bkd.abs(wrong.residual(state, 0.0)))) > 1e-2

    @pytest.mark.parametrize("mapped_last", [True, False])
    def test_param_jacobian(self, bkd: Backend[Array], mapped_last: bool) -> None:
        """dc/dp with alpha, beta and g on side 0 depending on the
        parameters passes DerivativeChecker, whichever side wins the
        corners. The error ratio bottoms out near 1.3e-6 on both backends,
        so the bound is 1e-5; dc/dp with the wrong sign stays above 1e-2."""
        rows, data = _mapped_rows(bkd, mapped_last)
        jacobian_fn = rows.derivatives().param_jacobian
        assert callable(jacobian_fn)
        state = _state(bkd)

        def fun(params: Array) -> Array:
            _set(data, params)
            return rows.residual(state, 0.0)

        def jac(params: Array) -> Array:
            _set(data, params)
            return jacobian_fn(state, 0.0)

        def wrong(params: Array) -> Array:
            return -jac(params)

        params = bkd.reshape(_params(bkd), (_NPARAMS, 1))
        right_fn = _VectorFunction(bkd, _NPARAMS, _NPTS, fun, jac)
        wrong_fn = _VectorFunction(bkd, _NPARAMS, _NPTS, fun, wrong)
        assert _check(right_fn, params, bkd) <= 1e-5
        assert _check(wrong_fn, params, bkd) > 1e-2

    @pytest.mark.parametrize("mapped_last", [True, False])
    def test_mixed(self, bkd: Backend[Array], mapped_last: bool) -> None:
        """sum_i w_i d2c_i/du dp is the Jacobian of p -> c_u^T w, by
        DerivativeChecker, with weights on every replaced row (the shared
        corners included); a mixed term 10% off stays above 1e-2."""
        rows, data = _mapped_rows(bkd, mapped_last)
        mixed_fn = rows.derivatives().mixed
        assert callable(mixed_fn)
        state, weights = _state(bkd), _row_weights(bkd, rows)

        def fun(params: Array) -> Array:
            _set(data, params)
            return rows.jacobian(state, 0.0).T @ weights

        def jac(params: Array) -> Array:
            _set(data, params)
            return mixed_fn(state, 0.0, weights)

        def wrong(params: Array) -> Array:
            return 1.1 * jac(params)

        params = bkd.reshape(_params(bkd), (_NPARAMS, 1))
        assert (
            _check(_VectorFunction(bkd, _NPARAMS, _NPTS, fun, jac), params, bkd) <= 1e-5
        )
        assert (
            _check(_VectorFunction(bkd, _NPARAMS, _NPTS, fun, wrong), params, bkd)
            > 1e-2
        )

    @pytest.mark.parametrize("mapped_last", [True, False])
    def test_param_curvature(self, bkd: Backend[Array], mapped_last: bool) -> None:
        """sum_i w_i d2c_i/dp2 is the Jacobian of p -> c_p^T w, by
        DerivativeChecker; a curvature 10% off stays above 1e-2."""
        rows, data = _mapped_rows(bkd, mapped_last)
        bundle = rows.derivatives()
        jacobian_fn, curvature_fn = bundle.param_jacobian, bundle.param_curvature
        assert callable(jacobian_fn) and callable(curvature_fn)
        state, weights = _state(bkd), _row_weights(bkd, rows)

        def fun(params: Array) -> Array:
            _set(data, params)
            return jacobian_fn(state, 0.0).T @ weights

        def jac(params: Array) -> Array:
            _set(data, params)
            return curvature_fn(state, 0.0, weights)

        def wrong(params: Array) -> Array:
            return 1.1 * jac(params)

        params = bkd.reshape(_params(bkd), (_NPARAMS, 1))
        assert (
            _check(_VectorFunction(bkd, _NPARAMS, _NPARAMS, fun, jac), params, bkd)
            <= 1e-5
        )
        assert (
            _check(_VectorFunction(bkd, _NPARAMS, _NPARAMS, fun, wrong), params, bkd)
            > 1e-2
        )

    def test_declared_derivatives(self, bkd: Backend[Array]) -> None:
        """Gradient rows: c_uu, c_p, c_up, c_pp stated zero with nothing
        parameterized. Flux rows: c_uu not known and every parameter
        derivative left to the system. c_up stated zero when only g
        depends on the parameters; c_pp absent when a map has no second
        derivative."""
        gradient = RobinRows(
            bkd, StateLayout.uniform(_NPTS, 1), _constant_parts(bkd, "gradient")
        )
        bundle = gradient.derivatives()
        assert bundle.state_curvature is STATED_ZERO
        assert bundle.param_jacobian is STATED_ZERO
        assert bundle.mixed is STATED_ZERO
        assert bundle.param_curvature is STATED_ZERO
        flux = RobinRows(
            bkd, StateLayout.uniform(_NPTS, 1), _constant_parts(bkd, "flux")
        ).derivatives()
        assert flux.state_curvature is None
        assert flux.param_jacobian is None
        assert flux.mixed is None and flux.param_curvature is None
        mesh = _mesh(bkd)
        g_only = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 1),
            [
                RobinPart(
                    mesh.boundary_indices(0),
                    _normal_term(bkd, 0, "gradient"),
                    1.0,
                    1.0,
                    _mapped(bkd, 1.0),
                )
            ],
        ).derivatives()
        assert callable(g_only.param_jacobian)
        assert g_only.mixed is STATED_ZERO
        assert callable(g_only.param_curvature)
        first_order, _ = _mapped_rows(bkd, True, second_derivative=False)
        assert first_order.derivatives().param_curvature is None
        assert callable(first_order.derivatives().mixed)

    def test_time_invariance(self, bkd: Backend[Array]) -> None:
        """Constant data: gradient rows are time-invariant, and so are flux
        rows over a steady diffusion; flux rows over a diffusion declared
        time-dependent are not."""
        assert RobinRows(
            bkd, StateLayout.uniform(_NPTS, 1), _constant_parts(bkd, "gradient")
        ).is_time_invariant()
        assert RobinRows(
            bkd, StateLayout.uniform(_NPTS, 1), _constant_parts(bkd, "flux")
        ).is_time_invariant()
        mesh = _mesh(bkd)
        supplier: TimeDependent[Array] = TimeDependent(_growing)
        physics = AdvectionDiffusionReaction(
            ChebyshevBasis2D(mesh, bkd),
            bkd,
            diffusion=NodalSupplier(supplier, mesh.points()),
        )
        part = RobinPart(
            mesh.boundary_indices(0),
            FluxNormalOperator(
                bkd, mesh.boundary_indices(0), mesh.boundary_normals(0), physics
            ),
            1.0,
            1.0,
            0.0,
        )
        assert not RobinRows(
            bkd, StateLayout.uniform(_NPTS, 1), [part]
        ).is_time_invariant()

    def test_refusals(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        indices = mesh.boundary_indices(0)
        normal = _normal_term(bkd, 0, "gradient")
        with pytest.raises(ValueError, match="at least one part"):
            RobinRows(bkd, StateLayout.uniform(_NPTS, 1), [])
        with pytest.raises(ValueError, match="component"):
            RobinRows(
                bkd,
                StateLayout.uniform(_NPTS, 1),
                [RobinPart(indices, normal, 1.0, 1.0, 0.0, 1)],
            )
        with pytest.raises(TypeError, match="RobinNormalTermProtocol"):
            RobinRows(
                bkd,
                StateLayout.uniform(_NPTS, 1),
                [RobinPart(indices, 1.0, 1.0, 1.0, 0.0)],  # type: ignore[arg-type]
            )
        three = MappedBoundaryData(
            BasisExpansion(bkd, 0.0, [indices * 0.0 + 1.0] * 3), bkd.zeros((3,)), bkd
        )
        with pytest.raises(ValueError, match="number of parameters"):
            RobinRows(
                bkd,
                StateLayout.uniform(_NPTS, 1),
                [RobinPart(indices, normal, _mapped(bkd, 1.0), three, 0.0)],
            )
        # A scalar normal term on a two-component state has the wrong width
        two = RobinRows(
            bkd,
            StateLayout.uniform(_NPTS, 2),
            [RobinPart(indices, normal, 1.0, 1.0, 0.0)],
        )
        with pytest.raises(ValueError):
            two.jacobian(bkd.zeros((2 * _NPTS,)), 0.0)
        with pytest.raises(ValueError, match="state must have shape"):
            two.residual(bkd.zeros((_NPTS,)), 0.0)

    def test_pickle_round_trip(self, bkd: Backend[Array]) -> None:
        rows = RobinRows(
            bkd, StateLayout.uniform(_NPTS, 1), _constant_parts(bkd, "flux")
        )
        loaded = pickle.loads(pickle.dumps(rows))
        state = _state(bkd)
        bkd.assert_allclose(
            loaded.residual(state, 0.0), rows.residual(state, 0.0), rtol=1e-14
        )


class TestHyperelasticTractionRows:
    """Hyperelastic traction rows are RobinRows over the PK1 traction
    operator: neo-Hookean, lambda = 2, mu = 1.5, 6 x 6 mesh, sides 0 and
    2 (sharing a corner) on both displacement components, alpha = 0.5,
    beta = 1, small random displacements (scale 0.01)."""

    _N1D_TRACTION = 6

    def _setup(
        self, bkd: Backend[Array]
    ) -> Tuple[HyperelasticityPhysics[Array], TransformedMesh2D[Array], int]:
        mesh = TransformedMesh2D(self._N1D_TRACTION, self._N1D_TRACTION, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        physics = HyperelasticityPhysics(
            basis, bkd, NeoHookeanLaw(bkd), mu=1.5, lamda=2.0
        )
        return physics, mesh, basis.npts()

    def _rows(self, bkd: Backend[Array]) -> Tuple[RobinRows[Array], int]:
        physics, mesh, npts = self._setup(bkd)
        parts = [
            RobinPart(
                mesh.boundary_indices(side),
                HyperelasticTractionNormalOperator(
                    bkd,
                    mesh.boundary_indices(side),
                    mesh.boundary_normals(side),
                    physics,
                    component,
                ),
                0.5,
                1.0,
                0.1 * (component + 1),
                component,
            )
            for side in (0, 2)
            for component in (0, 1)
        ]
        return RobinRows(bkd, StateLayout.uniform(npts, 2), parts), npts

    def _displacement(self, bkd: Backend[Array], npts: int) -> Array:
        rng = np.random.default_rng(42)
        return bkd.asarray(0.01 * rng.standard_normal(2 * npts))

    def test_equals_old_traction_rows(self, bkd: Backend[Array]) -> None:
        """c and c_u equal the old hyperelastic traction Robin rows in the
        same order, to 1e-14."""
        physics, mesh, npts = self._setup(bkd)
        rows, _ = self._rows(bkd)
        state = self._displacement(bkd, npts)
        old_residual = bkd.zeros((2 * npts,))
        old_jacobian = bkd.zeros((2 * npts, 2 * npts))
        for side in (0, 2):
            for component in (0, 1):
                old = hyperelastic_traction_robin_bc(
                    bkd,
                    mesh.boundary_indices(side),
                    mesh.boundary_normals(side),
                    physics,
                    npts,
                    component,
                    0.5,
                    1.0,
                    0.1 * (component + 1),
                )
                old_residual = old.apply_to_residual(old_residual, state, 0.0)
                old_jacobian = old.apply_to_jacobian(old_jacobian, state, 0.0)
        bkd.assert_allclose(
            rows.residual(state, 0.0), old_residual, rtol=1e-14, atol=1e-15
        )
        bkd.assert_allclose(
            rows.jacobian(state, 0.0), old_jacobian, rtol=1e-14, atol=1e-15
        )

    def test_state_jacobian(self, bkd: Backend[Array]) -> None:
        """c_u of the nonlinear traction rows passes DerivativeChecker."""
        rows, npts = self._rows(bkd)
        function = _VectorFunction(
            bkd,
            2 * npts,
            2 * npts,
            lambda u: rows.residual(u, 0.0),
            lambda u: rows.jacobian(u, 0.0),
        )
        assert _check(function, self._displacement(bkd, npts), bkd) <= 1e-5

    def test_declared_derivatives(self, bkd: Backend[Array]) -> None:
        """The PK1 law supplies no second derivative, so c_uu is absent;
        the Lame dependence is left to the system, so every parameter
        field is absent; the law takes no time and the data are constant,
        so the rows are time-invariant."""
        rows, _ = self._rows(bkd)
        bundle = rows.derivatives()
        assert bundle.state_curvature is None
        assert bundle.param_jacobian is None
        assert bundle.mixed is None and bundle.param_curvature is None
        assert rows.is_time_invariant()


# -- helpers -----------------------------------------------------------------


class _Exp(Generic[Array]):
    """exp as transform and both its derivatives (picklable)."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, x: Array) -> Array:
        return self._bkd.exp(x)


def _mapped(
    bkd: Backend[Array], scale: float, second_derivative: bool = True
) -> MappedBoundaryData[Array]:
    """scale * exp(0.1 + p0 y / 2 + p1 (0.3 y^2 - 0.1)) on side 0, written
    as exp(log(scale) + 0.1 + ...)."""
    y = _side_coordinate(bkd, 0)
    inner = BasisExpansion(bkd, 0.1 + float(np.log(scale)), [0.5 * y, 0.3 * y**2 - 0.1])
    exp = _Exp(bkd)
    field_map = TransformedFieldMap(
        inner, exp, exp, bkd, transform_deriv2=exp if second_derivative else None
    )
    return MappedBoundaryData(field_map, _params(bkd), bkd)


def _mapped_rows(
    bkd: Backend[Array], mapped_last: bool, second_derivative: bool = True
) -> Tuple[RobinRows[Array], List[MappedBoundaryData[Array]]]:
    """Side 0 with alpha, beta and g as exp maps of two parameters (scales
    1, 0.5, 2), the other sides constant, side 0 listed last or first."""
    mesh = _mesh(bkd)
    alpha = _mapped(bkd, 1.0)
    beta = _mapped(bkd, 0.5, second_derivative)
    g = _mapped(bkd, 2.0)
    mapped = RobinPart(
        mesh.boundary_indices(0), _normal_term(bkd, 0, "gradient"), alpha, beta, g
    )
    others = _constant_parts(bkd, "gradient")[1:]
    parts = others + [mapped] if mapped_last else [mapped] + others
    return RobinRows(bkd, StateLayout.uniform(_NPTS, 1), parts), [alpha, beta, g]


def _set(data: Sequence[MappedBoundaryData[Array]], params: Array) -> None:
    for coefficient in data:
        coefficient.set_parameters(params)


def _row_weights(bkd: Backend[Array], rows: RobinRows[Array]) -> Array:
    """Random weights on the replaced rows, zero elsewhere. Shape:
    (nstates,)"""
    rng = np.random.default_rng(11)
    column = rows.rows().replace_rows(
        bkd.zeros((_NPTS, 1)), bkd.asarray(rng.standard_normal((_NPTS, 1)))
    )
    return column[:, 0]


def _params(bkd: Backend[Array]) -> Array:
    return bkd.asarray([0.4, -0.7])


class _VectorFunction(Generic[Array]):
    """A vector function with its Jacobian, for DerivativeChecker."""

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        nqoi: int,
        fun: Callable[[Array], Array],
        jac: Callable[[Array], Array],
    ) -> None:
        self._bkd, self._nvars, self._nqoi = bkd, nvars, nqoi
        self._fun, self._jac = fun, jac

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        return self._bkd.stack(
            [self._fun(samples[:, k]) for k in range(samples.shape[1])], axis=1
        )

    def jacobian(self, sample: Array) -> Array:
        return self._jac(sample[:, 0])

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


def _check(
    function: _VectorFunction[Array], sample: Array, bkd: Backend[Array]
) -> float:
    if sample.ndim == 1:
        sample = bkd.reshape(sample, (sample.shape[0], 1))
    checker: DerivativeChecker[Array] = DerivativeChecker(function)
    errors = checker.check_derivatives(sample, verbosity=0)
    return bkd.to_float(checker.error_ratio(errors[0]))
