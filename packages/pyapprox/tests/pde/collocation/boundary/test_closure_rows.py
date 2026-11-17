"""RobinRows: Robin closure rows alpha u + beta N(u) - g.

2D Chebyshev mesh, 5 x 5 nodes on [-1, 1]^2, four sides with the mesh's
outward normals. The normal term is the gradient operator (grad u . n) or
the flux operator of an ADR physics with D = 1.5 + 0.5 x y. The old path
(``RobinBC.apply_to_*`` in side order, from zero) is the reference.
"""

import pickle
from typing import Callable, Generic, List, Union

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
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.robin import RobinBC
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
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
        rows = RobinRows(bkd, _NPTS, 1, _constant_parts(bkd, kind))
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
        rows = RobinRows(bkd, _NPTS, 1, _constant_parts(bkd, kind))
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
        neumann = RobinRows(bkd, _NPTS, 1, [RobinPart(indices, normal, 0.0, 1.0, g)])
        bkd.assert_allclose(
            neumann.residual(state, 0.0)[indices],
            normal(state, 0.0) - g.values(0.0),
            rtol=0.0,
            atol=0.0,
        )
        dirichlet = RobinRows(bkd, _NPTS, 1, [RobinPart(indices, normal, 1.0, 0.0, g)])
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
        right = RobinRows(bkd, _NPTS, 1, [RobinPart(indices, normal, alpha, beta, g)])
        wrong = RobinRows(bkd, _NPTS, 1, [RobinPart(indices, normal, -alpha, beta, g)])
        assert bkd.to_float(bkd.max(bkd.abs(right.residual(state, 0.0)))) < 1e-12
        assert bkd.to_float(bkd.max(bkd.abs(wrong.residual(state, 0.0)))) > 1e-2

    @pytest.mark.parametrize("mapped_last", [True, False])
    def test_param_jacobian(self, bkd: Backend[Array], mapped_last: bool) -> None:
        """dc/dp with alpha, beta and g on side 0 depending on the
        parameters passes DerivativeChecker, whichever side wins the
        corners. The error ratio bottoms out near 1.3e-6 on both backends,
        so the bound is 1e-5; dc/dp with the wrong sign stays above 1e-2."""
        mesh = _mesh(bkd)
        alpha, beta, g = (_mapped(bkd, scale) for scale in (1.0, 0.5, 2.0))
        mapped = RobinPart(
            mesh.boundary_indices(0), _normal_term(bkd, 0, "gradient"), alpha, beta, g
        )
        others = _constant_parts(bkd, "gradient")[1:]
        parts = others + [mapped] if mapped_last else [mapped] + others
        rows = RobinRows(bkd, _NPTS, 1, parts)
        jacobian_fn = rows.parameter_derivatives().jacobian
        assert callable(jacobian_fn)
        state = _state(bkd)

        def fun(params: Array) -> Array:
            for data in (alpha, beta, g):
                data.set_parameters(params)
            return rows.residual(state, 0.0)

        def jac(params: Array) -> Array:
            for data in (alpha, beta, g):
                data.set_parameters(params)
            return jacobian_fn(state, 0.0)

        def wrong(params: Array) -> Array:
            return -jac(params)

        params = bkd.reshape(_params(bkd), (_NPARAMS, 1))
        right_fn = _VectorFunction(bkd, _NPARAMS, _NPTS, fun, jac)
        wrong_fn = _VectorFunction(bkd, _NPARAMS, _NPTS, fun, wrong)
        assert _check(right_fn, params, bkd) <= 1e-5
        assert _check(wrong_fn, params, bkd) > 1e-2

    def test_declared_parameter_dependence(self, bkd: Backend[Array]) -> None:
        """Stated zero with nothing parameterized; not available when a
        normal term reads a physics coefficient."""
        gradient = RobinRows(bkd, _NPTS, 1, _constant_parts(bkd, "gradient"))
        assert gradient.parameter_derivatives().jacobian is STATED_ZERO
        flux = RobinRows(bkd, _NPTS, 1, _constant_parts(bkd, "flux"))
        assert flux.parameter_derivatives().jacobian is None

    def test_time_invariance(self, bkd: Backend[Array]) -> None:
        assert RobinRows(
            bkd, _NPTS, 1, _constant_parts(bkd, "gradient")
        ).is_time_invariant()
        assert not RobinRows(
            bkd, _NPTS, 1, _constant_parts(bkd, "flux")
        ).is_time_invariant()

    def test_refusals(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        indices = mesh.boundary_indices(0)
        normal = _normal_term(bkd, 0, "gradient")
        with pytest.raises(ValueError, match="at least one part"):
            RobinRows(bkd, _NPTS, 1, [])
        with pytest.raises(ValueError, match="component"):
            RobinRows(bkd, _NPTS, 1, [RobinPart(indices, normal, 1.0, 1.0, 0.0, 1)])
        with pytest.raises(TypeError, match="RobinNormalTermProtocol"):
            RobinRows(
                bkd,
                _NPTS,
                1,
                [RobinPart(indices, 1.0, 1.0, 1.0, 0.0)],  # type: ignore[arg-type]
            )
        three = MappedBoundaryData(
            BasisExpansion(bkd, 0.0, [indices * 0.0 + 1.0] * 3), bkd.zeros((3,)), bkd
        )
        with pytest.raises(ValueError, match="number of parameters"):
            RobinRows(
                bkd,
                _NPTS,
                1,
                [RobinPart(indices, normal, _mapped(bkd, 1.0), three, 0.0)],
            )
        # A scalar normal term on a two-component state has the wrong width
        two = RobinRows(bkd, _NPTS, 2, [RobinPart(indices, normal, 1.0, 1.0, 0.0)])
        with pytest.raises(ValueError):
            two.jacobian(bkd.zeros((2 * _NPTS,)), 0.0)
        with pytest.raises(ValueError, match="state must have shape"):
            two.residual(bkd.zeros((_NPTS,)), 0.0)

    def test_pickle_round_trip(self, bkd: Backend[Array]) -> None:
        rows = RobinRows(bkd, _NPTS, 1, _constant_parts(bkd, "flux"))
        loaded = pickle.loads(pickle.dumps(rows))
        state = _state(bkd)
        bkd.assert_allclose(
            loaded.residual(state, 0.0), rows.residual(state, 0.0), rtol=1e-14
        )


# -- helpers -----------------------------------------------------------------


class _Exp(Generic[Array]):
    """exp as transform and both its derivatives (picklable)."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, x: Array) -> Array:
        return self._bkd.exp(x)


def _mapped(bkd: Backend[Array], scale: float) -> MappedBoundaryData[Array]:
    """scale * exp(0.1 + p0 y / 2 + p1 (0.3 y^2 - 0.1)) on side 0, written
    as exp(log(scale) + 0.1 + ...)."""
    y = _side_coordinate(bkd, 0)
    inner = BasisExpansion(bkd, 0.1 + float(np.log(scale)), [0.5 * y, 0.3 * y**2 - 0.1])
    exp = _Exp(bkd)
    field_map = TransformedFieldMap(inner, exp, exp, bkd, transform_deriv2=exp)
    return MappedBoundaryData(field_map, _params(bkd), bkd)


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
