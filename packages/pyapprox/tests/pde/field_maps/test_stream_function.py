"""Tests for the stream-function velocity map and the vector layouts.

The divergence is checked twice, separately. The field itself is
divergence-free: autograd through the map's coordinates gives
du/dx + dv/dy = 0 to rounding, with no mesh. The finite element
interpolant of it is not: a continuous P1 vector field cannot represent a
general divergence-free field, and the divergence of the interpolant
converges at O(h).
"""

import math
from typing import Any, List

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.field_maps.protocol import (
    FieldMapWithHVPProtocol,
    LinearFieldMapProtocol,
)
from pyapprox.pde.field_maps.stream_function import (
    StreamFunctionProtocol,
    StreamFunctionVelocityMap,
)
from pyapprox.pde.field_maps.vector_layout import (
    BlockedLayout,
    InterleavedLayout,
    VectorFieldLayoutProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available

_NPSI = 6


class _CellsPlusUniform:
    """psi = sin(pi x) sin(pi y) + y: a cell flow plus the uniform (1, 0)."""

    def __init__(self, bkd: Backend[Any]) -> None:
        self._bkd = bkd

    def gradient(self, points: Any) -> Any:
        bkd, x, y = self._bkd, points[0], points[1]
        return bkd.stack(
            [
                math.pi * bkd.cos(math.pi * x) * bkd.sin(math.pi * y),
                math.pi * bkd.sin(math.pi * x) * bkd.cos(math.pi * y) + 1.0,
            ],
            axis=0,
        )


class _Uniform:
    """psi = U y: the uniform flow (U, 0)."""

    def __init__(self, bkd: Backend[Any], speed: float) -> None:
        self._bkd, self._speed = bkd, speed

    def gradient(self, points: Any) -> Any:
        npts = points.shape[1]
        return self._bkd.stack(
            [self._bkd.zeros((npts,)), self._bkd.full((npts,), self._speed)],
            axis=0,
        )


def _spectrum(bkd: Backend[Array], npsi: int = _NPSI) -> Array:
    """lambda_ij = 1.5^2 / (i^2 + j^2)^2.5."""
    i = np.arange(1, npsi + 1)
    return bkd.asarray(1.5**2 / np.add.outer(i**2, i**2) ** 2.5)


def _points(bkd: Backend[Array], npts: int = 9) -> Array:
    rng = np.random.default_rng(1)
    return bkd.asarray(rng.uniform(0.05, 0.95, (2, npts)))


def _map(
    bkd: Backend[Array], points: Array, layout: Any = None
) -> StreamFunctionVelocityMap[Array]:
    return StreamFunctionVelocityMap(
        bkd,
        points,
        _spectrum(bkd),
        _CellsPlusUniform(bkd),
        BlockedLayout(bkd) if layout is None else layout,
    )


def _eta(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.default_rng(0).standard_normal(_NPSI**2))


class TestVectorLayouts:
    def test_interleaved_and_blocked_orders(self, bkd: Backend[Array]) -> None:
        components = bkd.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        interleaved = InterleavedLayout(bkd)
        blocked = BlockedLayout(bkd)
        assert isinstance(interleaved, VectorFieldLayoutProtocol)
        assert isinstance(blocked, VectorFieldLayoutProtocol)
        bkd.assert_allclose(
            interleaved.flatten(components),
            bkd.asarray([1.0, 10.0, 2.0, 20.0, 3.0, 30.0]),
        )
        bkd.assert_allclose(
            blocked.flatten(components),
            bkd.asarray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
        )

    def test_trailing_axes_are_carried(self, bkd: Backend[Array]) -> None:
        """Flattening a Jacobian flattens each of its columns alike."""
        components = bkd.asarray(np.arange(24.0).reshape(2, 3, 4))
        for layout in (InterleavedLayout(bkd), BlockedLayout(bkd)):
            flat = layout.flatten(components)
            assert tuple(flat.shape) == (6, 4)
            for col in range(4):
                bkd.assert_allclose(
                    flat[:, col], layout.flatten(components[:, :, col])
                )


class TestStreamFunctionVelocityMap:
    def test_protocols_and_shapes(self, bkd: Backend[Array]) -> None:
        points = _points(bkd)
        field_map = _map(bkd, points)
        assert isinstance(field_map, FieldMapWithHVPProtocol)
        assert isinstance(field_map, LinearFieldMapProtocol)
        assert field_map.is_linear()
        assert field_map.nvars() == _NPSI**2
        assert tuple(field_map(_eta(bkd)).shape) == (2 * points.shape[1],)
        bkd.assert_allclose(
            field_map.hvp(_eta(bkd), _eta(bkd), _eta(bkd)),
            bkd.zeros((_NPSI**2,)),
        )

    def test_mode_matches_formula(self, bkd: Backend[Array]) -> None:
        """Column of mode (i, j) = (2, 3) is sqrt(lam) (j pi sin(i pi x)
        cos(j pi y), -i pi cos(i pi x) sin(j pi y))."""
        points = _points(bkd)
        field_map = _map(bkd, points)
        i, j = 2, 3
        column = field_map.jacobian(_eta(bkd))[:, (i - 1) * _NPSI + (j - 1)]
        x, y = points[0], points[1]
        amplitude = math.sqrt(1.5**2 / (i**2 + j**2) ** 2.5)
        u = amplitude * j * math.pi * bkd.sin(i * math.pi * x) * bkd.cos(
            j * math.pi * y
        )
        v = -amplitude * i * math.pi * bkd.cos(i * math.pi * x) * bkd.sin(
            j * math.pi * y
        )
        npts = points.shape[1]
        bkd.assert_allclose(column[:npts], u, rtol=1e-12, atol=1e-14)
        bkd.assert_allclose(column[npts:], v, rtol=1e-12, atol=1e-14)

    def test_mean_is_perpendicular_gradient(self, bkd: Backend[Array]) -> None:
        """psi = U y gives the uniform flow (U, 0)."""
        points = _points(bkd)
        field_map = StreamFunctionVelocityMap(
            bkd, points, _spectrum(bkd), _Uniform(bkd, 2.5), BlockedLayout(bkd)
        )
        npts = points.shape[1]
        bkd.assert_allclose(
            field_map(bkd.zeros((_NPSI**2,))),
            bkd.concatenate([bkd.full((npts,), 2.5), bkd.zeros((npts,))]),
        )

    def test_layout_only_reorders(self, bkd: Backend[Array]) -> None:
        points = _points(bkd)
        blocked = _map(bkd, points, BlockedLayout(bkd))(_eta(bkd))
        interleaved = _map(bkd, points, InterleavedLayout(bkd))(_eta(bkd))
        npts = points.shape[1]
        bkd.assert_allclose(interleaved[0::2], blocked[:npts])
        bkd.assert_allclose(interleaved[1::2], blocked[npts:])

    def test_divergence_free_by_central_differences(
        self, bkd: Backend[Array]
    ) -> None:
        """Mesh-free: central differences of the field at shifted points.
        Their error is O(h^2) times third derivatives of psi, which reach
        about (n_psi pi)^3 sqrt(lam); with h = 1e-4 the bound is 1e-4."""
        points = _points(bkd)
        npts = points.shape[1]
        eta = _eta(bkd)
        step = 1e-4

        def shifted(axis: int, sign: float) -> Array:
            offset = bkd.zeros((2, 1))
            offset[axis, 0] = sign * step
            return _map(bkd, points + offset)(eta)

        du_dx = (shifted(0, 1.0)[:npts] - shifted(0, -1.0)[:npts]) / (2 * step)
        dv_dy = (shifted(1, 1.0)[npts:] - shifted(1, -1.0)[npts:]) / (2 * step)
        bkd.assert_allclose(du_dx + dv_dy, bkd.zeros((npts,)), atol=1e-4)

    def test_jacobian_passes_derivative_check(
        self, bkd: Backend[Array]
    ) -> None:
        field_map = _map(bkd, _points(bkd))
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=field_map(_eta(bkd)).shape[0],
            nvars=field_map.nvars(),
            fun=lambda samples: bkd.stack(
                [field_map(samples[:, ii]) for ii in range(samples.shape[1])],
                axis=1,
            ),
            jacobian=lambda sample: field_map.jacobian(sample[:, 0]),
            bkd=bkd,
        )
        errors = DerivativeChecker(wrapper).check_derivatives(
            _eta(bkd)[:, None]
        )[0]
        assert float(bkd.min(errors) / bkd.max(errors)) <= 1e-6

    def test_rejects_bad_inputs(self, numpy_bkd: Backend[Array]) -> None:
        bkd = numpy_bkd
        points = _points(bkd)
        layout = BlockedLayout(bkd)
        mean = _CellsPlusUniform(bkd)
        with pytest.raises(ValueError, match="points"):
            StreamFunctionVelocityMap(
                bkd, bkd.zeros((3, 4)), _spectrum(bkd), mean, layout
            )
        with pytest.raises(ValueError, match="square"):
            StreamFunctionVelocityMap(
                bkd, points, bkd.zeros((2, 3)), mean, layout
            )
        with pytest.raises(ValueError, match="variances"):
            StreamFunctionVelocityMap(
                bkd, points, -bkd.ones((2, 2)), mean, layout
            )
        with pytest.raises(TypeError, match="StreamFunctionProtocol"):
            StreamFunctionVelocityMap(
                bkd, points, _spectrum(bkd), object(), layout  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="VectorFieldLayoutProtocol"):
            StreamFunctionVelocityMap(
                bkd, points, _spectrum(bkd), mean, object()  # type: ignore[arg-type]
            )


def test_divergence_free_by_autograd(torch_bkd: Backend[Array]) -> None:
    """Exact and mesh-free: differentiate the field with respect to the
    point coordinates; du/dx + dv/dy vanishes to rounding."""
    import torch

    bkd = torch_bkd
    points = _points(bkd).clone().requires_grad_(True)
    assert isinstance(_CellsPlusUniform(bkd), StreamFunctionProtocol)
    velocity = _map(bkd, points)(_eta(bkd))
    npts = points.shape[1]
    # Each point's velocity depends only on that point, so the gradient
    # of the summed component holds every pointwise derivative.
    (grad_u,) = torch.autograd.grad(velocity[:npts].sum(), points, retain_graph=True)
    (grad_v,) = torch.autograd.grad(velocity[npts:].sum(), points)
    divergence = grad_u[0] + grad_v[1]
    scale = float(torch.max(torch.abs(grad_u[0])))
    assert scale > 1.0
    bkd.assert_allclose(
        divergence.detach(), bkd.zeros((npts,)), atol=1e-12 * scale
    )


_DIVERGENCE_L2: List[Any] = [
    (10, 3.5535),
    (20, 1.8951),
    (40, 0.9636),
    pytest.param(80, 0.4839, marks=pytest.mark.slow_on("*")),
]


@pytest.mark.skipif(not package_available("skfem"), reason="skfem")
@pytest.mark.parametrize("nelem,expected", _DIVERGENCE_L2)
def test_p1_interpolant_divergence_converges_at_first_order(
    numpy_bkd: Backend[Array], nelem: int, expected: float
) -> None:
    """The finite element error, separate from the field's: the P1
    interpolant's divergence is O(h), not zero. Interpolation does not
    commute with divergence, and each of du_h/dx and dv_h/dy carries its
    own O(h) error.

    Setup: unit square, ``StructuredMesh2D(nelem, nelem,
    element_type="tri")``, P1 ``VectorLagrangeBasis``, n_psi = 6,
    lambda_ij = 1.5^2 / (i^2 + j^2)^2.5, psi_mean = sin(pi x) sin(pi y)
    + y, eta = standard normal with seed 0; L2 norm over the domain with
    skfem's quadrature.
    """
    from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.pde.galerkin.mesh import StructuredMesh2D

    bkd = numpy_bkd
    basis = VectorLagrangeBasis(
        StructuredMesh2D(
            nelem, nelem, [[0.0, 1.0], [0.0, 1.0]], bkd, element_type="tri"
        ),
        degree=1,
    )
    points = basis.dof_coordinates()[:, 0::2]
    field_map = _map(bkd, points, basis.component_layout())
    skfem_basis = basis.skfem_basis()
    field = skfem_basis.interpolate(bkd.to_numpy(field_map(_eta(bkd))))
    divergence = field.grad[0, 0] + field.grad[1, 1]
    l2 = math.sqrt(float(np.sum(divergence**2 * skfem_basis.dx)))
    bkd.assert_allclose(
        bkd.asarray([l2]), bkd.asarray([expected]), rtol=2e-4
    )
