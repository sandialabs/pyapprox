"""Tests for the assembly fast path on nodal coefficient fields.

``values_on_basis`` must agree with ``values`` to round-off. Assembled
operators, and every adjoint and HVP derived from them, are built from
these numbers, so a discrepancy would not surface as an error --- it
would surface as slightly wrong results everywhere.

The two routes are mathematically identical but arithmetically distinct:
``values`` builds a sparse interpolation operator and applies it, while
``values_on_basis`` sums basis functions element by element. They differ
only in summation order, so the tolerance below admits round-off (a few
parts in 1e12, growing with element degree) while still failing loudly
on the errors that matter --- a wrong element or a transposed component
axis shifts values by order one.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from pyapprox.pde.constitutive.coefficient_functions import (
    BasisEvaluableFieldProtocol,
    ConstantDiffusion,
    CoordinateDiffusion,
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldLinearReaction,
    NodalFieldVelocity,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh2D


def _mesh(bkd, nx=6):
    return StructuredMesh2D(
        nx=nx, ny=nx, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
    )


def _quadrature_coords(skfem_basis):
    """The coordinates assembly passes to a coefficient's ``values``."""
    return np.asarray(skfem_basis.global_coordinates())


class TestScalarFields:
    """Scalar nodal fields: diffusivity, forcing, linear reaction."""

    @pytest.mark.parametrize("degree", [1, 2])
    @pytest.mark.parametrize(
        "field_cls,evaluate_name",
        [
            (NodalFieldDiffusion, "values"),
            (NodalFieldForcing, "__call__"),
            (NodalFieldLinearReaction, "values"),
        ],
    )
    def test_agrees_with_values(
        self, numpy_bkd, degree, field_cls, evaluate_name
    ) -> None:
        """The fast path reproduces the general path to round-off."""
        bkd = numpy_bkd
        basis = LagrangeBasis(_mesh(bkd), degree=degree)

        rng = np.random.default_rng(0)
        dofs = rng.normal(size=basis.ndofs())
        field = field_cls(basis, dofs)

        skfem_basis = basis.skfem_basis()
        coords = _quadrature_coords(skfem_basis)

        general = np.asarray(getattr(field, evaluate_name)(coords))
        fast = np.asarray(field.values_on_basis(skfem_basis))

        assert fast.shape == general.shape
        bkd.assert_allclose(
            bkd.asarray(fast), bkd.asarray(general), rtol=1e-11, atol=1e-13
        )

    def test_tracks_set_dofs(self, numpy_bkd) -> None:
        """The fast path sees DOF updates, not a stale cache."""
        bkd = numpy_bkd
        basis = LagrangeBasis(_mesh(bkd), degree=1)
        field = NodalFieldDiffusion(basis, np.ones(basis.ndofs()))
        skfem_basis = basis.skfem_basis()

        before = np.asarray(field.values_on_basis(skfem_basis))
        field.set_dofs(np.full(basis.ndofs(), 3.0))
        after = np.asarray(field.values_on_basis(skfem_basis))

        bkd.assert_allclose(
            bkd.asarray(after), bkd.asarray(3.0 * before),
            rtol=1e-11, atol=1e-13,
        )

    def test_rejects_foreign_basis(self, numpy_bkd) -> None:
        """A basis the DOFs do not belong to raises rather than
        silently returning values at the wrong locations."""
        bkd = numpy_bkd
        field = NodalFieldDiffusion(LagrangeBasis(_mesh(bkd, nx=6), degree=1))
        other = LagrangeBasis(_mesh(bkd, nx=8), degree=1)

        with pytest.raises(ValueError, match="values_on_basis requires"):
            field.values_on_basis(other.skfem_basis())


class TestVectorField:
    """The vector case, where the component axis could be transposed."""

    @pytest.mark.parametrize("degree", [1, 2])
    def test_agrees_with_values(self, numpy_bkd, degree) -> None:
        bkd = numpy_bkd
        basis = VectorLagrangeBasis(_mesh(bkd), degree=degree)

        rng = np.random.default_rng(0)
        field = NodalFieldVelocity(basis, rng.normal(size=basis.ndofs()))

        skfem_basis = basis.skfem_basis()
        coords = _quadrature_coords(skfem_basis)

        general = np.asarray(field.values(coords))
        fast = np.asarray(field.values_on_basis(skfem_basis))

        # (ncomponents, nelems, nquad) — a transposed component axis
        # would still broadcast in assembly, so check the shape too.
        assert fast.shape == general.shape
        assert fast.shape[0] == coords.shape[0]
        bkd.assert_allclose(
            bkd.asarray(fast), bkd.asarray(general), rtol=1e-11, atol=1e-13
        )

    def test_components_are_not_swapped(self, numpy_bkd) -> None:
        """A constant field per component pins the component order."""
        bkd = numpy_bkd
        basis = VectorLagrangeBasis(_mesh(bkd), degree=1)

        # Interleaved DOFs: [ux_0, uy_0, ux_1, uy_1, ...]
        dofs = np.empty(basis.ndofs())
        dofs[0::2] = 2.0
        dofs[1::2] = -5.0
        field = NodalFieldVelocity(basis, dofs)

        fast = np.asarray(field.values_on_basis(basis.skfem_basis()))
        bkd.assert_allclose(
            bkd.asarray([fast[0].min(), fast[0].max()]),
            bkd.asarray([2.0, 2.0]), rtol=1e-13, atol=1e-14,
        )
        bkd.assert_allclose(
            bkd.asarray([fast[1].min(), fast[1].max()]),
            bkd.asarray([-5.0, -5.0]), rtol=1e-13, atol=1e-14,
        )


class TestProtocol:
    """Capability is declared, so consumers can branch on it."""

    def test_nodal_fields_satisfy_protocol(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = LagrangeBasis(_mesh(bkd), degree=1)
        vector_basis = VectorLagrangeBasis(_mesh(bkd), degree=1)

        for field in (
            NodalFieldDiffusion(basis),
            NodalFieldForcing(basis, np.ones(basis.ndofs())),
            NodalFieldLinearReaction(basis, np.ones(basis.ndofs())),
            NodalFieldVelocity(
                vector_basis, np.ones(vector_basis.ndofs())
            ),
        ):
            assert isinstance(field, BasisEvaluableFieldProtocol)

    def test_non_nodal_coefficients_do_not(self) -> None:
        """Constants and coordinate callables have no basis to evaluate
        on, so consumers must keep the general path for them."""
        assert not isinstance(
            ConstantDiffusion(2.0), BasisEvaluableFieldProtocol
        )
        assert not isinstance(
            CoordinateDiffusion(lambda x: np.ones(x.shape[-1])),
            BasisEvaluableFieldProtocol,
        )
