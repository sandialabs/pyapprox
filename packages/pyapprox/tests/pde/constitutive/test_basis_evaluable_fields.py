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


class TestTimeVaryingFieldsMustVaryOnBothPaths:
    """A field declaring time dependence must honour it on BOTH routes.

    The trap this closes: override ``values``/``__call__`` to consult
    time but leave ``values_on_basis`` returning fixed DOFs, and the
    coordinate path varies while the fast path does not. Assembly
    prefers the fast path, so the solve silently uses frozen values --
    and because the cache still invalidates each step (the field
    declares itself time-dependent), the result looks right.

    Agreement at a SINGLE time cannot catch it: the two paths coincide
    wherever the frozen values happen to be correct, which includes
    whatever time the field was built at.
    """

    @staticmethod
    def _check_paths_agree_over_time(bkd, field, basis, times):
        skfem_basis = basis.skfem_basis()
        coords = _quadrature_coords(skfem_basis)
        for time in times:
            general = np.asarray(field.values(coords, time))
            fast = np.asarray(field.values_on_basis(skfem_basis, time))
            assert fast.shape == general.shape
            bkd.assert_allclose(
                bkd.asarray(fast),
                bkd.asarray(general),
                rtol=1e-11,
                atol=1e-13,
            )

    def test_a_correct_time_varying_field_passes(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis = LagrangeBasis(_mesh(bkd), degree=1)

        class _Consistent(NodalFieldDiffusion):
            """Both routes scale the DOFs by the same factor."""

            def is_time_dependent(self) -> bool:
                return True

            def _scaled(self, time):
                return self.dofs() * (1.0 + 10.0 * time)

            def values(self, coords, time=0.0):
                coords_np = np.asarray(coords)
                flat = coords_np.reshape(coords_np.shape[0], -1)
                values = np.asarray(
                    self._basis.evaluate(self._scaled(time), flat)
                )
                return values.reshape(coords_np.shape[1:])

            def values_on_basis(self, skfem_basis, time=0.0):
                return np.asarray(
                    skfem_basis.interpolate(self._scaled(time))
                )

        field = _Consistent(basis, np.linspace(0.5, 2.0, basis.ndofs()))
        self._check_paths_agree_over_time(
            bkd, field, basis, (0.0, 0.3, 1.0)
        )

    def test_a_field_varying_on_one_path_only_is_caught(
        self, numpy_bkd
    ) -> None:
        """The guard on the test above: it must be able to fail.

        This double is exactly the mistake -- ``values`` consults time,
        ``values_on_basis`` is inherited and does not.
        """
        bkd = numpy_bkd
        basis = LagrangeBasis(_mesh(bkd), degree=1)

        class _InconsistentOnFastPath(NodalFieldDiffusion):
            def is_time_dependent(self) -> bool:
                return True

            def values(self, coords, time=0.0):
                coords_np = np.asarray(coords)
                flat = coords_np.reshape(coords_np.shape[0], -1)
                values = np.asarray(
                    self._basis.evaluate(
                        self.dofs() * (1.0 + 10.0 * time), flat
                    )
                )
                return values.reshape(coords_np.shape[1:])

        field = _InconsistentOnFastPath(
            basis, np.linspace(0.5, 2.0, basis.ndofs())
        )
        # Agrees at t = 0, where the frozen DOFs are still correct.
        self._check_paths_agree_over_time(bkd, field, basis, (0.0,))
        # Diverges as soon as time moves.
        with pytest.raises(AssertionError):
            self._check_paths_agree_over_time(bkd, field, basis, (0.5,))

    def test_shipped_fields_declare_no_time_dependence(
        self, numpy_bkd
    ) -> None:
        """Why the shipped fields need no such check: their DOFs are
        fixed, so both routes ignore the time by construction."""
        bkd = numpy_bkd
        basis = LagrangeBasis(_mesh(bkd), degree=1)
        dofs = np.ones(basis.ndofs())
        for field_cls in (
            NodalFieldDiffusion,
            NodalFieldForcing,
            NodalFieldLinearReaction,
        ):
            assert not field_cls(basis, dofs).is_time_dependent()


class TestTimeModulatedNodalFieldLinearReaction:
    """A separable coefficient r(x,t) = sum_k c_k b_k(t) s_k(x).

    The field holds NUMBERS -- modes, a modulation, coefficients -- and
    answers what its values are at a point and time. It knows nothing
    about field maps, parameters, or derivatives: computing those here
    would put the parameterization's job in the constitutive layer.
    """

    @staticmethod
    def _field(numpy_bkd, coefficients=None):
        from pyapprox.pde.constitutive.coefficient_functions import (
            TimeModulatedNodalFieldLinearReaction,
        )
        from pyapprox.pde.field_maps.modulation import (
            PiecewiseLinearModulation,
        )
        from pyapprox.pde.galerkin.basis import LagrangeBasis

        basis = LagrangeBasis(_mesh(numpy_bkd), degree=1)
        coords = numpy_bkd.to_numpy(basis.dof_coordinates())
        modes = np.stack(
            [
                np.exp(-20.0 * (coords[0] - centre) ** 2)
                for centre in (0.25, 0.5, 0.75)
            ],
            axis=1,
        )
        modulation = PiecewiseLinearModulation(
            numpy_bkd, [0.0, 0.5, 1.0]
        )
        if coefficients is None:
            coefficients = np.array([2.0, 1.0, 3.0])
        return (
            TimeModulatedNodalFieldLinearReaction(
                basis, modes, modulation, coefficients
            ),
            basis,
            modes,
        )

    def test_declares_time_dependence(self, numpy_bkd) -> None:
        field, _, _ = self._field(numpy_bkd)
        assert field.is_time_dependent()

    def test_dofs_realize_the_separable_sum(self, numpy_bkd) -> None:
        """At a knot the hat basis is a unit vector, so the realized
        DOFs are exactly that mode scaled by its coefficient --- an
        analytic check on the whole construction."""
        field, _, modes = self._field(numpy_bkd)
        realized = np.asarray(field.dofs_at(0.0))
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(realized),
            numpy_bkd.asarray(2.0 * modes[:, 0]),
            rtol=1e-13,
        )

    def test_dofs_track_time(self, numpy_bkd) -> None:
        field, _, _ = self._field(numpy_bkd)
        early = np.asarray(field.dofs_at(0.0))
        late = np.asarray(field.dofs_at(1.0))
        assert np.abs(late - early).max() > 1e-8

    def test_both_evaluation_paths_agree_over_time(self, numpy_bkd) -> None:
        """The step-3 invariant: a field varying in time must vary on
        the coordinate path AND the basis fast path, or assembly reads
        frozen values while the cache looks correct."""
        field, basis, _ = self._field(numpy_bkd)
        skfem_basis = basis.skfem_basis()
        coords = _quadrature_coords(skfem_basis)
        for time in (0.0, 0.25, 0.75, 1.0):
            general = np.asarray(field.values(coords, time))
            fast = np.asarray(field.values_on_basis(skfem_basis, time))
            assert fast.shape == general.shape
            numpy_bkd.assert_allclose(
                numpy_bkd.asarray(fast),
                numpy_bkd.asarray(general),
                rtol=1e-11,
                atol=1e-13,
            )

    def test_zero_coefficients_give_a_zero_field(self, numpy_bkd) -> None:
        field, _, _ = self._field(numpy_bkd, np.zeros(3))
        for time in (0.0, 0.5, 1.0):
            assert np.abs(np.asarray(field.dofs_at(time))).max() == 0.0

    def test_version_bumps_on_coefficient_update(self, numpy_bkd) -> None:
        """So assembled operators keyed on it invalidate."""
        field, _, _ = self._field(numpy_bkd)
        before = field.version()
        field.set_coefficients(np.array([1.0, 1.0, 1.0]))
        assert field.version() > before

    def test_rejects_a_mode_count_mismatch(self, numpy_bkd) -> None:
        from pyapprox.pde.constitutive.coefficient_functions import (
            TimeModulatedNodalFieldLinearReaction,
        )
        from pyapprox.pde.field_maps.modulation import ConstantModulation
        from pyapprox.pde.galerkin.basis import LagrangeBasis

        basis = LagrangeBasis(_mesh(numpy_bkd), degree=1)
        modes = np.ones((basis.ndofs(), 2))
        with pytest.raises(ValueError, match="each mode"):
            TimeModulatedNodalFieldLinearReaction(
                basis, modes, ConstantModulation(numpy_bkd, 3)
            )

    def test_rejects_wrong_mode_row_count(self, numpy_bkd) -> None:
        from pyapprox.pde.constitutive.coefficient_functions import (
            TimeModulatedNodalFieldLinearReaction,
        )
        from pyapprox.pde.field_maps.modulation import ConstantModulation
        from pyapprox.pde.galerkin.basis import LagrangeBasis

        basis = LagrangeBasis(_mesh(numpy_bkd), degree=1)
        with pytest.raises(ValueError, match="spatial_modes must have"):
            TimeModulatedNodalFieldLinearReaction(
                basis,
                np.ones((basis.ndofs() - 1, 2)),
                ConstantModulation(numpy_bkd, 2),
            )
