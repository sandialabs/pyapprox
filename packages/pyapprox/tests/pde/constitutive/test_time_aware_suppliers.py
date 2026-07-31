"""Tests for how coefficient suppliers declare their time-awareness.

Time is threaded as an argument and whether a supplier consults it is
DECLARED, never inferred. These tests pin the three things that make
that safe: the declaration is honoured, an undeclared-but-ambiguous
supplier is refused rather than guessed at, and a wrong guess can no
longer hide --- a forcing that raises must surface its own error rather
than be silently re-invoked without the time.
"""

import pickle

import numpy as np
import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.pde.constitutive.coefficient_functions import (
    CoordinateDiffusion,
    CoordinateVelocity,
    NodalFieldForcing,
    TimeAwareCallableProtocol,
    TimeDependent,
    TimeIndependent,
    as_time_aware,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)


def _basis(bkd, nx=6):
    mesh = StructuredMesh2D(
        nx=nx, ny=nx, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
    )
    return LagrangeBasis(mesh, degree=1)


def _physics(bkd, **kwargs):
    return AdvectionDiffusionReaction(
        basis=_basis(bkd), diffusivity=1.0, bkd=bkd, **kwargs
    )


class TestNormalization:
    """``as_time_aware`` maps every valid supplier to one call form."""

    def test_bare_callable_is_time_independent(self, numpy_bkd) -> None:
        """``f(coords)`` needs no wrapper: it is the documented form."""
        bkd = numpy_bkd

        def forcing(coords):
            return np.ones(coords.shape[-1])

        supplier = as_time_aware(forcing)
        assert not supplier.is_time_dependent()
        bkd.assert_allclose(
            bkd.asarray(supplier(np.zeros((2, 3)), 7.0)),
            bkd.asarray(np.ones(3)),
            rtol=1e-14,
        )

    @pytest.mark.parametrize("wrapper", [TimeIndependent, TimeDependent])
    def test_declared_suppliers_pass_through(self, wrapper) -> None:
        """Normalization is idempotent: a declared supplier is
        returned unchanged, so wrapping twice cannot double-wrap."""
        supplier = wrapper(lambda *args: np.ones(1))
        assert as_time_aware(supplier) is supplier
        assert as_time_aware(as_time_aware(supplier)) is supplier

    def test_declaration_is_honoured(self, numpy_bkd) -> None:
        """A declared time-dependent supplier receives the time."""
        bkd = numpy_bkd

        def forcing(coords, time):
            return (1.0 + time) * np.ones(coords.shape[-1])

        supplier = as_time_aware(TimeDependent(forcing))
        assert supplier.is_time_dependent()
        bkd.assert_allclose(
            bkd.asarray(supplier(np.zeros((2, 3)), 3.0)),
            bkd.asarray(4.0 * np.ones(3)),
            rtol=1e-14,
        )

    def test_wrappers_are_picklable(self) -> None:
        """Wrapping must not make a picklable supplier unpicklable:
        parallel workers receive physics by pickling."""
        for supplier in (TimeIndependent(np.negative),
                         TimeDependent(np.add)):
            restored = pickle.loads(pickle.dumps(supplier))
            assert (
                restored.is_time_dependent()
                == supplier.is_time_dependent()
            )

    def test_nodal_field_forcing_declares_and_accepts_time(
        self, numpy_bkd
    ) -> None:
        """A callable field object satisfies the supplier contract it
        declares --- it must accept the time it says it ignores."""
        bkd = numpy_bkd
        basis = _basis(bkd)
        field = NodalFieldForcing(basis, np.ones(basis.ndofs()))

        assert isinstance(field, TimeAwareCallableProtocol)
        assert not field.is_time_dependent()
        assert as_time_aware(field) is field

        coords = bkd.to_numpy(basis.dof_coordinates())
        bkd.assert_allclose(
            bkd.asarray(field(coords, 5.0)),
            bkd.asarray(field(coords)),
            rtol=1e-14,
        )


class TestAmbiguityIsRefused:
    """An undeclared supplier that could take a time is rejected.

    This is the rule that removes the whole class of bug: with no third
    category there is nothing left to guess about.
    """

    def test_undeclared_two_arity_raises(self) -> None:
        def forcing(coords, time):
            return np.ones(coords.shape[-1])

        with pytest.raises(TypeError, match="whether it depends on time"):
            as_time_aware(forcing)

    def test_undeclared_defaulted_time_raises(self) -> None:
        """The dangerous shape: called with one argument it SUCCEEDS
        and silently evaluates at t=0, so nothing would surface."""

        def forcing(coords, time=0.0):
            return (1.0 + time) * np.ones(coords.shape[-1])

        with pytest.raises(TypeError, match="TimeDependent"):
            as_time_aware(forcing)

    def test_var_positional_raises(self) -> None:
        with pytest.raises(TypeError, match="TimeIndependent"):
            as_time_aware(lambda *args: np.ones(1))

    def test_message_names_both_wrappers(self) -> None:
        """The error must say how to fix it, not merely refuse."""
        with pytest.raises(TypeError) as excinfo:
            as_time_aware(lambda coords, time=0.0: coords)
        message = str(excinfo.value)
        assert "TimeDependent" in message
        assert "TimeIndependent" in message

    def test_declaring_resolves_the_refusal(self, numpy_bkd) -> None:
        """The documented fix works for both readings."""
        bkd = numpy_bkd

        def forcing(coords, time=0.0):
            return (1.0 + time) * np.ones(coords.shape[-1])

        dependent = as_time_aware(TimeDependent(forcing))
        independent = as_time_aware(TimeIndependent(forcing))
        assert dependent.is_time_dependent()
        assert not independent.is_time_dependent()
        # The declaration decides whether the time reaches the supplier.
        bkd.assert_allclose(
            bkd.asarray(dependent(np.zeros((2, 2)), 1.0)),
            bkd.asarray(2.0 * np.ones(2)),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            bkd.asarray(independent(np.zeros((2, 2)), 1.0)),
            bkd.asarray(np.ones(2)),
            rtol=1e-14,
        )


class TestErrorsAreNotMasked:
    """The regression this work exists to prevent."""

    def test_typeerror_inside_forcing_propagates(self, numpy_bkd) -> None:
        """A TypeError raised INSIDE a forcing must surface as itself.

        The old arity probe caught it, mistook it for a wrong-arity
        call, and re-invoked without the time --- turning a bug in user
        code into either a confusing second error or a silently wrong
        answer. This test fails on the pre-fix code.
        """
        bkd = numpy_bkd
        sentinel = "deliberate failure inside the forcing"

        def broken_forcing(coords):
            raise TypeError(sentinel)

        physics = _physics(bkd, forcing=broken_forcing)
        with pytest.raises(TypeError, match=sentinel):
            physics.spatial_residual(bkd.zeros((physics.nstates(),)), 0.0)


class TestTimeDependentCoefficients:
    """Coefficients may now depend on time; the assembly must see it."""

    def test_coordinate_diffusivity_tracks_time(self, numpy_bkd) -> None:
        """Swept over several times rather than compared at two: two
        points can agree by coincidence, a whole range cannot."""
        bkd = numpy_bkd

        def kappa(coords, time):
            return (1.0 + time) * np.ones(coords.shape[-1])

        diffusion = CoordinateDiffusion(TimeDependent(kappa))
        assert diffusion.is_time_dependent()

        coords = np.zeros((2, 4))
        for time in (0.0, 0.5, 1.0, 2.5, 7.0):
            bkd.assert_allclose(
                bkd.asarray(diffusion.values(coords, time)),
                bkd.asarray((1.0 + time) * np.ones(4)),
                rtol=1e-14,
            )

    def test_coordinate_velocity_tracks_time(self, numpy_bkd) -> None:
        """The transient-flow case: a velocity from an upstream solve
        must be evaluated at the assembly's time, not frozen."""
        bkd = numpy_bkd

        def velocity(coords, time):
            return (1.0 + time) * np.ones(coords.shape)

        field = CoordinateVelocity(TimeDependent(velocity))
        assert field.is_time_dependent()

        coords = np.zeros((2, 4))
        for time in (0.0, 0.5, 1.0, 2.5, 7.0):
            bkd.assert_allclose(
                bkd.asarray(field.values(coords, time)),
                bkd.asarray((1.0 + time) * np.ones((2, 4))),
                rtol=1e-14,
            )

    def test_steady_coefficients_declare_independence(
        self, numpy_bkd
    ) -> None:
        """A bare callable coefficient stays time-independent, so the
        existing single-key cache path is unaffected."""
        assert not CoordinateDiffusion(
            lambda coords: np.ones(coords.shape[-1])
        ).is_time_dependent()
        assert not CoordinateVelocity(
            lambda coords: np.ones(coords.shape)
        ).is_time_dependent()


class TestLoadIsNotFrozenInTime:
    """Assembled loads must follow a time-dependent forcing."""

    def test_declared_forcing_reassembles_per_time(
        self, numpy_bkd
    ) -> None:
        bkd = numpy_bkd

        def forcing(coords, time):
            return (1.0 + time) * np.ones(coords.shape[-1])

        physics = _physics(bkd, forcing=TimeDependent(forcing))
        zeros = bkd.zeros((physics.nstates(),))
        load_t0 = physics.spatial_residual(zeros, 0.0)
        load_t1 = physics.spatial_residual(zeros, 1.0)

        bkd.assert_allclose(load_t1, 2.0 * load_t0, rtol=1e-12)

    def test_steady_forcing_is_stable_across_times(
        self, numpy_bkd
    ) -> None:
        """The complement: a steady forcing must NOT vary with time,
        which is what makes caching it legitimate."""
        bkd = numpy_bkd
        physics = _physics(
            bkd, forcing=lambda coords: np.ones(coords.shape[-1])
        )
        zeros = bkd.zeros((physics.nstates(),))
        bkd.assert_allclose(
            physics.spatial_residual(zeros, 3.0),
            physics.spatial_residual(zeros, 0.0),
            rtol=1e-14,
        )
