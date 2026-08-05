"""Tests for the forcing contribution to the ADR load vector.

Two properties, both of which fail silently rather than loudly when
broken, so each is pinned by a test that asserts on assembled numbers
rather than on cache internals:

(i) The assembled load is cached across time steps ONLY for a forcing
    that declares its values do not depend on time. The cache key is the
    coefficient version and carries no time component, so caching a
    time-varying forcing would freeze it at its first evaluation.

(ii) The basis fast path (``values_on_basis``) and the coordinate path
     produce the same load. The fast path exists because evaluating a
     nodal field by coordinate repeats the element search the basis has
     already done.

The invariant guards at the end are the ones that matter for future
work: a time-varying nodal forcing must NOT become cacheable, and must
NOT reach ``_FieldOnBasisEvaluator``, which discards the time it is
handed.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from pyapprox.pde.constitutive.coefficient_functions import (
    BasisEvaluableFieldProtocol,
    NodalFieldForcing,
    TimeDependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.physics.advection_diffusion import (
    _coefficient_evaluator,
    _FieldOnBasisEvaluator,
    _ForcingKernel,
)
from skfem import LinearForm, asm


def _ramp(coords, time):
    """A forcing that genuinely varies in time."""
    return np.full(coords.shape[1], 1.0 + 10.0 * time)


class _TimeVaryingNodalForcing(NodalFieldForcing):
    """A nodal forcing whose values depend on time.

    Subclasses ``NodalFieldForcing`` ON PURPOSE. That is the shape a
    future time-dependent forcing is most likely to arrive in, and it is
    the shape that reaches the version-keyed load cache — whose key has
    no time component. A test double that merely duck-types the
    interface would take the assemble-fresh branch in every version of
    the code and so could never catch the freezing.

    ``version()`` is left inherited and constant, again deliberately:
    encoding the time into it would paper over the defect rather than
    expose it.
    """

    def is_time_dependent(self) -> bool:
        return True

    def __call__(self, coords, time: float = 0.0):
        return np.full(coords.shape[1], 1.0 + 10.0 * time)

    def values_on_basis(self, skfem_basis, time: float = 0.0):
        """Time-varying on the fast path too.

        Overriding only ``__call__`` would leave this returning the
        frozen inherited DOFs, so a fast path that ignored time would
        still look correct here — the test would be unable to fail.
        """
        return np.full(
            skfem_basis.global_coordinates().shape[1:], 1.0 + 10.0 * time
        )


def _make_basis(numpy_bkd):
    """The basis alone: forcings need it before the physics exists."""
    mesh = StructuredMesh2D(
        nx=4, ny=4, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=numpy_bkd
    )
    return LagrangeBasis(mesh, degree=1)


def _make_physics(numpy_bkd, basis, forcing):
    return AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=1.0,
        bkd=numpy_bkd,
        forcing=forcing,
        boundary_conditions=[
            DirichletBC(basis, name, 0.0, numpy_bkd)
            for name in ("left", "right", "bottom", "top")
        ],
    )


class TestForcingLoadTimeDependence:
    def test_declared_time_dependent_nodal_forcing_is_not_frozen(
        self, numpy_bkd
    ) -> None:
        """A nodal forcing declaring time dependence must not be cached.

        The regression: the load cache keys on version() alone, so a
        forcing that truthfully reports is_time_dependent() would have
        the first step's load reused for every later time.
        """
        basis = _make_basis(numpy_bkd)
        forcing = _TimeVaryingNodalForcing(
            basis, dofs=np.ones(basis.ndofs())
        )
        physics = _make_physics(numpy_bkd, basis, forcing)

        loads = [physics._assemble_forcing_load(t) for t in (0.0, 0.5, 1.0)]
        totals = numpy_bkd.asarray(
            np.array([float(np.asarray(load).sum()) for load in loads])
        )
        # f = 1 + 10t on the unit square: (w, f) sums to the area times
        # the value, so the totals are the values themselves.
        numpy_bkd.assert_allclose(
            totals, numpy_bkd.asarray(np.array([1.0, 6.0, 11.0])), rtol=1e-12
        )

    def test_callable_time_dependent_forcing_tracks_time(
        self, numpy_bkd
    ) -> None:
        """The declared-callable path was already correct; keep it so."""
        basis = _make_basis(numpy_bkd)
        physics = _make_physics(numpy_bkd, basis, TimeDependent(_ramp))

        loads = [physics._assemble_forcing_load(t) for t in (0.0, 0.5, 1.0)]
        totals = numpy_bkd.asarray(
            np.array([float(np.asarray(load).sum()) for load in loads])
        )
        numpy_bkd.assert_allclose(
            totals, numpy_bkd.asarray(np.array([1.0, 6.0, 11.0])), rtol=1e-12
        )

    def test_time_independent_nodal_forcing_is_cached(
        self, numpy_bkd
    ) -> None:
        """The fast path for the common case must survive the fix."""
        basis = _make_basis(numpy_bkd)
        forcing = NodalFieldForcing(basis, dofs=np.ones(basis.ndofs()))
        physics = _make_physics(numpy_bkd, basis, forcing)

        first = physics._assemble_forcing_load(0.0)
        again = physics._assemble_forcing_load(7.5)
        # Same object: the load was reused rather than reassembled, and
        # the time it was asked for made no difference.
        assert again is first

    def test_set_dofs_invalidates_the_cache(self, numpy_bkd) -> None:
        basis = _make_basis(numpy_bkd)
        forcing = NodalFieldForcing(basis, dofs=np.ones(basis.ndofs()))
        physics = _make_physics(numpy_bkd, basis, forcing)

        before = np.asarray(physics._assemble_forcing_load(0.0)).sum()
        forcing.set_dofs(3.0 * np.ones(basis.ndofs()))
        after = np.asarray(physics._assemble_forcing_load(0.0)).sum()

        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(np.array([after])),
            numpy_bkd.asarray(np.array([3.0 * before])),
            rtol=1e-12,
        )


class TestForcingAssemblyFastPath:
    def test_basis_and_coordinate_paths_agree(self, numpy_bkd) -> None:
        """The fast path must be a pure optimization.

        Both routes integrate the same nodal interpolant; the basis one
        skips locating the element holding each quadrature point.
        """
        basis = _make_basis(numpy_bkd)
        forcing = NodalFieldForcing(
            basis, dofs=np.linspace(0.5, 2.0, basis.ndofs())
        )
        skfem_basis = basis.skfem_basis()

        by_coordinate = np.asarray(
            asm(LinearForm(_ForcingKernel(forcing, 0.0)), skfem_basis)
        )
        by_basis = np.asarray(
            asm(
                LinearForm(
                    _ForcingKernel(
                        _FieldOnBasisEvaluator(forcing, skfem_basis), 0.0
                    )
                ),
                skfem_basis,
            )
        )
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(by_basis),
            numpy_bkd.asarray(by_coordinate),
            atol=1e-15,
            rtol=0.0,
        )


class TestForcingCapabilityInvariants:
    """Guards on the two assumptions the forcing path relies on.

    Both are about a field that does not exist yet. They are written
    now because the failure they describe is silent: a time-varying
    nodal forcing that acquires either capability would be evaluated at
    the wrong time with nothing raising.
    """

    def test_time_varying_forcing_must_not_be_load_cacheable(
        self, numpy_bkd
    ) -> None:
        """Declaring time dependence must defeat the version cache.

        If a future field becomes cacheable while varying in time, the
        version-keyed cache — which has no time component — will serve
        the first step's load forever.
        """
        basis = _make_basis(numpy_bkd)
        physics = _make_physics(
            numpy_bkd,
            basis,
            _TimeVaryingNodalForcing(basis, dofs=np.ones(basis.ndofs())),
        )
        assert not physics._forcing_load_cacheable

    def test_time_varying_forcing_keeps_the_basis_fast_path(
        self, numpy_bkd
    ) -> None:
        """A time-varying field is still eligible for the fast path.

        Excluding it would be safe but expensive on exactly the path
        that pays: a fixed field assembles once, a time-varying one
        every step.
        """
        basis = _make_basis(numpy_bkd)
        forcing = _TimeVaryingNodalForcing(
            basis, dofs=np.ones(basis.ndofs())
        )
        assert isinstance(forcing, BasisEvaluableFieldProtocol)

        evaluator = _coefficient_evaluator(
            forcing, basis.skfem_basis(), forcing
        )
        assert isinstance(evaluator, _FieldOnBasisEvaluator)

    def test_the_fast_path_tracks_time(self, numpy_bkd) -> None:
        """The property the fast path must not quietly lose.

        Comparing the two paths at a SINGLE time cannot catch a fast
        path that ignores time: both would agree there and disagree
        everywhere else. The cache would also invalidate correctly each
        step, so the frozen field would look right.
        """
        basis = _make_basis(numpy_bkd)
        forcing = _TimeVaryingNodalForcing(
            basis, dofs=np.ones(basis.ndofs())
        )
        evaluator = _coefficient_evaluator(
            forcing, basis.skfem_basis(), forcing
        )
        coords = numpy_bkd.to_numpy(basis.dof_coordinates())

        early = np.asarray(evaluator(coords, 0.0))
        late = np.asarray(evaluator(coords, 1.0))
        assert np.abs(late - early).max() > 1e-8

    def test_fast_path_defers_time_dependence_to_the_field(
        self, numpy_bkd
    ) -> None:
        """So wrapping a field does not change how caches key on it."""
        basis = _make_basis(numpy_bkd)
        varying = _TimeVaryingNodalForcing(
            basis, dofs=np.ones(basis.ndofs())
        )
        fixed = NodalFieldForcing(basis, dofs=np.ones(basis.ndofs()))
        skfem_basis = basis.skfem_basis()

        assert _FieldOnBasisEvaluator(
            varying, skfem_basis
        ).is_time_dependent()
        assert not _FieldOnBasisEvaluator(
            fixed, skfem_basis
        ).is_time_dependent()

    def test_fixed_nodal_forcing_keeps_both_capabilities(
        self, numpy_bkd
    ) -> None:
        """The converse, so the guards above cannot pass vacuously."""
        basis = _make_basis(numpy_bkd)
        forcing = NodalFieldForcing(basis, dofs=np.ones(basis.ndofs()))
        physics = _make_physics(numpy_bkd, basis, forcing)

        assert isinstance(forcing, BasisEvaluableFieldProtocol)
        assert physics._forcing_load_cacheable
