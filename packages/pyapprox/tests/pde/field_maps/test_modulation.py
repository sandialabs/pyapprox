"""Tests for temporal modulations of a separable parameterized field.

The properties that matter are the ones a wrong implementation would
still look plausible without: that the profiles form a partition of
unity (so a control's total amplitude is preserved), that they are
nonnegative when they declare themselves so (an extraction rate that
goes negative is injection), and that the interpolation choice is
genuinely the user's rather than the time integrator's.
"""

import numpy as np
import pytest
from pyapprox.pde.field_maps.modulation import (
    ConstantModulation,
    NonNegativeModulationProtocol,
    PiecewiseConstantModulation,
    PiecewiseLinearModulation,
    TimeModulationProtocol,
)

_KNOTS = [0.0, 0.5, 1.0, 1.5, 2.0]


def _modulations(bkd):
    return (
        ConstantModulation(bkd, 3),
        PiecewiseConstantModulation(bkd, _KNOTS),
        PiecewiseLinearModulation(bkd, _KNOTS),
    )


class TestModulationProtocols:
    def test_all_satisfy_the_protocol(self, numpy_bkd) -> None:
        for modulation in _modulations(numpy_bkd):
            assert isinstance(modulation, TimeModulationProtocol)

    def test_all_declare_nonnegativity(self, numpy_bkd) -> None:
        """Extraction controls need the guarantee; all shipped bases
        provide it. A sign-changing basis (Fourier, a temporal KLE)
        would satisfy the base protocol but NOT this one."""
        for modulation in _modulations(numpy_bkd):
            assert isinstance(modulation, NonNegativeModulationProtocol)
            assert modulation.is_non_negative()

    def test_time_dependence_is_declared_not_inferred(
        self, numpy_bkd
    ) -> None:
        constant, piecewise, linear = _modulations(numpy_bkd)
        assert not constant.is_time_dependent()
        assert piecewise.is_time_dependent()
        assert linear.is_time_dependent()

    def test_values_shape_matches_nmodes(self, numpy_bkd) -> None:
        for modulation in _modulations(numpy_bkd):
            values = modulation.values(0.75)
            assert values.shape == (modulation.nmodes(),)


class TestPartitionOfUnity:
    """Both knot bases must sum to one, so the control's total
    amplitude is what the parameters say it is."""

    @pytest.mark.parametrize("time", [0.0, 0.25, 0.5, 1.1, 1.75, 2.0])
    def test_piecewise_constant_sums_to_one(
        self, numpy_bkd, time: float
    ) -> None:
        modulation = PiecewiseConstantModulation(numpy_bkd, _KNOTS)
        total = numpy_bkd.sum(modulation.values(time))
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(np.array([float(total)])),
            numpy_bkd.asarray(np.array([1.0])),
            rtol=1e-14,
        )

    @pytest.mark.parametrize("time", [0.0, 0.25, 0.5, 1.1, 1.75, 2.0])
    def test_piecewise_linear_sums_to_one(
        self, numpy_bkd, time: float
    ) -> None:
        modulation = PiecewiseLinearModulation(numpy_bkd, _KNOTS)
        total = numpy_bkd.sum(modulation.values(time))
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(np.array([float(total)])),
            numpy_bkd.asarray(np.array([1.0])),
            rtol=1e-14,
        )

    @pytest.mark.parametrize("time", [0.0, 0.3, 0.5, 1.4, 2.0])
    def test_values_are_nonnegative(self, numpy_bkd, time: float) -> None:
        for modulation in _modulations(numpy_bkd):
            values = numpy_bkd.to_numpy(modulation.values(time))
            assert values.min() >= 0.0


class TestPiecewiseLinearIsContinuous:
    def test_hat_peaks_at_its_own_knot(self, numpy_bkd) -> None:
        modulation = PiecewiseLinearModulation(numpy_bkd, _KNOTS)
        for index, knot in enumerate(_KNOTS):
            values = numpy_bkd.to_numpy(modulation.values(knot))
            assert values[index] == pytest.approx(1.0)
            assert values.sum() == pytest.approx(1.0)

    def test_midpoint_splits_evenly(self, numpy_bkd) -> None:
        """The defining property of linear interpolation, and what makes
        Crank-Nicolson and implicit midpoint agree instead of straddling
        a discontinuity."""
        modulation = PiecewiseLinearModulation(numpy_bkd, _KNOTS)
        values = numpy_bkd.to_numpy(modulation.values(0.25))
        assert values[0] == pytest.approx(0.5)
        assert values[1] == pytest.approx(0.5)

    def test_continuity_across_a_knot(self, numpy_bkd) -> None:
        modulation = PiecewiseLinearModulation(numpy_bkd, _KNOTS)
        eps = 1e-9
        below = numpy_bkd.to_numpy(modulation.values(1.0 - eps))
        above = numpy_bkd.to_numpy(modulation.values(1.0 + eps))
        assert np.abs(below - above).max() < 1e-6


class TestPiecewiseConstantConventions:
    def test_interior_knot_belongs_to_the_interval_ending_there(
        self, numpy_bkd
    ) -> None:
        """So a step's right endpoint lands in that step's own interval.
        With half-open intervals it would fall into the NEXT one, and a
        backward-Euler step whose knots align with the mesh would read
        the wrong window."""
        modulation = PiecewiseConstantModulation(numpy_bkd, _KNOTS)
        values = numpy_bkd.to_numpy(modulation.values(0.5))
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(0.0)

    def test_is_an_indicator(self, numpy_bkd) -> None:
        modulation = PiecewiseConstantModulation(numpy_bkd, _KNOTS)
        values = numpy_bkd.to_numpy(modulation.values(1.2))
        assert set(np.unique(values)).issubset({0.0, 1.0})
        assert values.sum() == pytest.approx(1.0)


class TestConstruction:
    def test_knots_must_increase(self, numpy_bkd) -> None:
        for cls in (PiecewiseConstantModulation, PiecewiseLinearModulation):
            with pytest.raises(ValueError, match="strictly increasing"):
                cls(numpy_bkd, [0.0, 1.0, 0.5])

    def test_at_least_two_knots(self, numpy_bkd) -> None:
        for cls in (PiecewiseConstantModulation, PiecewiseLinearModulation):
            with pytest.raises(ValueError, match="at least 2"):
                cls(numpy_bkd, [0.0])

    def test_nmodes_differs_between_the_two_bases(self, numpy_bkd) -> None:
        """Piecewise-constant parameters are per-INTERVAL; hat
        parameters are per-KNOT, i.e. the control's values there."""
        assert (
            PiecewiseConstantModulation(numpy_bkd, _KNOTS).nmodes()
            == len(_KNOTS) - 1
        )
        assert (
            PiecewiseLinearModulation(numpy_bkd, _KNOTS).nmodes()
            == len(_KNOTS)
        )

    def test_constant_modulation_rejects_empty(self, numpy_bkd) -> None:
        with pytest.raises(ValueError, match="positive"):
            ConstantModulation(numpy_bkd, 0)


class TestPicklable:
    def test_modulations_round_trip(self, numpy_bkd) -> None:
        """Parameterizations holding a modulation are shipped to workers
        during parallel sampling, so a closure-based basis would break
        there rather than here."""
        import pickle

        for modulation in _modulations(numpy_bkd):
            restored = pickle.loads(pickle.dumps(modulation))
            numpy_bkd.assert_allclose(
                restored.values(0.75), modulation.values(0.75), rtol=1e-14
            )


class TestLinearFieldMapMarker:
    """Only a linear map may carry a modulation.

    For a pointwise-nonlinear map the two do not commute --
    ``exp(sum_k p_k b_k(t) s_k)`` is not ``b(t) exp(sum_k p_k s_k)`` --
    so scaling jacobian columns would describe a field the forward
    solve never evaluates. The marker is what lets a consumer refuse
    the combination at construction instead of trusting the caller.
    """

    def test_linear_maps_declare_it(self, numpy_bkd) -> None:
        from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
        from pyapprox.pde.field_maps.mesh_kle_field_map import (
            MeshKLEFieldMap,
        )
        from pyapprox.pde.field_maps.protocol import LinearFieldMapProtocol
        from pyapprox.pde.field_maps.scalar import ScalarAmplitude

        maps = (
            BasisExpansion(
                numpy_bkd,
                0.0,
                [
                    numpy_bkd.asarray(np.ones(4)),
                    numpy_bkd.asarray(np.arange(4.0)),
                ],
            ),
            MeshKLEFieldMap(
                numpy_bkd,
                numpy_bkd.asarray(np.zeros(4)),
                numpy_bkd.asarray(np.ones((4, 2))),
            ),
            ScalarAmplitude(numpy_bkd, numpy_bkd.asarray(np.ones(4))),
        )
        for field_map in maps:
            assert isinstance(field_map, LinearFieldMapProtocol)
            assert field_map.is_linear()

    def test_exp_kle_does_not(self, numpy_bkd) -> None:
        from pyapprox.pde.field_maps.kle_factory import (
            create_lognormal_kle_field_map,
        )
        from pyapprox.pde.field_maps.protocol import LinearFieldMapProtocol

        lognormal = create_lognormal_kle_field_map(
            mesh_coords=numpy_bkd.asarray(np.linspace(0, 1, 4)[None, :]),
            mean_log_field=numpy_bkd.asarray(np.zeros(4)),
            bkd=numpy_bkd,
            correlation_length=0.5,
            num_kle_terms=2,
            sigma=0.5,
        )
        assert not isinstance(lognormal, LinearFieldMapProtocol)


class TestModulatedJacobianAgainstFiniteDifferences:
    """The per-column rule, checked against the field it must describe.

    This is the property every consumer depends on and the one a
    plausible-looking implementation gets wrong: ``b(t)`` scales each
    PARAMETER's column individually. A scalar or row-wise application
    produces a jacobian that is not the derivative of the separable
    field, by an O(1) margin rather than a small factor.

    Checked here, at the level of the two objects alone, because a
    downstream finite-difference check on the full PDE gradient can pass
    against a wrong rule if the forward field is wrong the same way.
    """

    def _setup(self, bkd, nmodes=3, npts=6):
        from pyapprox.pde.field_maps.basis_expansion import BasisExpansion

        rng = np.random.default_rng(0)
        modes = [bkd.asarray(rng.normal(size=npts)) for _ in range(nmodes)]
        field_map = BasisExpansion(bkd, 0.0, modes)
        modulation = PiecewiseLinearModulation(bkd, [0.0, 0.5, 1.0])
        return field_map, modulation, modes

    def _field(self, bkd, modes, modulation, params, time):
        """The separable field the forward solve is meant to evaluate."""
        profiles = bkd.to_numpy(modulation.values(time))
        columns = np.stack([bkd.to_numpy(mode) for mode in modes], axis=1)
        return columns @ (np.asarray(params) * profiles)

    @pytest.mark.parametrize("time", [0.0, 0.25, 0.5, 0.75])
    def test_per_column_scaling_matches_fd(
        self, numpy_bkd, time: float
    ) -> None:
        field_map, modulation, modes = self._setup(numpy_bkd)
        rng = np.random.default_rng(1)
        params = rng.normal(size=modulation.nmodes())
        step = 1e-7

        finite_difference = np.stack(
            [
                (
                    self._field(
                        numpy_bkd,
                        modes,
                        modulation,
                        params + step * np.eye(len(params))[k],
                        time,
                    )
                    - self._field(
                        numpy_bkd, modes, modulation, params, time
                    )
                )
                / step
                for k in range(len(params))
            ],
            axis=1,
        )
        analytic = numpy_bkd.to_numpy(
            field_map.jacobian(numpy_bkd.asarray(params))
        ) * numpy_bkd.to_numpy(modulation.values(time))[None, :]

        assert np.abs(analytic - finite_difference).max() < 1e-6

    def test_scalar_scaling_is_wrong_by_order_one(self, numpy_bkd) -> None:
        """The guard on the test above: it must be able to fail."""
        field_map, modulation, modes = self._setup(numpy_bkd)
        rng = np.random.default_rng(1)
        params = rng.normal(size=modulation.nmodes())
        time = 0.25
        step = 1e-7

        finite_difference = np.stack(
            [
                (
                    self._field(
                        numpy_bkd,
                        modes,
                        modulation,
                        params + step * np.eye(len(params))[k],
                        time,
                    )
                    - self._field(
                        numpy_bkd, modes, modulation, params, time
                    )
                )
                / step
                for k in range(len(params))
            ],
            axis=1,
        )
        scalar_scaled = numpy_bkd.to_numpy(
            field_map.jacobian(numpy_bkd.asarray(params))
        ) * float(numpy_bkd.to_numpy(modulation.values(time)).mean())

        assert np.abs(scalar_scaled - finite_difference).max() > 0.1
