"""Tests for zone weights (pluggable spatial QoI weighting)."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.zone_weights import (
    ElementAlignedRectangleZone,
    SmoothDiscZone,
    ZoneWeightProtocol,
)
from pyapprox.pde.zoo.obstructed_flow import build_obstructed_mesh
from pyapprox.util.backends.numpy import NumpyBkd

# The rectangle directly above the uppermost obstruction block: all
# edges lie on the coarse tensor grid (x in {4/7, 5/7}, y in {3/4, 1}),
# which nested uniform refinement preserves.
_ZONE_XLIM = (4.0 / 7.0, 5.0 / 7.0)
_ZONE_YLIM = (0.75, 1.0)
_ZONE_AREA = (5.0 / 7.0 - 4.0 / 7.0) * 0.25


def _basis(bkd, nrefine):
    return LagrangeBasis(build_obstructed_mesh(bkd, nrefine), degree=1)


class TestElementAlignedRectangleZone:
    def test_protocol_conformance(self) -> None:
        zone = ElementAlignedRectangleZone(_ZONE_XLIM, _ZONE_YLIM)
        assert isinstance(zone, ZoneWeightProtocol)

    def test_extent_validation(self) -> None:
        with pytest.raises(ValueError, match="positive extent"):
            ElementAlignedRectangleZone((0.5, 0.4), (0.0, 1.0))

    @pytest.mark.parametrize("nrefine", [0, 1, 2])
    def test_exact_assembly_integrates_area(
        self, numpy_bkd: NumpyBkd, nrefine: int
    ) -> None:
        """1^T W 1 = int_zone 1 dx = the rectangle area, EXACTLY, at
        every nested-refinement level — the property cut elements
        destroy."""
        bkd = numpy_bkd
        zone = ElementAlignedRectangleZone(_ZONE_XLIM, _ZONE_YLIM)
        weight = zone.assemble_weighted_mass(_basis(bkd, nrefine), bkd)
        ones = np.ones(weight.shape[0])
        total = ones @ bkd.to_numpy(weight) @ ones
        np.testing.assert_allclose(total, _ZONE_AREA, rtol=1e-13)

    def test_symmetric_and_localized(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        basis = _basis(bkd, 1)
        zone = ElementAlignedRectangleZone(_ZONE_XLIM, _ZONE_YLIM)
        weight = bkd.to_numpy(zone.assemble_weighted_mass(basis, bkd))
        np.testing.assert_allclose(weight, weight.T, rtol=1e-14)
        # Rows of DOFs far from the zone are identically zero.
        coords = bkd.to_numpy(basis.dof_coordinates())
        far = coords[0] < 0.3
        np.testing.assert_allclose(weight[far], 0.0, atol=1e-15)

    def test_cut_elements_raise(self, numpy_bkd: NumpyBkd) -> None:
        """A rectangle whose edge falls inside elements must fail
        loudly, not silently commit O(1) quadrature error."""
        bkd = numpy_bkd
        zone = ElementAlignedRectangleZone((0.6, 0.69), (0.8, 0.97))
        with pytest.raises(ValueError, match="straddle"):
            zone.assemble_weighted_mass(_basis(bkd, 1), bkd)

    def test_empty_zone_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        # Inside obstruction block C: mesh has no elements there.
        zone = ElementAlignedRectangleZone(
            (4.0 / 7.0, 5.0 / 7.0), (0.5, 0.75)
        )
        with pytest.raises(ValueError, match="no elements"):
            zone.assemble_weighted_mass(_basis(bkd, 0), bkd)

    def test_outline_is_closed_rectangle(self) -> None:
        zone = ElementAlignedRectangleZone(_ZONE_XLIM, _ZONE_YLIM)
        outline = zone.outline_vertices()
        assert outline.shape == (2, 5)
        np.testing.assert_allclose(outline[:, 0], outline[:, -1])


class TestSmoothDiscZone:
    def test_protocol_conformance(self) -> None:
        zone = SmoothDiscZone((0.5, 0.85), 0.1, 0.02)
        assert isinstance(zone, ZoneWeightProtocol)

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SmoothDiscZone((0.5, 0.5), -0.1, 0.02)
        with pytest.raises(ValueError, match="positive"):
            SmoothDiscZone((0.5, 0.5), 0.1, 0.0)

    def test_weight_values_transition(self) -> None:
        zone = SmoothDiscZone((0.5, 0.5), 0.1, 0.01)
        pts = np.array([[0.5, 0.5, 0.5], [0.5, 0.6, 0.8]])
        vals = zone.weight_values(pts)
        assert vals[0] > 0.99  # deep inside
        np.testing.assert_allclose(vals[1], 0.5, atol=1e-12)  # on edge
        assert vals[2] < 0.01  # far outside

    def test_assembly_approximates_disc_area(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """1^T W 1 ~ pi r^2 when the transition layer is resolved.
        Quadrature assembly is APPROXIMATE — the tolerance here is
        orders looser than the aligned rectangle's 1e-13."""
        bkd = numpy_bkd
        # Disc in the open corridor left of the blocks.
        zone = SmoothDiscZone((0.15, 0.6), 0.1, 0.02)
        weight = zone.assemble_weighted_mass(_basis(bkd, 2), bkd)
        weight_np = bkd.to_numpy(weight)
        np.testing.assert_allclose(weight_np, weight_np.T, rtol=1e-14)
        ones = np.ones(weight_np.shape[0])
        total = ones @ weight_np @ ones
        np.testing.assert_allclose(total, np.pi * 0.1**2, rtol=5e-2)

    def test_outline_is_closed_circle(self) -> None:
        zone = SmoothDiscZone((0.5, 0.85), 0.1, 0.02)
        outline = zone.outline_vertices()
        assert outline.shape == (2, 65)
        np.testing.assert_allclose(outline[:, 0], outline[:, -1])
        radii = np.hypot(outline[0] - 0.5, outline[1] - 0.85)
        np.testing.assert_allclose(radii, 0.1, rtol=1e-12)
