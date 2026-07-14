"""Tests for VectorLagrangeBasis boundary DOF extraction."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np

from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh2D, StructuredMesh3D


def _make_mesh(ndim, element_type, bkd):
    if ndim == 2:
        return StructuredMesh2D(
            nx=2,
            ny=2,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
            element_type=element_type,
        )
    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        bkd=bkd,
        element_type=element_type,
    )


def _scalar_boundary_dofs(basis, boundary_name):
    """Boundary DOFs of the underlying scalar basis, via skfem directly."""
    scalar_skfem = basis.scalar_basis().skfem_basis()
    return np.unique(np.asarray(scalar_skfem.get_dofs(boundary_name)).flatten())


class TestVectorLagrangeGetDofs:

    @pytest.mark.parametrize(
        "ndim,element_type",
        [(2, "quad"), (2, "tri"), (3, "hex"), (3, "tet")],
    )
    @pytest.mark.parametrize("degree", [1, 2])
    def test_get_dofs_no_duplicates(
        self, numpy_bkd, ndim, element_type, degree
    ) -> None:
        """Regression: 3D boundary DOFs were each returned twice.

        Checks no duplicates and exact agreement with the interleaved
        expansion of the scalar basis boundary DOFs
        (vector_dof = scalar_dof * ndim + component).
        """
        bkd = numpy_bkd
        mesh = _make_mesh(ndim, element_type, bkd)
        basis = VectorLagrangeBasis(mesh, degree=degree)
        boundary_name = "left"

        dofs = bkd.to_numpy(basis.get_dofs(boundary_name))
        assert len(dofs) == len(np.unique(dofs))

        scalar_dofs = _scalar_boundary_dofs(basis, boundary_name)
        assert len(dofs) == ndim * len(scalar_dofs)
        expected = np.unique(
            np.asarray(
                [d * ndim + c for d in scalar_dofs for c in range(ndim)]
            )
        )
        np.testing.assert_array_equal(dofs, expected)

    @pytest.mark.parametrize("degree", [1, 2])
    def test_get_dofs_components_3d(self, numpy_bkd, degree) -> None:
        """components=(c,) returns exactly the DOFs of component c."""
        bkd = numpy_bkd
        mesh = _make_mesh(3, "hex", bkd)
        basis = VectorLagrangeBasis(mesh, degree=degree)
        boundary_name = "back"

        all_dofs = bkd.to_numpy(basis.get_dofs(boundary_name))
        scalar_count = len(_scalar_boundary_dofs(basis, boundary_name))

        component_sets = []
        for comp in range(3):
            comp_dofs = bkd.to_numpy(
                basis.get_dofs(boundary_name, components=(comp,))
            )
            # interleaved DOFs: component of DOF d is d % ndim
            assert np.all(comp_dofs % 3 == comp)
            assert len(comp_dofs) == scalar_count
            component_sets.append(set(comp_dofs.tolist()))

        union = component_sets[0] | component_sets[1] | component_sets[2]
        assert union == set(all_dofs.tolist())

        pair_dofs = bkd.to_numpy(
            basis.get_dofs(boundary_name, components=(0, 2))
        )
        assert set(pair_dofs.tolist()) == component_sets[0] | component_sets[2]

    def test_get_dofs_components_2d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = _make_mesh(2, "quad", bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)

        comp_dofs = bkd.to_numpy(basis.get_dofs("top", components=(1,)))
        assert np.all(comp_dofs % 2 == 1)
        all_dofs = bkd.to_numpy(basis.get_dofs("top"))
        assert len(comp_dofs) == len(all_dofs) // 2

    def test_get_dofs_invalid_component_raises(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = _make_mesh(3, "hex", bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        with pytest.raises(ValueError, match="out of range"):
            basis.get_dofs("left", components=(3,))
        with pytest.raises(ValueError, match="out of range"):
            basis.get_dofs("left", components=(-1,))
