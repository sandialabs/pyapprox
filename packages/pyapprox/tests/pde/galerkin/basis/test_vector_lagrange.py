"""Tests for VectorLagrangeBasis boundary DOF extraction."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh2D, StructuredMesh3D
from pyapprox.util.backends.numpy import NumpyBkd

_NumpyArray = NDArray[Any]


def _make_mesh(
    ndim: int, element_type: str, bkd: NumpyBkd
) -> Union[StructuredMesh2D[_NumpyArray], StructuredMesh3D[_NumpyArray]]:
    if ndim == 2:
        return StructuredMesh2D(
            nx=2,
            ny=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            bkd=bkd,
            element_type=element_type,
        )
    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        bkd=bkd,
        element_type=element_type,
    )


def _scalar_boundary_dofs(
    basis: VectorLagrangeBasis[_NumpyArray], boundary_name: str
) -> _NumpyArray:
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
        self,
        numpy_bkd: NumpyBkd,
        ndim: int,
        element_type: str,
        degree: int,
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
    def test_get_dofs_components_3d(
        self, numpy_bkd: NumpyBkd, degree: int
    ) -> None:
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

    def test_get_dofs_components_2d(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = _make_mesh(2, "quad", bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)

        comp_dofs = bkd.to_numpy(basis.get_dofs("top", components=(1,)))
        assert np.all(comp_dofs % 2 == 1)
        all_dofs = bkd.to_numpy(basis.get_dofs("top"))
        assert len(comp_dofs) == len(all_dofs) // 2

    def test_get_dofs_invalid_component_raises(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = _make_mesh(3, "hex", bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        with pytest.raises(ValueError, match="out of range"):
            basis.get_dofs("left", components=(3,))
        with pytest.raises(ValueError, match="out of range"):
            basis.get_dofs("left", components=(-1,))


class TestVectorLagrangeEvaluate:
    def test_evaluate_recovers_interpolated_linear_field(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """evaluate() inverts interpolate() for a P1-exact linear field."""
        bkd = numpy_bkd
        mesh = _make_mesh(2, "quad", bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)

        def field(x: _NumpyArray) -> _NumpyArray:
            return np.stack([1.0 + 2.0 * x[0] - x[1], 3.0 * x[1]], axis=0)

        coeffs = basis.interpolate(field)
        points = bkd.asarray(
            np.array([[0.3, 0.7, 0.5], [0.2, 0.9, 0.5]])
        )
        values = basis.evaluate(coeffs, points)
        expected = field(bkd.to_numpy(points))
        assert values.shape == (2, 3)
        bkd.assert_allclose(values, bkd.asarray(expected), rtol=1e-12)

    def test_evaluate_satisfies_basis_protocol(self, numpy_bkd: NumpyBkd) -> None:
        from pyapprox.pde.galerkin.protocols.basis import (
            GalerkinBasisProtocol,
        )

        basis = VectorLagrangeBasis(_make_mesh(2, "quad", numpy_bkd), degree=1)
        assert isinstance(basis, GalerkinBasisProtocol)
