"""Tests for elasticity post-processing (strain/stress/von Mises)."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np

from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh2D, StructuredMesh3D
from pyapprox.pde.galerkin.postprocessing import (
    integrate,
    strain_from_displacement,
    stress_from_strain,
    von_mises_stress,
)


def _make_basis(ndim, element_type, bkd, degree=1, nx=2):
    if ndim == 2:
        mesh = StructuredMesh2D(
            nx=nx,
            ny=nx,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
            element_type=element_type,
        )
    else:
        mesh = StructuredMesh3D(
            nx=nx,
            ny=nx,
            nz=nx,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
            element_type=element_type,
        )
    return VectorLagrangeBasis(mesh, degree=degree)


def _linear_displacement_dofs(basis, A, b, bkd):
    """DOF values of u = A x + b via basis interpolation."""
    return basis.interpolate(lambda x: A @ x + b[:, None])


class TestStrainFromDisplacement:

    @pytest.mark.parametrize(
        "ndim,element_type",
        [(2, "quad"), (2, "tri"), (3, "hex"), (3, "tet")],
    )
    @pytest.mark.parametrize("degree", [1, 2])
    def test_constant_strain_exact(
        self, numpy_bkd, ndim, element_type, degree
    ) -> None:
        """Linear displacement => exact constant strain at every
        quadrature point."""
        bkd = numpy_bkd
        rng = np.random.RandomState(0)
        A = 0.1 * rng.randn(ndim, ndim)
        b = 0.1 * rng.randn(ndim)
        basis = _make_basis(ndim, element_type, bkd, degree=degree)
        u = _linear_displacement_dofs(basis, A, b, bkd)

        strain = strain_from_displacement(basis, u)
        sym = 0.5 * (A + A.T)
        if ndim == 2:
            expected = [sym[0, 0], sym[1, 1], sym[0, 1]]
        else:
            expected = [
                sym[0, 0],
                sym[1, 1],
                sym[2, 2],
                sym[0, 1],
                sym[0, 2],
                sym[1, 2],
            ]
        for row, val in zip(strain, expected):
            bkd.assert_allclose(
                bkd.asarray(row),
                bkd.full(row.shape, val),
                atol=1e-13,
            )

    def test_1d_raises(self, numpy_bkd) -> None:
        from pyapprox.pde.galerkin.mesh import StructuredMesh1D

        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=3, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        u = bkd.zeros((basis.ndofs(),))
        with pytest.raises(ValueError, match="2D or 3D"):
            strain_from_displacement(basis, u)


class TestStressFromStrain:

    def test_uniaxial_strain_all_assumptions(self, numpy_bkd) -> None:
        """Hand-computed uniaxial strain e_xx = e0, all else zero."""
        bkd = numpy_bkd
        lam, mu = 2.0, 3.0
        e0 = 0.01
        shape = (4, 5)

        # 3d
        strain3 = np.zeros((6,) + shape)
        strain3[0] = e0
        s3 = stress_from_strain(strain3, lam, mu, "3d")
        bkd.assert_allclose(
            bkd.asarray(s3[0]), bkd.full(shape, (lam + 2 * mu) * e0)
        )
        bkd.assert_allclose(bkd.asarray(s3[1]), bkd.full(shape, lam * e0))
        bkd.assert_allclose(bkd.asarray(s3[2]), bkd.full(shape, lam * e0))

        # plane strain: same in-plane law, s_zz = lam*tr
        strain2 = np.zeros((3,) + shape)
        strain2[0] = e0
        s_ps = stress_from_strain(strain2, lam, mu, "plane_strain")
        bkd.assert_allclose(
            bkd.asarray(s_ps[0]), bkd.full(shape, (lam + 2 * mu) * e0)
        )
        bkd.assert_allclose(bkd.asarray(s_ps[2]), bkd.full(shape, lam * e0))

        # plane stress: s_zz = 0, effective lambda
        lam_eff = 2.0 * lam * mu / (lam + 2.0 * mu)
        s_pt = stress_from_strain(strain2, lam, mu, "plane_stress")
        bkd.assert_allclose(
            bkd.asarray(s_pt[0]), bkd.full(shape, (lam_eff + 2 * mu) * e0)
        )
        bkd.assert_allclose(bkd.asarray(s_pt[2]), bkd.zeros(shape))

    def test_plane_strain_matches_3d_embedding(self, numpy_bkd) -> None:
        """2D strain + plane_strain == embedded 3D strain (e_zz=0) + 3d."""
        bkd = numpy_bkd
        rng = np.random.RandomState(1)
        shape = (4, 5)
        strain2 = 0.01 * rng.randn(3, *shape)
        lam = 1.0 + rng.rand(4)  # per-element
        mu = 0.5 + rng.rand(4)

        s2 = stress_from_strain(strain2, lam, mu, "plane_strain")
        strain3 = np.zeros((6,) + shape)
        strain3[0], strain3[1], strain3[3] = strain2
        s3 = stress_from_strain(strain3, lam, mu, "3d")
        bkd.assert_allclose(bkd.asarray(s2), bkd.asarray(s3), rtol=1e-14)

    def test_invalid_assumption_and_shape_raise(self, numpy_bkd) -> None:
        strain2 = np.zeros((3, 2, 2))
        strain3 = np.zeros((6, 2, 2))
        with pytest.raises(ValueError, match="assumption must be"):
            stress_from_strain(strain2, 1.0, 1.0, "planestress")
        with pytest.raises(ValueError, match="requires 6 strain"):
            stress_from_strain(strain2, 1.0, 1.0, "3d")
        with pytest.raises(ValueError, match="requires 3 strain"):
            stress_from_strain(strain3, 1.0, 1.0, "plane_strain")


class TestVonMises:

    def test_pure_shear(self, numpy_bkd) -> None:
        """VM of pure shear stress tau is sqrt(3)*tau (any assumption)."""
        bkd = numpy_bkd
        mu = 3.0
        gamma = 0.01  # tensor shear strain
        tau = 2.0 * mu * gamma
        strain2 = np.zeros((3, 2, 2))
        strain2[2] = gamma
        for assumption in ["plane_stress", "plane_strain"]:
            stress = stress_from_strain(strain2, 2.0, mu, assumption)
            sxx, syy, szz, sxy, sxz, syz = stress
            vm = np.sqrt(
                0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                + 3.0 * (sxy**2 + sxz**2 + syz**2)
            )
            bkd.assert_allclose(
                bkd.asarray(vm),
                bkd.full((2, 2), np.sqrt(3.0) * tau),
                rtol=1e-14,
            )

    @pytest.mark.parametrize(
        "ndim,element_type,assumption",
        [(2, "quad", "plane_strain"), (3, "hex", "3d")],
    )
    def test_von_mises_linear_displacement(
        self, numpy_bkd, ndim, element_type, assumption
    ) -> None:
        """VM of a linear displacement equals the hand-computed constant."""
        bkd = numpy_bkd
        lam, mu = 2.0, 3.0
        rng = np.random.RandomState(2)
        A = 0.1 * rng.randn(ndim, ndim)
        b = np.zeros(ndim)
        basis = _make_basis(ndim, element_type, bkd)
        u = _linear_displacement_dofs(basis, A, b, bkd)

        vm = von_mises_stress(basis, u, lam, mu, assumption)

        # hand computation from the exact constant strain
        sym = 0.5 * (A + A.T)
        eps3 = np.zeros((3, 3))
        eps3[:ndim, :ndim] = sym  # e_zz = 0: plane strain when ndim == 2
        sigma = lam * np.trace(eps3) * np.eye(3) + 2.0 * mu * eps3
        dev = sigma - np.trace(sigma) / 3.0 * np.eye(3)
        vm_exact = np.sqrt(1.5 * np.sum(dev * dev))
        bkd.assert_allclose(
            bkd.asarray(vm), bkd.full(vm.shape, vm_exact), rtol=1e-12
        )


class TestIntegrate:

    @pytest.mark.parametrize(
        "ndim,element_type",
        [(2, "quad"), (2, "tri"), (3, "hex"), (3, "tet")],
    )
    def test_unit_field_gives_volume(self, numpy_bkd, ndim, element_type):
        bkd = numpy_bkd
        basis = _make_basis(ndim, element_type, bkd)
        dx_shape = np.asarray(basis.skfem_basis().dx).shape
        total = integrate(basis, np.ones(dx_shape))
        bkd.assert_allclose(
            bkd.asarray([total]), bkd.asarray([1.0]), rtol=1e-13
        )

    def test_shape_mismatch_raises(self, numpy_bkd):
        basis = _make_basis(2, "quad", numpy_bkd)
        with pytest.raises(ValueError, match="does not match quadrature"):
            integrate(basis, np.ones((1, 1)))
