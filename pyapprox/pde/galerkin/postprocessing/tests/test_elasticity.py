"""Tests for elasticity post-processing: strain, stress, von Mises.

Tests use known analytical solutions on simple meshes to verify
correctness of the strain recovery, stress computation, and von Mises
formula.
"""


import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.test_utils import slow_test

from pyapprox.pde.galerkin.postprocessing.elasticity import (
    strain_from_displacement_2d,
    stress_from_strain_2d,
    von_mises_stress_2d,
)


class TestStrainRecoveryQuad:
    """Test strain recovery on a single quad element."""

    def _single_quad(self):
        """Return a single unit-square quad: nodes at corners."""
        coordx = np.array([0.0, 1.0, 1.0, 0.0])
        coordy = np.array([0.0, 0.0, 1.0, 1.0])
        conn = np.array([[0, 1, 2, 3]])
        return coordx, coordy, conn

    def test_uniform_x_extension(self, numpy_bkd):
        """Uniform exx=0.1 from ux=0.1*x, uy=0."""
        bkd = numpy_bkd
        coordx, coordy, conn = self._single_quad()
        ux = 0.1 * coordx
        uy = np.zeros_like(coordy)
        exx, eyy, exy = strain_from_displacement_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
        )
        np.testing.assert_allclose(exx, [0.1], atol=1e-14)
        np.testing.assert_allclose(eyy, [0.0], atol=1e-14)
        np.testing.assert_allclose(exy, [0.0], atol=1e-14)

    def test_uniform_y_extension(self, numpy_bkd):
        """Uniform eyy=0.2 from uy=0.2*y, ux=0."""
        bkd = numpy_bkd
        coordx, coordy, conn = self._single_quad()
        ux = np.zeros_like(coordx)
        uy = 0.2 * coordy
        exx, eyy, exy = strain_from_displacement_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
        )
        np.testing.assert_allclose(exx, [0.0], atol=1e-14)
        np.testing.assert_allclose(eyy, [0.2], atol=1e-14)
        np.testing.assert_allclose(exy, [0.0], atol=1e-14)

    def test_pure_shear(self, numpy_bkd):
        """Pure shear: ux=gamma*y, uy=0 gives exy=gamma/2."""
        bkd = numpy_bkd
        coordx, coordy, conn = self._single_quad()
        gamma = 0.05
        ux = gamma * coordy
        uy = np.zeros_like(coordy)
        exx, eyy, exy = strain_from_displacement_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
        )
        np.testing.assert_allclose(exx, [0.0], atol=1e-14)
        np.testing.assert_allclose(eyy, [0.0], atol=1e-14)
        np.testing.assert_allclose(exy, [gamma / 2], atol=1e-14)

    def test_rigid_body_rotation(self, numpy_bkd):
        """Small rigid rotation: ux=-theta*y, uy=theta*x gives zero strain."""
        bkd = numpy_bkd
        coordx, coordy, conn = self._single_quad()
        theta = 0.01
        ux = -theta * coordy
        uy = theta * coordx
        exx, eyy, exy = strain_from_displacement_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
        )
        np.testing.assert_allclose(exx, [0.0], atol=1e-14)
        np.testing.assert_allclose(eyy, [0.0], atol=1e-14)
        np.testing.assert_allclose(exy, [0.0], atol=1e-14)

    def test_non_unit_quad(self, numpy_bkd):
        """Strain recovery on a scaled/translated quad."""
        bkd = numpy_bkd
        coordx = np.array([2.0, 5.0, 5.0, 2.0])
        coordy = np.array([1.0, 1.0, 4.0, 4.0])
        conn = np.array([[0, 1, 2, 3]])
        # ux = 0.1*x, uy = -0.05*y
        ux = 0.1 * coordx
        uy = -0.05 * coordy
        exx, eyy, exy = strain_from_displacement_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
        )
        np.testing.assert_allclose(exx, [0.1], atol=1e-14)
        np.testing.assert_allclose(eyy, [-0.05], atol=1e-14)
        np.testing.assert_allclose(exy, [0.0], atol=1e-14)


class TestStrainRecoveryTri:
    """Test strain recovery on triangular elements."""

    def _single_tri(self):
        coordx = np.array([0.0, 1.0, 0.0])
        coordy = np.array([0.0, 0.0, 1.0])
        conn = np.array([[0, 1, 2]])
        return coordx, coordy, conn

    def test_uniform_x_extension(self, numpy_bkd):
        bkd = numpy_bkd
        coordx, coordy, conn = self._single_tri()
        ux = 0.1 * coordx
        uy = np.zeros(3)
        exx, eyy, exy = strain_from_displacement_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
        )
        np.testing.assert_allclose(exx, [0.1], atol=1e-14)
        np.testing.assert_allclose(eyy, [0.0], atol=1e-14)
        np.testing.assert_allclose(exy, [0.0], atol=1e-14)

    def test_pure_shear(self, numpy_bkd):
        bkd = numpy_bkd
        coordx, coordy, conn = self._single_tri()
        gamma = 0.05
        ux = gamma * coordy
        uy = np.zeros(3)
        exx, eyy, exy = strain_from_displacement_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
        )
        np.testing.assert_allclose(exx, [0.0], atol=1e-14)
        np.testing.assert_allclose(eyy, [0.0], atol=1e-14)
        np.testing.assert_allclose(exy, [gamma / 2], atol=1e-14)


class TestStressFromStrain:
    """Test Hooke's law stress computation."""

    def test_uniaxial_tension(self, numpy_bkd):
        """Uniaxial tension: exx=e, eyy=exy=0."""
        bkd = numpy_bkd
        e = 0.001
        E, nu = 1e4, 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        exx = np.array([e])
        eyy = np.array([0.0])
        exy = np.array([0.0])
        sxx, syy, sxy = stress_from_strain_2d(
            exx,
            eyy,
            exy,
            np.array([lam]),
            np.array([mu]),
        )
        # sigma_xx = (lam + 2*mu) * e
        expected_sxx = (lam + 2 * mu) * e
        expected_syy = lam * e
        np.testing.assert_allclose(sxx, [expected_sxx], rtol=1e-14)
        np.testing.assert_allclose(syy, [expected_syy], rtol=1e-14)
        np.testing.assert_allclose(sxy, [0.0], atol=1e-14)

    def test_pure_shear_stress(self, numpy_bkd):
        """Pure shear: only exy nonzero."""
        bkd = numpy_bkd
        gamma_half = 0.01
        mu = 5000.0
        lam = 3000.0
        exx = np.array([0.0])
        eyy = np.array([0.0])
        exy = np.array([gamma_half])
        sxx, syy, sxy = stress_from_strain_2d(
            exx,
            eyy,
            exy,
            np.array([lam]),
            np.array([mu]),
        )
        np.testing.assert_allclose(sxx, [0.0], atol=1e-14)
        np.testing.assert_allclose(syy, [0.0], atol=1e-14)
        np.testing.assert_allclose(sxy, [2 * mu * gamma_half], rtol=1e-14)

    def test_multi_element(self, numpy_bkd):
        """Vectorized stress over multiple elements."""
        bkd = numpy_bkd
        nelems = 5
        exx = np.linspace(0.001, 0.005, nelems)
        eyy = np.zeros(nelems)
        exy = np.zeros(nelems)
        lam = np.full(nelems, 3000.0)
        mu = np.full(nelems, 5000.0)
        sxx, syy, sxy = stress_from_strain_2d(exx, eyy, exy, lam, mu)
        expected_sxx = (lam + 2 * mu) * exx
        expected_syy = lam * exx
        np.testing.assert_allclose(sxx, expected_sxx, rtol=1e-14)
        np.testing.assert_allclose(syy, expected_syy, rtol=1e-14)


class TestVonMisesStress:
    """Test von Mises stress computation."""

    def test_uniaxial_tension_vm(self, numpy_bkd):
        """Von Mises = |sigma_xx| for uniaxial tension."""
        bkd = numpy_bkd
        coordx = np.array([0.0, 1.0, 1.0, 0.0])
        coordy = np.array([0.0, 0.0, 1.0, 1.0])
        conn = np.array([[0, 1, 2, 3]])
        e = 0.001
        ux = e * coordx
        uy = np.zeros(4)
        E, nu = 1e4, 0.3
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu_val = E / (2 * (1 + nu))
        lam_arr = np.array([lam])
        mu_arr = np.array([mu_val])

        vm = von_mises_stress_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
            lam_arr,
            mu_arr,
        )

        # For uniaxial: sxx = (lam+2mu)*e, syy = lam*e, sxy = 0
        sxx = (lam + 2 * mu_val) * e
        syy = lam * e
        expected_vm = np.sqrt(sxx**2 - sxx * syy + syy**2)
        np.testing.assert_allclose(vm, [expected_vm], rtol=1e-12)

    def test_pure_shear_vm(self, numpy_bkd):
        """Von Mises = sqrt(3)*|tau| for pure shear."""
        bkd = numpy_bkd
        coordx = np.array([0.0, 1.0, 1.0, 0.0])
        coordy = np.array([0.0, 0.0, 1.0, 1.0])
        conn = np.array([[0, 1, 2, 3]])
        gamma = 0.01
        ux = gamma * coordy
        uy = gamma * coordx
        mu_val = 5000.0
        lam_arr = np.array([0.0])
        mu_arr = np.array([mu_val])

        vm = von_mises_stress_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
            lam_arr,
            mu_arr,
        )

        tau = 2 * mu_val * gamma  # sxy = 2*mu*exy, exy = gamma
        expected_vm = np.sqrt(3) * abs(tau)
        np.testing.assert_allclose(vm, [expected_vm], rtol=1e-12)

    def test_zero_displacement_vm_zero(self, numpy_bkd):
        """Von Mises = 0 when displacement is zero everywhere."""
        bkd = numpy_bkd
        coordx = np.array([0.0, 1.0, 1.0, 0.0])
        coordy = np.array([0.0, 0.0, 1.0, 1.0])
        conn = np.array([[0, 1, 2, 3]])
        ux = np.zeros(4)
        uy = np.zeros(4)
        lam_arr = np.array([3000.0])
        mu_arr = np.array([5000.0])

        vm = von_mises_stress_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
            lam_arr,
            mu_arr,
        )
        np.testing.assert_allclose(vm, [0.0], atol=1e-14)

    def test_equal_biaxial_tension_vm(self, numpy_bkd):
        """Equal biaxial tension: sxx=syy, sxy=0 gives known VM.

        In plane stress szz=0, so VM is not zero even for sxx=syy.
        VM = sqrt(sxx^2 - sxx*syy + syy^2) = |sxx| for sxx=syy.
        """
        bkd = numpy_bkd
        coordx = np.array([0.0, 1.0, 1.0, 0.0])
        coordy = np.array([0.0, 0.0, 1.0, 1.0])
        conn = np.array([[0, 1, 2, 3]])
        e = 0.001
        ux = e * coordx
        uy = e * coordy
        lam_arr = np.array([0.0])
        mu_arr = np.array([5000.0])

        vm = von_mises_stress_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
            lam_arr,
            mu_arr,
        )
        # sxx = syy = 2*mu*e = 10; VM = |sxx| = 10
        sxx = 2.0 * 5000.0 * e
        np.testing.assert_allclose(vm, [sxx], rtol=1e-12)

    def test_multi_element_mesh(self, numpy_bkd):
        """Von Mises on a 2x1 mesh (two quad elements)."""
        bkd = numpy_bkd
        # Two adjacent quads: [0,1,4,3] and [1,2,5,4]
        coordx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        coordy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        conn = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
        # Uniform extension
        e = 0.002
        ux = e * coordx
        uy = np.zeros(6)
        lam = 3000.0
        mu = 5000.0
        lam_arr = np.full(2, lam)
        mu_arr = np.full(2, mu)

        vm = von_mises_stress_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
            lam_arr,
            mu_arr,
        )
        # Both elements should have identical von Mises
        np.testing.assert_allclose(vm[0], vm[1], rtol=1e-14)
        # And match the single-element result
        sxx = (lam + 2 * mu) * e
        syy = lam * e
        expected = np.sqrt(sxx**2 - sxx * syy + syy**2)
        np.testing.assert_allclose(vm, [expected, expected], rtol=1e-12)

    def test_unsupported_element_raises(self, numpy_bkd):
        """Elements with neither 3 nor 4 nodes raise ValueError."""
        bkd = numpy_bkd
        coordx = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
        coordy = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        conn = np.array([[0, 1, 2, 3, 4]])
        with pytest.raises(ValueError):
            von_mises_stress_2d(
                coordx,
                coordy,
                conn,
                np.zeros(5),
                np.zeros(5),
                np.array([1.0]),
                np.array([1.0]),
            )


class TestVonMisesWithFEMSolve:
    """Integration test: von Mises from an actual FEM solve."""

    @slow_test
    def test_cantilever_beam_stress_positive(self, numpy_bkd):
        """Von Mises stress is non-negative and nonzero for loaded beam."""
        bkd = numpy_bkd
        from skfem.models.elasticity import lame_parameters

        from pyapprox.benchmarks.instances.pde.cantilever_beam import (
            _DEFAULT_MESH_PATH,
        )
        from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
        from pyapprox.pde.galerkin.boundary.implementations import (
            DirichletBC,
            NeumannBC,
        )
        from pyapprox.pde.galerkin.mesh import UnstructuredMesh2D
        from pyapprox.pde.galerkin.physics import (
            CompositeLinearElasticity,
        )
        from pyapprox.pde.galerkin.solvers import SteadyStateSolver
        from pyapprox.util.backends.numpy import NumpyBkd

        bkd = NumpyBkd()
        L, _H, q0 = 100.0, 30.0, 10.0

        mesh = UnstructuredMesh2D(
            _DEFAULT_MESH_PATH,
            bkd,
            rescale_origin=(0.0, 0.0),
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        skm = mesh.skfem_mesh()
        sub_names = mesh.subdomain_names()
        sub_elems = {n: mesh.subdomain_elements(n) for n in sub_names}

        material_map = {
            "bottom_layer": (2e4, 0.3),
            "inner_core": (5e3, 0.3),
            "top_layer": (2e4, 0.3),
        }

        bc_left = DirichletBC(
            basis,
            "left_edge",
            lambda c, t=0.0: np.zeros(c.shape[1]),
            bkd,
        )
        bc_top = NeumannBC(
            basis,
            "top_edge",
            lambda c, t=0.0: np.vstack(
                [
                    np.zeros(c.shape[1]),
                    -q0 * c[0] / L,
                ]
            ),
            bkd,
        )

        physics = CompositeLinearElasticity(
            basis=basis,
            material_map=material_map,
            element_materials=sub_elems,
            bkd=bkd,
            boundary_conditions=[bc_left, bc_top],
        )
        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=1)
        result = solver.solve(bkd.asarray(np.zeros(physics.nstates())))
        sol = bkd.to_numpy(result.solution)

        ux, uy = sol[0::2], sol[1::2]
        coordx, coordy = skm.p[0], skm.p[1]
        conn = skm.t.T

        # Build per-element Lame arrays
        nelems = conn.shape[0]
        lam_arr = np.empty(nelems)
        mu_arr = np.empty(nelems)
        for name, (E_val, nu_val) in material_map.items():
            lam_i, mu_i = lame_parameters(E_val, nu_val)
            lam_arr[sub_elems[name]] = lam_i
            mu_arr[sub_elems[name]] = mu_i

        vm = von_mises_stress_2d(
            coordx,
            coordy,
            conn,
            ux,
            uy,
            lam_arr,
            mu_arr,
        )

        # All non-negative
        assert np.all(vm >= 0)
        # At least some nonzero (beam is loaded)
        assert np.max(vm) > 0
        # Max stress should be near the clamped end (left)
        elem_centers_x = np.mean(coordx[conn], axis=1)
        max_stress_elem = np.argmax(vm)
        assert elem_centers_x[max_stress_elem] < L / 2
