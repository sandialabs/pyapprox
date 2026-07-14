"""Integration test: von Mises post-processing from an actual FEM solve.

Unit-level coverage of strain/stress/von Mises lives in
packages/pyapprox/tests/pde/galerkin/postprocessing/test_elasticity.py;
this tier exercises the post-processing on a benchmark mesh solve
(imports pyapprox + pyapprox_benchmarks).
"""


import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.galerkin.postprocessing import integrate, von_mises_stress
from tests._helpers.markers import slow_test


class TestVonMisesWithFEMSolve:
    """Integration test: von Mises from an actual FEM solve."""

    @slow_test
    def test_cantilever_beam_stress_positive(self, numpy_bkd):
        """Von Mises stress is non-negative and nonzero for loaded beam."""
        bkd = numpy_bkd
        from skfem.models.elasticity import lame_parameters

        from pyapprox_benchmarks.pde.cantilever_beam import (
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

        conn = skm.t.T

        # Build per-element Lame arrays
        nelems = conn.shape[0]
        lam_arr = np.empty(nelems)
        mu_arr = np.empty(nelems)
        for name, (E_val, nu_val) in material_map.items():
            lam_i, mu_i = lame_parameters(E_val, nu_val)
            lam_arr[sub_elems[name]] = lam_i
            mu_arr[sub_elems[name]] = mu_i

        # plane_strain matches the constitutive law the 2D physics
        # assembles (sigma = lam*tr(eps)*I + 2*mu*eps)
        vm = von_mises_stress(
            basis, result.solution, lam_arr, mu_arr, "plane_strain"
        )

        # All non-negative
        assert np.all(vm >= 0)
        # At least some nonzero (beam is loaded)
        assert np.max(vm) > 0
        # The domain-integrated stress is positive and finite
        total = integrate(basis, vm)
        assert np.isfinite(total) and total > 0
        # Max stress should be near the clamped end (left)
        coordx = skm.p[0]
        elem_centers_x = np.mean(coordx[conn], axis=1)
        vm_elem = vm.mean(axis=1)
        max_stress_elem = int(np.argmax(vm_elem))
        assert elem_centers_x[max_stress_elem] < L / 2
