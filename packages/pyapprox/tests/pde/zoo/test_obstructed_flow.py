"""Tests for the obstructed-channel flow substrate."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.zoo.obstructed_flow import (
    ParabolicInlet,
    ZeroVelocity,
    build_obstructed_mesh,
    extract_velocity_callable,
    solve_obstructed_stokes,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestObstructedMeshBuilder:
    def test_boundary_labels(self, numpy_bkd: NumpyBkd) -> None:
        mesh = build_obstructed_mesh(numpy_bkd, nrefine=0)
        boundaries = set(mesh.skfem_mesh().boundaries)
        assert {
            "left", "right", "bottom", "top", "obs0", "obs1", "obs2"
        } <= boundaries

    def test_subdomain_edges_become_grid_lines(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        subdomain = (0.05, 0.25, 0.3, 0.6)
        mesh = build_obstructed_mesh(
            numpy_bkd, nrefine=1, subdomain=subdomain
        )
        coords = np.asarray(mesh.skfem_mesh().p)
        for xv in subdomain[:2]:
            assert np.any(np.abs(coords[0] - xv) <= 1e-12)
        for yv in subdomain[2:]:
            assert np.any(np.abs(coords[1] - yv) <= 1e-12)

    def test_subdomain_outside_domain_raises(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        with pytest.raises(ValueError, match="subdomain"):
            build_obstructed_mesh(
                numpy_bkd, nrefine=0, subdomain=(0.1, 1.2, 0.1, 0.5)
            )


class TestInletProfiles:
    def test_parabolic_inlet_shape_and_walls(self) -> None:
        inlet = ParabolicInlet(2.5, 2.5)
        pts = np.vstack([np.zeros(5), np.linspace(0.0, 1.0, 5)])
        vals = inlet(pts)
        assert vals.shape == (5, 2)
        np.testing.assert_allclose(vals[[0, -1], 0], 0.0, atol=1e-14)
        assert np.all(vals[1:-1, 0] > 0.0)
        np.testing.assert_allclose(vals[:, 1], 0.0, atol=1e-14)

    def test_zero_velocity(self) -> None:
        pts = np.random.default_rng(3).uniform(0, 1, (2, 4))
        np.testing.assert_allclose(ZeroVelocity()(pts), 0.0, atol=1e-14)

    def test_bc_callables_pickle(self) -> None:
        import pickle

        for obj in (ParabolicInlet(2.5, 2.5), ZeroVelocity()):
            assert isinstance(
                pickle.loads(pickle.dumps(obj)), type(obj)
            )


class TestObstructedStokes:
    def test_solve_and_extract_velocity(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Low-Re solve on the coarsest mesh: finite solution, inlet-
        driven positive x-flow, and shape-preserving velocity callable
        on a coarser transport basis."""
        bkd = numpy_bkd
        mesh = build_obstructed_mesh(bkd, nrefine=0)
        sol, stokes, vel_basis, pres_basis = solve_obstructed_stokes(
            mesh, bkd, reynolds_num=5.0, vel_shape_params=[2.5, 2.5]
        )
        sol_np = bkd.to_numpy(sol)
        assert np.all(np.isfinite(sol_np))

        adr_basis = LagrangeBasis(mesh, degree=1)
        velocity = extract_velocity_callable(
            sol, stokes, vel_basis, pres_basis, adr_basis, bkd
        )
        pts = np.array([[0.1, 0.15, 0.9], [0.5, 0.4, 0.5]])
        vals = velocity(pts)
        assert vals.shape == (2, 3)
        assert np.all(np.isfinite(vals))
        # The inlet drives flow left-to-right through the channel.
        assert np.all(vals[0] > 0.0)
        # Quadrature-shaped inputs preserve their trailing shape.
        pts_q = pts.reshape(2, 3, 1)
        assert velocity(pts_q).shape == (2, 3, 1)

    def test_probes_cache_shared(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = build_obstructed_mesh(bkd, nrefine=0)
        sol, stokes, vel_basis, pres_basis = solve_obstructed_stokes(
            mesh, bkd, reynolds_num=5.0, vel_shape_params=[2.5, 2.5]
        )
        adr_basis = LagrangeBasis(mesh, degree=1)
        cache: dict = {}
        velocity = extract_velocity_callable(
            sol, stokes, vel_basis, pres_basis, adr_basis, bkd,
            probes_cache=cache,
        )
        pts = np.array([[0.1, 0.9], [0.5, 0.5]])
        velocity(pts)
        assert 2 in cache
