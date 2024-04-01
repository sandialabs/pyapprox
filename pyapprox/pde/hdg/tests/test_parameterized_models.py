import unittest

import numpy as np

from pyapprox.pde.hdg.parameterized_models import (
    SteadyObstructedFlowModel, FEMIntegrateRectangularSubdomain)
from pyapprox.pde.galerkin.util import _get_mesh, _get_element
from skfem import Basis
from pyapprox.pde.galerkin.physics import (
    LinearAdvectionDiffusionReaction, AdvectionDiffusionReaction)
from pyapprox.pde.galerkin.solvers import SteadyStatePDE


class TestParameterizedPDEs(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_obstructed_flow(self):
        # domain_bounds = [0, 1]
        # mesh = _get_mesh(domain_bounds, 2)
        # element = _get_element(mesh, 1)
        # basis = Basis(mesh, element)
        # D_bndry_conds = {"left": [lambda x: 0*x], "right": [lambda x: 0*x]}
        # bndry_conds = [D_bndry_conds, {}, {}]
        # diff_fun = lambda x: 0*x[0] + 1.
        # forc_fun = lambda x: 0*x[0] + 1
        # vel_fun = lambda x: 0*x
        # nl_diff_funs = [lambda x, sol: diff_fun(x), lambda x, sol: x[0]*0]
        # # physics = LinearAdvectionDiffusionReaction(
        # #     mesh, element, basis, bndry_conds, diff_fun, forc_fun, vel_fun)
        # physics = AdvectionDiffusionReaction(
        #     mesh, element, basis, bndry_conds, diff_fun, forc_fun, vel_fun,
        #     nl_diff_funs, [None, None])
        # solver = SteadyStatePDE(physics)
        # sol = solver.solve(maxiters=1)

        # physics = LinearAdvectionDiffusionReaction(
        #     mesh, element, basis, bndry_conds, diff_fun, forc_fun, vel_fun)
        # sol = physics.init_guess()

        # import matplotlib.pyplot as plt
        # mesh_pts = np.sort(mesh.p)
        # plt.plot(mesh_pts[0], basis.interpolator(sol)(mesh_pts), '-ok')
        # plt.show()
        # assert False

        L = 3
        nrefine = 4
        orders = [25, 25]
        re_num = 500
        flow_type = "navier_stokes"
        bndry_info = [1, 1, 10, 1.]
        source_info = [10, 0.2, 0.9, 0.9]
        vel_filename = None  # f"obstructed_navier_stokes_velocity_{nrefine}.pkl"

        hdg_model = SteadyObstructedFlowModel(
            L, orders, bndry_info, source_info, vel_filename=vel_filename,
            reynolds_num=re_num, flow_type=flow_type, tracer_solver_type="hdg",
            nrefine=nrefine)
        log_diff = np.log(0.05)
        sample = np.array(source_info+[log_diff])[:, None]
        import time
        t0 = time.time()
        hdg_csols = hdg_model._solve(sample[:, 0])
        print("hdg time", time.time()-t0)

        fem_model = SteadyObstructedFlowModel(
            L, orders, bndry_info, source_info, vel_filename=vel_filename,
            reynolds_num=re_num, flow_type=flow_type, tracer_solver_type="fem",
            nrefine=nrefine)
        t0 = time.time()
        fem_sol = fem_model._solve(sample[:, 0])

        t0 = time.time()
        integrate = FEMIntegrateRectangularSubdomain(
            fem_model.pressure_solver._decomp._subdomain_bounds[0])
        integral = integrate.assemble(
            fem_model._tracer_fem_basis,
            y=fem_model._tracer_fem_basis.interpolate(fem_sol)*0+1)
        print("integrate time", time.time()-t0)
        assert np.allclose(integral, 2/7)

        t0 = time.time()
        fem_interp = fem_model._tracer_fem_basis.interpolator(fem_sol)
        fem_csols = [
            fem_interp(subdomain_model.physics.mesh.mesh_pts)
            for ii, subdomain_model in enumerate(
                    hdg_model.tracer_solver._decomp._subdomain_models)]
        print("interp time", time.time()-t0)
        sq_diffs = [(hdg_sol-fem_sol)**2
                    for hdg_sol, fem_sol in zip(hdg_csols, fem_csols)]
        assert hdg_model.tracer_solver._decomp.integrate(sq_diffs) < 3.1e-3

        # import matplotlib.pyplot as plt
        # from pyapprox.util.visualization import get_meshgrid_samples
        # from skfem.visuals.matplotlib import draw
        # X, Y, xx = get_meshgrid_samples(hdg_model.domain_bounds, 101)
        # axs = plt.subplots(1, 2, figsize=(6*8, 6))[1]
        # draw(fem_model.tracer_solver.physics.mesh, ax=axs[1])
        # hdg_model.plot_velocities(axs[0], cmap='coolwarm', levels=21)
        # Z = hdg_model.tracer_domain_decomp.interpolate(
        #     hdg_csols, xx, default_val=-np.inf).reshape(X.shape)
        # im = axs[1].contourf(
        #     X, Y, Z, levels=np.linspace(Z[np.isfinite(Z)].min(), Z.max(), 51),
        #     cmap="coolwarm")
        # plt.colorbar(im, ax=axs[1])

        # X, Y, xx = get_meshgrid_samples(fem_model.domain_bounds, 101)
        # axs = plt.subplots(1, 2, figsize=(6*8, 6))[1]
        # fem_model.plot_velocities(axs[0], cmap='coolwarm', levels=21)
        # Z = hdg_model.tracer_domain_decomp.interpolate(
        #     fem_csols, xx, default_val=-np.inf).reshape(X.shape)
        # im = axs[1].contourf(
        #     X, Y, Z, levels=np.linspace(Z[np.isfinite(Z)].min(), Z.max(), 51),
        #     cmap="coolwarm")
        # plt.colorbar(im, ax=axs[1])
        # plt.show()


if __name__ == "__main__":
    parameterized_pdes_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestParameterizedPDEs)
    unittest.TextTestRunner(verbosity=2).run(parameterized_pdes_test_suite)
