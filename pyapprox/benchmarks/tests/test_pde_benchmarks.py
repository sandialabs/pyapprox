import unittest
import numpy as np
from functools import partial

from pyapprox.optimization.pya_minimize import pyapprox_minimize
from pyapprox.pde.autopde.mesh import cartesian_mesh_solution_functional
from pyapprox.util.utilities import (
    check_gradients, get_all_sample_combinations)
from pyapprox.benchmarks.benchmarks import setup_benchmark


class TestPDEBenchmarks(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_setup_inverse_advection_diffusion_benchmark(self):
        nobs = 10
        noise_std = 1e-8  # 0.01 make sure it does not dominate observed values
        nvars = 4
        benchmark = setup_benchmark(
            "advection_diffusion_kle_inversion", kle_nvars=nvars,
            noise_stdev=noise_std, nobs=nobs)
        inv_model, variable, true_params, noiseless_obs, obs = (
            benchmark.negloglike, benchmark.variable, benchmark.true_sample,
            benchmark.noiseless_obs, benchmark.obs)

        # inv_model.base_model._functional = partial(
        #     inv_model.base_model._functional, ignore_constants=True)
        # inv_model.base_model._adj_solver._dqdu = partial(
        #     inv_model.base_model._adj_solver._dqdu, ignore_constants=True)

        # TODO add std to params list
        init_guess = true_params + np.random.normal(0, 0.01, true_params.shape)
        # init_guess = variable.rvs(1)
        errors = check_gradients(
            partial(inv_model, jac=True),
            True, init_guess, plot=False,
            fd_eps=3*np.logspace(-12, 1, 14)[::-1])
        # print(np.log10(errors[0]/errors.min()))
        assert np.log10(errors[0]/errors.min()) > 5.3

        def scipy_obj(sample):
            vals, grad = inv_model(sample[:, None], jac=jac)
            return vals[:, 0], grad[0, :]

        jac = True
        opt_result = pyapprox_minimize(
            scipy_obj, init_guess,
            method="trust-constr", jac=jac,
            options={"verbose": 2, "gtol": 1e-7, "xtol": 1e-16})
        # print(opt_result.x)
        # print(true_params.T)
        # print(opt_result.x-true_params.T)
        assert np.allclose(opt_result.x, true_params.T, atol=2e-6)

    def test_setup_multi_index_advection_diffusion_benchmark(self):
        length_scale = .1
        sigma = 1
        nvars = 5

        config_values = [2*np.arange(1, 11), 2*np.arange(1, 11)]
        benchmark = setup_benchmark(
            "multi_index_advection_diffusion", kle_nvars=nvars,
            kle_length_scale=length_scale, kle_stdev=sigma,
            config_values=config_values)
        model, variable = benchmark.fun, benchmark.variable
        print(variable.num_vars())

        nrandom_samples = 10
        random_samples = variable.rvs(nrandom_samples)

        # import matplotlib.pyplot as plt
        # import torch
        # pde1 = model._model_ensemble.functions[-1]
        # mesh1 = pde1._fwd_solver.physics.mesh
        # kle_vals1 = pde1._kle(torch.as_tensor(random_samples[:, :1]))
        # pde2 = model._model_ensemble.functions[0]
        # mesh2 = pde2._fwd_solver.physics.mesh
        # kle_vals2 = pde2._kle(torch.as_tensor(random_samples[:, :1]))
        # kle_vals2 = mesh2.interpolate(kle_vals2, mesh1.mesh_pts)
        # im = mesh1.plot(
        #     kle_vals1-kle_vals2, 50,  ncontour_levels=30)
        # plt.colorbar(im)
        # fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
        # im1 = mesh1.plot(
        #     kle_vals1, 50,  ncontour_levels=30, ax=axs[0])
        # im2 = mesh1.plot(
        #     kle_vals2, 50,  ncontour_levels=30, ax=axs[1])
        # plt.colorbar(im1, ax=axs[0])
        # plt.colorbar(im2, ax=axs[1])
        # plt.show()

        config_samples = np.vstack([c[None, :] for c in config_values])
        samples = get_all_sample_combinations(random_samples, config_samples)
        values = model(samples)
        np.set_printoptions(precision=16)
        values = values.reshape((nrandom_samples, config_samples.shape[1]))
        qoi_means = values.mean(axis=0)
        rel_diffs = np.abs((qoi_means[-1]-qoi_means[:-1])/qoi_means[-1])
        print(rel_diffs)
        assert (rel_diffs.max() > 1e-1 and rel_diffs.min() < 3e-5)
        # ndof = (config_samples+1).prod(axis=0)
        # plt.loglog(
        #     ndof[:-1], np.abs((qoi_means[-1]-qoi_means[:-1])/qoi_means[-1]))
        # plt.show()

    def test_setup_transient_multi_index_advection_diffusion_benchmark(self):
        length_scale = .1
        sigma = 1
        nvars = 5
        final_time = .2

        time_scenario = {
            "final_time": final_time,
            "butcher_tableau": "im_crank2",
            "deltat": final_time/100,
            "init_sol_fun": None,
            # "init_sol_fun": partial(full_fun_axis_1, 0),
            "sink": [50, 0.1, [0.75, 0.75]]
        }
        nlevels = 8
        config_values = [2*np.arange(4, 4+nlevels), 2*np.arange(4, 4+nlevels),
                         final_time/(2**np.arange(1, 1+nlevels)*2)]
                         # np.array([final_time/(2**5*2)]*nlevels)]

        # subdomain_bounds = np.array([0.75, 1, 0, 0.25])
        # functional = partial(subdomain_integral_functional, subdomain_bounds)
        # functional = partial(final_time_functional, functional)
        domain_bounds = [0, 1, 0, 1]
        from pyapprox.util.visualization import get_meshgrid_samples
        XX, YY, xx = get_meshgrid_samples(domain_bounds, 30)
        tt = np.linspace(0, 1, 51)  # normalized time
        functional = partial(
            cartesian_mesh_solution_functional, xx, tt=tt)

        benchmark = setup_benchmark(
            "multi_index_advection_diffusion", kle_nvars=nvars,
            kle_length_scale=length_scale, kle_stdev=sigma,
            config_values=config_values, time_scenario=time_scenario,
            functional=functional)
        model, variable = benchmark.fun, benchmark.variable
        print(variable.num_vars())

        nrandom_samples = 1
        random_samples = variable.rvs(nrandom_samples)

        config_samples = np.vstack([c[None, :] for c in config_values])
        samples = get_all_sample_combinations(random_samples, config_samples)

        # values = model(samples)
        # idx = samples.shape[1]-1
        # nn = xx.shape[1]
        # fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
        # ims = []
        # ZZ_0 = values[idx, 0*nn:(0+1)*nn].reshape(XX.shape)
        # ZZ_N = values[idx, (len(tt)-1)*nn:(len(tt))*nn].reshape(XX.shape)
        # for dtx in range(len(tt)):
        #     im0 = []
        #     ZZ = values[idx, dtx*nn:(dtx+1)*nn].reshape(XX.shape)
        #     # print(ZZ.min(), ZZ.max(), dtx, tt[dtx])
        #     im = axs[0].contourf(
        #         XX, YY, ZZ, levels=np.linspace(ZZ_N.min(), ZZ_0.max(), 20),)
        #     im0 = im.collections
        #     II = 3*nn//4
        #     im1, = axs[1].plot((tt*final_time)[:dtx],
        #                        values[idx, II::nn][:dtx], c='k')
        #     im2, = axs[0].plot(xx[0, II], xx[1, II], 'ok', ms=20)
        #     ims.append(im0+[im1, im2])

        # import matplotlib.animation as animation
        # # interval in milliseconds
        # ani = animation.ArtistAnimation(
        #     fig, ims,
        #     # interval=final_time/tt.shape[0]/1e-3,
        #     interval=200,
        #     blit=True, repeat_delay=1000)
        # plt.show()

        benchmark = setup_benchmark(
            "multi_index_advection_diffusion", kle_nvars=nvars,
            kle_length_scale=length_scale, kle_stdev=sigma,
            config_values=config_values, time_scenario=time_scenario,
            functional=None)
        model, variable = benchmark.fun, benchmark.variable
        values = model(samples)
        
        np.set_printoptions(precision=16)
        values = values.reshape((nrandom_samples, config_samples.shape[1]))
        qoi_means = values.mean(axis=0)
        rel_diffs = np.abs((qoi_means[-1]-qoi_means[:-1])/qoi_means[-1])
        print(rel_diffs)
        ntsteps = time_scenario["final_time"]/config_samples[2]
        print(ntsteps)
        print(time_scenario["final_time"]-config_samples[2]*ntsteps)
        ndof = (config_samples[:2]+1).prod(axis=0)*ntsteps
        # plt.loglog(
        #     ndof[:-1], np.abs((qoi_means[-1]-qoi_means[:-1])/qoi_means[-1]))
        # plt.show()
        assert (rel_diffs.max() > 4e-2 and rel_diffs.min() < 8e-5)


if __name__ == "__main__":
    pde_benchmarks_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestPDEBenchmarks)
    unittest.TextTestRunner(verbosity=2).run(pde_benchmarks_test_suite)
