#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy import stats

from pyapprox.util.utilities import check_gradients
from pyapprox.analysis.visualize import get_meshgrid_function_data_from_variable

def fwd_solver_finite_difference_wrapper(
        fwd_solver, functional, set_params, params, **newton_kwargs):
    # Warning newton tol must be smaller than finite difference step size
    set_params(fwd_solver.physics, torch.as_tensor(params[:, 0]))
    fd_sol = fwd_solver.solve(**newton_kwargs)
    qoi = np.asarray([functional(fd_sol, torch.as_tensor(params[:, 0]))])
    return qoi


def set_kle_diff_params(kle, residual, params):
    kle_vals = kle(params[:kle.nterms, None])
    residual._diff_fun = partial(residual.mesh.interpolate, kle_vals)

from pyapprox.benchmarks.pde_benchmarks import (
    _setup_inverse_advection_diffusion_benchmark)


def advection_diffusion():

    nobs = 20
    noise_std = 1e-8 #0.01  # make sure it does not dominate observed values
    length_scale = .5
    sigma = 1
    nvars = 10
    orders = [20, 20]
    
    inv_model, variable, true_params, noiseless_obs, obs = (
        _setup_inverse_advection_diffusion_benchmark(
            nobs, noise_std, length_scale, sigma, nvars, orders))

    # TODO add std to params list
    init_guess = variable.rvs(1)
    # init_guess = true_params
    errors = check_gradients(
        lambda zz: inv_model._eval(zz[:, 0], return_grad=True),
        True, init_guess, plot=False,
        fd_eps=np.logspace(-13, 0, 14)[::-1])

    from pyapprox.optimization.pya_minimize import pyapprox_minimize
    return_grad = True
    opt_result = pyapprox_minimize(
        partial(inv_model._eval, return_grad=return_grad),
        init_guess,
        method="trust-constr", return_grad=return_grad,
        options={"verbose": 2, "gtol": 1e-6, "xtol": 1e-16})
    print(opt_result.x)
    print(true_params.T)

    if nvars != 2:
        return
    X, Y, Z = get_meshgrid_function_data_from_variable(inv_model, variable, 10)
    p = plt.contourf(X, Y, Z, np.linspace(Z.min(), Z.max(), 20))
    plt.plot(true_params[0], true_params[1], 'ko')
    # X, Y, Z = get_meshgrid_function_data(
    #     obs_model._fwd_solver.physics._diff_fun, domain_bounds, 50)
    # X, Y, Z = get_meshgrid_function_data(
    #     partial(mesh.interpolate, kle.eig_vecs[:, 1]), domain_bounds, 50)
    # p = plt.contourf(X, Y, Z, np.linspace(Z.min(), Z.max(), 20))
    plt.colorbar(p)
    plt.plot(opt_result.x[0], opt_result.x[1], 'ks')
    plt.show()


from pyapprox.benchmarks import setup_benchmark
from pyapprox.surrogates import adaptive_approximate, approximate
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.analysis.sensitivity_analysis import run_sensitivity_analysis
from pyapprox.surrogates.polychaos.gpc import (
    define_poly_options_from_variable_transformation, PolynomialChaosExpansion
)
from pyapprox.surrogates.polychaos.sparse_grid_to_gpc import (
    convert_sparse_grid_to_polynomial_chaos_expansion
)
def ecology():
    time = np.linspace(0., 10, 101)
    benchmark = setup_benchmark("hastings_ecology",
                                qoi_functional=lambda sol: sol[:, -2],
                                time=time)
    nsamples = 10
    train_samples = benchmark.variable.rvs(nsamples)
    train_values = benchmark.fun(train_samples)

    degree = 1
    indices = compute_hyperbolic_indices(
        benchmark.variable.num_vars(), degree)
    print(indices)
    # pce = approximate(
    #     train_samples, train_values, 'polynomial_chaos',
    #     {'basis_type': 'fixed', 'variable': benchmark.variable,
    #      'options': {'indices': indices, "solver_type": "lstsq"}}).approx
    sparse_grid = adaptive_approximate(
        benchmark.fun, benchmark.variable, 'sparse_grid',
        {"max_nsamples": 1000}).approx
    samples = benchmark.variable.rvs(100)
    sg_vals = sparse_grid(samples)
    vals = benchmark.fun(samples)
    error = np.linalg.norm(sg_vals-vals, axis=0)/np.linalg.norm(vals, axis=0)
    #print(error)
    #assert False

    #from pyapprox.analysis.visualize import plot_1d_cross_sections
    #plot_1d_cross_sections(benchmark.fun, benchmark.variable, nsamples_1d = 20)
    #plt.show()

    from pyapprox import analysis
    fig, axs = analysis.generate_parameter_sweeps_and_plot_from_variable(
        benchmark.fun, benchmark.variable, num_samples_per_sweep=20, num_sweeps=3,
        qoi_indices=np.array([-1]))
    plt.show()

    pce_opts = define_poly_options_from_variable_transformation(
        sparse_grid.var_trans)
    pce = convert_sparse_grid_to_polynomial_chaos_expansion(
        sparse_grid, pce_opts)
    res = run_sensitivity_analysis("pce_sobol", pce, benchmark.variable)

    bottom = 0
    for ii in range(res.main_effects.shape[0]):
        top = bottom + res.main_effects[ii, :]
        plt.fill_between(time, bottom, top)
        bottom = top
    plt.show()



if __name__ == "__main__":
    np.random.seed(2)
    advection_diffusion()
    # ecology()

#visualize profile
# gprof2dot -f pstats profile.out | dot -Tpng -o output.png && eog output.png
