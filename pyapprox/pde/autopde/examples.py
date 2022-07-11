#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from pyapprox.pde.autopde.solvers import (
    Function, SteadyStatePDE, SteadyStateAdjointPDE
)
from pyapprox.pde.autopde.physics import (
    AdvectionDiffusionReaction
)
from pyapprox.pde.autopde.mesh import (
    CartesianProductCollocationMesh
)
from pyapprox.pde.karhunen_loeve_expansion import MeshKLE

from pyapprox.util.utilities import check_gradients


def fwd_solver_finite_difference_wrapper(
        fwd_solver, functional, set_params, params, **newton_kwargs):
    # Warning newton tol must be smaller than finite difference step size
    set_params(fwd_solver.residual, torch.as_tensor(params[:, 0]))
    fd_sol = fwd_solver.solve(**newton_kwargs)
    qoi = np.asarray([functional(fd_sol, torch.as_tensor(params[:, 0]))])
    return qoi


def loglike_functional(obs, obs_indices, noise_std, sol, params):
    assert obs.ndim == 1 and sol.ndim == 1
    nobs = obs_indices.shape[0]
    tmp = 1/(2*noise_std**2)
    ll = 0.5*np.log(tmp/np.pi)*nobs
    pred_obs = sol[obs_indices]
    ll += -torch.sum((obs-pred_obs)**2*tmp)
    return ll


def loglike_functional_dqdu(obs, obs_indices, noise_std, sol, params):
    tmp = 1/(2*noise_std**2)
    pred_obs = sol[obs_indices]
    grad = torch.zeros_like(sol)
    grad[obs_indices] = (obs-pred_obs)*2*tmp
    return grad


def loglike_functional_dqdp(obs, obs_indices, noise_std, sol, params):
    return params*0


def advection_diffusion_reaction_kle_dRdp(kle, residual, sol, param_vals):
    mesh = residual.mesh
    dmats = [residual.mesh._dmat(dd) for dd in range(mesh.nphys_vars)]
    if kle.use_log:
        # compute gradient of diffusivity with respect to KLE coeff
        assert param_vals.ndim == 1
        kle_vals = kle(param_vals[:, None])
        assert kle_vals.ndim == 2
        dkdp = kle_vals*kle.eig_vecs
    Du = [torch.linalg.multi_dot((dmats[dd], sol))
          for dd in range(mesh.nphys_vars)]
    kDu = [Du[dd][:, None]*dkdp for dd in range(mesh.nphys_vars)]
    dRdp = sum([torch.linalg.multi_dot((dmats[dd], kDu[dd]))
               for dd in range(mesh.nphys_vars)])
    return dRdp


def zeros_fun_axis_1(x):
    # axis_1 used when x is mesh points
    return np.zeros((x.shape[1], 1))


def set_kle_diff_params(kle, residual, params):
    kle_vals = kle(params[:kle.nterms, None])
    residual._diff_fun = partial(residual.mesh.interpolate, kle_vals)


def advection_diffusion():

    true_noise_std = 1e-5 #0.01  # make sure it does not dominate observed values
    length_scale = .5
    nrandom_vars = 2
    true_kle_params = torch.as_tensor(
        np.random.normal(0, 1, (nrandom_vars)), dtype=torch.double)
    true_params = true_kle_params
    print(true_params.numpy())

    orders = [20, 20]
    obs_indices = np.array([200, 225, 300])
    # orders = [3, 3]
    # obs_indices = np.array([6, 7, 10])
    domain_bounds = [0, 1, 0, 1]
    mesh = CartesianProductCollocationMesh(domain_bounds, orders)

    kle = MeshKLE(mesh.mesh_pts, use_log=True, use_torch=True)
    kle.compute_basis(length_scale, sigma=1, nterms=nrandom_vars)

    plt.plot(np.arange(kle.nterms), kle.eig_vals, 'o')

    def vel_fun(xx):
        return torch.hstack((
            torch.ones(xx.shape[1], 1), torch.zeros(xx.shape[1], 1)))

    react_funs = [
        lambda sol: 0*sol,
        lambda sol: torch.zeros((sol.shape[0], sol.shape[0]))]

    def forc_fun(xx):
        amp, scale = 100.0, 0.1
        loc = torch.tensor([0.25, 0.75])[:, None]
        return amp*torch.exp(
            -torch.sum((torch.as_tensor(xx)-loc)**2/scale**2, axis=0))[:, None]

    bndry_conds = [
        [zeros_fun_axis_1, "D"],
        [zeros_fun_axis_1, "D"],
        [zeros_fun_axis_1, "D"],
        [zeros_fun_axis_1, "D"]]

    newton_kwargs = {"maxiters": 1, "tol": 1e-8}
    diff_fun = partial(mesh.interpolate, kle(true_kle_params[:, None]))
    fwd_solver = SteadyStatePDE(AdvectionDiffusionReaction(
        mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
        react_funs[1]))

    noise = np.random.normal(0, true_noise_std, (obs_indices.shape[0]))
    true_sol = fwd_solver.solve(**newton_kwargs)

    # p = mesh.plot(true_sol[:, None], nplot_pts_1d=50)
    # plt.colorbar(p)
    # plt.plot(mesh.mesh_pts[0, obs_indices], mesh.mesh_pts[1, obs_indices], 'ko')
    # plt.show() 
    
    obs = true_sol[obs_indices] + noise
    functional = partial(loglike_functional, obs, obs_indices, true_noise_std)
    dqdu = partial(loglike_functional_dqdu, obs, obs_indices, true_noise_std)
    dqdp = partial(loglike_functional_dqdp, obs, obs_indices, true_noise_std)
    dRdp = partial(advection_diffusion_reaction_kle_dRdp, kle)
    adj_solver = SteadyStateAdjointPDE(
        fwd_solver, functional, dqdu, dqdp, dRdp)
    set_params = partial(set_kle_diff_params, kle)

    def objective_single_sample(
            adj_solver, functional, params, **newton_kwargs):
        set_params(adj_solver.residual, torch.as_tensor(params))
        sol = adj_solver.solve(**newton_kwargs)
        obj = functional(sol, params)
        return obj

    from pyapprox.interface.wrappers import (
        evaluate_1darray_function_on_2d_array)
    objective = partial(
        evaluate_1darray_function_on_2d_array, partial(
            objective_single_sample, adj_solver, functional, **newton_kwargs))
    from pyapprox.util.visualization import get_meshgrid_function_data
    # X, Y, Z = get_meshgrid_function_data(objective, [0, 1, 0, 1], 10)
    # p = plt.contourf(X, Y, Z, np.linspace(Z.min(), Z.max(), 20))
    # X, Y, Z = get_meshgrid_function_data(
    #     adj_solver.residual._diff_fun, domain_bounds, 50)
    # X, Y, Z = get_meshgrid_function_data(
    #     partial(mesh.interpolate, kle.eig_vecs[:, 1]), domain_bounds, 50)
    # p = plt.contourf(X, Y, Z, np.linspace(Z.min(), Z.max(), 20))
    # plt.colorbar(p)
    # plt.show()

    # TODO add std to params list
    init_guess = (
        true_params[:, None] +
        np.random.normal(0, 1, (true_params.shape[0], 1)))
    errors = check_gradients(
        partial(fwd_solver_finite_difference_wrapper, fwd_solver,
                functional, set_params),
        lambda p: adj_solver.compute_gradient(
            set_params, torch.as_tensor(p)[:, 0], **newton_kwargs).numpy(),
        init_guess.numpy(), plot=False,
        fd_eps=np.logspace(-13, 0, 14)[::-1])

    from pyapprox.optimization.pya_minimize import pyapprox_minimize
    def objective(p):
        # scioy will pass in 1D variable
        obj, jac = adj_solver.compute_gradient(
            set_params, torch.as_tensor(p), return_obj=True,
            **newton_kwargs)
        if obj.ndim == 0:
            obj = torch.as_tensor([obj])
        # print(p, obj.item())
        # print(jac.numpy())
        print(obj.item())
        return -obj.numpy(), -jac[0, :].numpy()
    opt_result = pyapprox_minimize(
        objective, init_guess, method="trust-constr", jac=True)
    print(opt_result.x)
    print(true_params.numpy())


from pyapprox.benchmarks import setup_benchmark
def ecology():
    benchmark = setup_benchmark("hastings_ecology",
                                qoi_functional=lambda sol: sol[:, -2])
    nsamples = 10
    samples = benchmark.variable.rvs(nsamples)
    values = benchmark.fun(samples)
    print(values.shape)

if __name__ == "__main__":
    np.random.seed(1)
    # advection_diffusion()
    ecology()
