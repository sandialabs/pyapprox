import torch
from functools import partial
from scipy import stats
import numpy as np

from pyapprox.pde.autopde.solvers import (
    SteadyStatePDE, SteadyStateAdjointPDE
)
from pyapprox.pde.autopde.physics import (
    AdvectionDiffusionReaction
)
from pyapprox.pde.autopde.mesh import (
    full_fun_axis_1, CartesianProductCollocationMesh
)
from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.pde.karhunen_loeve_expansion import MeshKLE
from pyapprox.interface.wrappers import evaluate_1darray_function_on_2d_array
from pyapprox.variables.transforms import ConfigureVariableTransformation


def constant_vel_fun(vels, xx):
    return torch.hstack([
        full_fun_axis_1(vels[ii], xx, oned=False) for ii in range(len(vels))])


def gauss_forc_fun(amp, scale, loc, xx):
    return amp*torch.exp(
        -torch.sum((torch.as_tensor(xx)-loc)**2/scale**2, axis=0))[:, None]


def mesh_locations_obs_functional(obs_indices, sol, params):
    return sol[obs_indices]


def negloglike_functional(obs, obs_indices, noise_std, sol, params):
    assert obs.ndim == 1 and sol.ndim == 1
    nobs = obs_indices.shape[0]
    tmp = 1/(2*noise_std**2)
    ll = 0.5*np.log(tmp/np.pi)*nobs
    pred_obs = sol[obs_indices]
    ll += -torch.sum((obs-pred_obs)**2*tmp)
    return -ll


def negloglike_functional_dqdu(obs, obs_indices, noise_std, sol, params):
    tmp = 1/(2*noise_std**2)
    pred_obs = sol[obs_indices]
    grad = torch.zeros_like(sol)
    grad[obs_indices] = (obs-pred_obs)*2*tmp
    return -grad


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


class AdvectionDiffusionReactionKLEModel():
    def __init__(self, mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
                 functional, functional_deriv_funs=[None, None]):
        self._kle = kle
        # TODO pass in parameterized functions for diffusiviy and forcing and
        # reaction and use same process as used for KLE currently

        if react_funs is None:
            react_funs = [
                lambda sol: 0*sol,
                lambda sol: torch.zeros((sol.shape[0], sol.shape[0]))]

        self._fwd_solver = SteadyStatePDE(AdvectionDiffusionReaction(
            mesh, bndry_conds, partial(full_fun_axis_1, 1), vel_fun,
            react_funs[0], forc_fun, react_funs[1]))
        self._functional = functional

        dqdu, dqdp = functional_deriv_funs
        dRdp = partial(advection_diffusion_reaction_kle_dRdp, self._kle)
        self._adj_solver = SteadyStateAdjointPDE(
            self._fwd_solver, self._functional, dqdu, dqdp, dRdp)

        self._mesh_basis_mat = mesh._get_lagrange_basis_mat(
            mesh._canonical_mesh_pts_1d,
            mesh._map_samples_to_canonical_domain(mesh.mesh_pts))

    def _fast_interpolate(self, values, xx):
        # interpolate assuming need to evaluate all mesh points
        mesh = self._fwd_solver.residual.mesh
        assert xx.shape[1] == mesh.mesh_pts.shape[1]
        assert np.allclose(xx, mesh.mesh_pts)
        interp_vals = torch.linalg.multi_dot((self._mesh_basis_mat, values))
        # assert np.allclose(interp_vals, mesh.interpolate(values, xx))
        return interp_vals

    def _eval(self, sample, jac=False):
        self._fwd_solver.residual._diff_fun = partial(
            self._fast_interpolate,
            self._kle(torch.as_tensor(sample[:, None])))
        newton_kwargs = {"maxiters": 1, "tol": 1e-8}
        sol = self._fwd_solver.solve(**newton_kwargs)
        qoi = self._functional(sol, torch.as_tensor(sample)).numpy()
        if not jac:
            return qoi
        return qoi, self._adj_solver.compute_gradient(
            lambda r, p: None, torch.as_tensor(sample),
            **newton_kwargs).numpy().squeeze()

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(self._eval, samples)


def _setup_inverse_advection_diffusion_benchmark(
        nobs, noise_std, length_scale, sigma, nvars, orders, kle=None):
    variable = IndependentMarginalsVariable([stats.norm(0, 1)]*nvars)
    true_kle_params = variable.rvs(1)

    domain_bounds = [0, 1, 0, 1]
    mesh = CartesianProductCollocationMesh(domain_bounds, orders)
    bndry_conds = [
        [partial(full_fun_axis_1, 0, oned=False), "D"],
        [partial(full_fun_axis_1, 0, oned=False), "D"],
        [partial(full_fun_axis_1, 0, oned=False), "D"],
        [partial(full_fun_axis_1, 0, oned=False), "D"]]
    react_funs = None
    amp, scale = 100.0, 0.1
    loc = torch.tensor([0.25, 0.75])[:, None]
    forc_fun = partial(gauss_forc_fun, amp, scale, loc)
    vel_fun = partial(constant_vel_fun, [5, 0])

    if kle is None:
        kle = MeshKLE(mesh.mesh_pts, use_log=True, use_torch=True)
        kle.compute_basis(
            length_scale, sigma=sigma, nterms=nvars)

    print(np.unique(np.hstack(mesh._bndry_indices)))
    obs_indices = np.random.permutation(
        np.delete(
            np.arange(mesh.mesh_pts.shape[1]),
            np.unique(np.hstack(mesh._bndry_indices))))[:nobs]
    obs_functional = partial(mesh_locations_obs_functional, obs_indices)
    obs_model = AdvectionDiffusionReactionKLEModel(
        mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
        obs_functional)
    noise = np.random.normal(0, noise_std, (obs_indices.shape[0]))
    noiseless_obs = obs_model(true_kle_params)
    obs = noiseless_obs[0, :] + noise

    inv_functional = partial(
        negloglike_functional,  torch.as_tensor(obs), obs_indices,
        noise_std)
    dqdu = partial(negloglike_functional_dqdu, torch.as_tensor(obs),
                   obs_indices, noise_std)
    dqdp = partial(loglike_functional_dqdp,  torch.as_tensor(obs),
                   obs_indices, noise_std)
    inv_functional_deriv_funs = [dqdu, dqdp]
    inv_model = AdvectionDiffusionReactionKLEModel(
        mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
        inv_functional, inv_functional_deriv_funs)

    return inv_model, variable, true_kle_params, noiseless_obs, obs


def _setup_multi_index_advection_diffusion_benchmark(
        nobs, noise_std, length_scale, sigma, nvars, hf_orders):

    setup_model = partial(
        _setup_advection_diffusion_benchmark(
            nobs, noise_std, length_scale, sigma, nvars, kle=None))

    inv_model, variable, true_kle_params, noiseless_obs, obs = (
        _setup_advection_diffusion_benchmark(
            nobs, noise_std, length_scale, sigma, nvars, kle=None))

    def setup_model(orders):
        pass

    config_values = [2*np.arange(1, 11)+1, 2*np.arange(1, 11)+1]
    config_var_trans = ConfigureVariableTransformation(config_values)

    config_samples = cartesian_product(config_values)
    models = []
    for ii in reange(config_samples):
        assert np.all(config_samples[:, ii] <= hf_orders)
        models.append(setup_model(config_samples[:, ii]))
    ModelEnsemble(models)
