import torch
from functools import partial
from scipy import stats
import numpy as np

from pyapprox.util.utilities import cartesian_product
from pyapprox.pde.autopde.solvers import (
    SteadyStatePDE, SteadyStateAdjointPDE
)
from pyapprox.pde.autopde.physics import (
    AdvectionDiffusionReaction
)
from pyapprox.pde.autopde.mesh import (
    full_fun_axis_1, CartesianProductCollocationMesh,
    subdomain_integral_functional
)
from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.pde.karhunen_loeve_expansion import MeshKLE
from pyapprox.interface.wrappers import (
    evaluate_1darray_function_on_2d_array, MultiIndexModel)


def constant_vel_fun(vels, xx):
    return torch.hstack([
        full_fun_axis_1(vels[ii], xx, oned=False) for ii in range(len(vels))])


def gauss_forc_fun(amp, scale, loc, xx):
    return amp*torch.exp(
        -torch.sum((torch.as_tensor(xx)-loc)**2/scale**2, axis=0))[:, None]


def mesh_locations_obs_functional(obs_indices, sol, params):
    return sol[obs_indices]


def negloglike_functional(obs, obs_indices, noise_std, sol, params,
                          ignore_constants=False):
    assert obs.ndim == 1 and sol.ndim == 1
    nobs = obs_indices.shape[0]
    if not ignore_constants:
        tmp = 1/(2*noise_std**2)
        # ll = 0.5*np.log(tmp/np.pi)*nobs
        ll = -nobs/2*np.log(2*noise_std**2*np.pi)
    else:
        ll, tmp = 0, 1
    pred_obs = sol[obs_indices]
    ll += -torch.sum((obs-pred_obs)**2*tmp)
    return -ll


def negloglike_functional_dqdu(obs, obs_indices, noise_std, sol, params,
                               ignore_constants=False):
    if not ignore_constants:
        tmp = 1/(2*noise_std**2)
    else:
        tmp = 1
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
                 functional, functional_deriv_funs=[None, None],
                 newton_kwargs={}):

        import inspect
        if "mesh" == inspect.getfullargspec(functional).args[0]:
            functional = partial(functional, mesh)
        for ii in range(len(functional_deriv_funs)):
            if (functional_deriv_funs[ii] is not None and
                "mesh" == inspect.getfullargspec(
                    functional_deriv_funs[ii]).args[0]):
                functional_deriv_funs[ii] = partial(
                    functional_deriv_funs[ii], mesh)

        self._newton_kwargs = newton_kwargs
        self._kle = kle
        # TODO pass in parameterized functions for diffusiviy and forcing and
        # reaction and use same process as used for KLE currently

        if react_funs is None:
            react_funs = [self._default_react_fun, self._default_react_fun_jac]

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

    def _default_react_fun(self, sol):
        return 0*sol

    def _default_react_fun_jac(self, sol):
        return torch.zeros((sol.shape[0], sol.shape[0]))

    def _fast_interpolate(self, values, xx):
        # interpolate assuming need to evaluate all mesh points
        mesh = self._fwd_solver.residual.mesh
        assert xx.shape[1] == mesh.mesh_pts.shape[1]
        assert np.allclose(xx, mesh.mesh_pts)
        interp_vals = torch.linalg.multi_dot((self._mesh_basis_mat, values))
        # assert np.allclose(interp_vals, mesh.interpolate(values, xx))
        return interp_vals

    def _eval(self, sample, jac=False):
        sample_copy = torch.as_tensor(sample.copy())
        self._fwd_solver.residual._diff_fun = partial(
            self._fast_interpolate,
            self._kle(sample_copy[:, None]))
        sol = self._fwd_solver.solve(**self._newton_kwargs)
        qoi = self._functional(sol, sample_copy).numpy()
        if not jac:
            return qoi
        grad = self._adj_solver.compute_gradient(
            lambda r, p: None, sample_copy, **self._newton_kwargs)
        return qoi, grad.detach().numpy().squeeze()

    def __call__(self, samples, jac=False):
        return evaluate_1darray_function_on_2d_array(
            self._eval, samples, jac=jac)


def _setup_advection_diffusion_benchmark(
        amp, scale, loc, length_scale, sigma, nvars, orders, functional,
        functional_deriv_funs=[None, None], kle_args=None,
        newton_kwargs={}):
    variable = IndependentMarginalsVariable([stats.norm(0, 1)]*nvars)
    orders = np.asarray(orders, dtype=int)

    domain_bounds = [0, 1, 0, 1]
    mesh = CartesianProductCollocationMesh(domain_bounds, orders)
    bndry_conds = [
        [partial(full_fun_axis_1, 0, oned=False), "D"],
        [partial(full_fun_axis_1, 0, oned=False), "D"],
        [partial(full_fun_axis_1, 0, oned=False), "D"],
        [partial(full_fun_axis_1, 0, oned=False), "D"]]
    react_funs = None
    forc_fun = partial(gauss_forc_fun, amp, scale, loc)
    vel_fun = partial(constant_vel_fun, [5, 0])

    if kle_args is None:
        kle = MeshKLE(mesh.mesh_pts, use_log=True, use_torch=True)
        kle.compute_basis(
            length_scale, sigma=sigma, nterms=nvars)
    else:
        kle = InterpolatedMeshKLE(kle_args[0], kle_args[1], mesh)

    model = AdvectionDiffusionReactionKLEModel(
        mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
        functional, functional_deriv_funs, newton_kwargs)

    return model, variable


class InterpolatedMeshKLE(MeshKLE):
    def __init__(self, kle_mesh, kle, mesh):
        self._kle_mesh = kle_mesh
        self._kle = kle
        self._mesh = mesh

        self._basis_mat = self._kle_mesh._get_lagrange_basis_mat(
            self._kle_mesh._canonical_mesh_pts_1d,
            mesh._map_samples_to_canonical_domain(self._mesh.mesh_pts))

    def _fast_interpolate(self, values, xx):
        assert xx.shape[1] == self._mesh.mesh_pts.shape[1]
        assert np.allclose(xx, self._mesh.mesh_pts)
        interp_vals = torch.linalg.multi_dot((self._basis_mat, values))
        # assert np.allclose(interp_vals, self._kle_mesh.interpolate(values, xx))
        return interp_vals

    def __call__(self, coef):
        use_log = self._kle.use_log
        self._kle.use_log = False
        vals = self._kle(coef)
        interp_vals = self._fast_interpolate(vals, self._mesh.mesh_pts)
        if use_log:
            interp_vals = np.exp(interp_vals)
        self._kle.use_log = use_log
        return interp_vals


def _setup_inverse_advection_diffusion_benchmark(
        amp, scale, loc, nobs, noise_std, length_scale, sigma, nvars, orders,
        obs_indices=None):

    loc = torch.as_tensor(loc)
    if loc.ndim == 1:
        loc = loc[:, None]

    ndof = np.prod(np.asarray(orders)+1)
    if obs_indices is None:
        bndry_indices = np.hstack(
            [np.arange(0, orders[0]+1),
             np.arange(ndof-orders[0]-1, ndof)] +
            [jj*(orders[0]+1) for jj in range(1, orders[1])] +
            [jj*(orders[0]+1)+orders[0] for jj in range(1, orders[1])])
        obs_indices = np.random.permutation(
            np.delete(np.arange(ndof), bndry_indices))[:nobs]
    obs_functional = partial(mesh_locations_obs_functional, obs_indices)
    obs_model, variable = _setup_advection_diffusion_benchmark(
        amp, scale, loc, length_scale, sigma, nvars, orders, obs_functional)

    true_kle_params = variable.rvs(1)
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

    newton_kwargs = {"maxiters": 1, "rel_error": True, "verbosity": 0}
    inv_model, variable = _setup_advection_diffusion_benchmark(
        amp, scale, loc, length_scale, sigma, nvars, orders,
        inv_functional, inv_functional_deriv_funs, newton_kwargs=newton_kwargs)

    return (inv_model, variable, true_kle_params, noiseless_obs, obs,
            obs_indices, obs_model)


def _setup_multi_index_advection_diffusion_benchmark(
        length_scale, sigma, nvars, config_values=None):

    amp, scale = 100.0, 0.1
    loc = torch.tensor([0.25, 0.75])[:, None]

    newton_kwargs = {"maxiters": 1, "rel_error": False}
    if config_values is None:
        config_values = [2*np.arange(1, 11)+1, 2*np.arange(1, 11)+1]

    subdomain_bounds = np.array([0.75, 1, 0, 0.25])
    functional = partial(subdomain_integral_functional, subdomain_bounds)
    hf_orders = np.array([config_values[0][-1], config_values[1][-1]])
    hf_model, variable = _setup_advection_diffusion_benchmark(
        amp, scale, loc, length_scale, sigma, nvars, hf_orders, functional,
        newton_kwargs=newton_kwargs)
    kle_args = [hf_model._fwd_solver.residual.mesh, hf_model._kle]

    def setup_model(orders):
        return _setup_advection_diffusion_benchmark(
            amp, scale, loc, length_scale, sigma, nvars, orders, functional,
            kle_args=kle_args, newton_kwargs=newton_kwargs)[0]
    model = MultiIndexModel(setup_model, config_values)
    return model, variable
