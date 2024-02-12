import torch
from functools import partial
from scipy import stats
import numpy as np

from pyapprox.pde.autopde.solvers import (
    SteadyStatePDE, SteadyStateAdjointPDE, TransientPDE, TransientFunction
)
from pyapprox.pde.autopde.physics import (
    AdvectionDiffusionReaction
)
from pyapprox.pde.autopde.mesh import (
    full_fun_axis_1, CartesianProductCollocationMesh,
    subdomain_integral_functional, final_time_functional
)
from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.variables.transforms import ConfigureVariableTransformation
from pyapprox.pde.karhunen_loeve_expansion import MeshKLE
from pyapprox.interface.wrappers import (
    evaluate_1darray_function_on_2d_array, MultiIndexModel, ModelEnsemble)


def constant_vel_fun(vels, xx):
    return torch.hstack([
        full_fun_axis_1(vels[ii], xx, oned=False) for ii in range(len(vels))])


def gauss_forc_fun(amp, scale, loc, xx):
    loc = torch.as_tensor(loc, dtype=torch.double)
    if loc.ndim == 1:
        loc = loc[:, None]
    vals = amp*torch.exp(
        -torch.sum((torch.as_tensor(xx, dtype=torch.double)-loc)**2/scale**2,
                   axis=0))[:, None]
    return vals


def beta_forc_fun(amp, scale, loc, xx):
    a1, b1 = 5, 5
    a2, b2 = 5, 5
    from scipy.special import beta
    amp /= beta(a1, b1)*beta(a2, b2)*10
    vals = torch.as_tensor((amp*xx[0]**(a1-1)*(1-xx[0])**(b1-1) *
                            xx[1]**(a2-1)*(1-xx[1])**(b2-1)),
                           dtype=torch.double)[:, None]
    return vals


def mesh_locations_obs_functional(obs_indices, sol, params):
    return sol[obs_indices]


def transient_multi_index_forcing(source1_args, xx, time=0,
                                  source2_args=None):
    vals = gauss_forc_fun(*source1_args, xx)
    # vals = beta_forc_fun(*source1_args, xx)
    if time == 0:
        return vals
    if source2_args is None:
        return vals*0
    vals -= gauss_forc_fun(*source2_args, xx)
    return vals


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


def raw_advection_diffusion_reaction_kle_dRdp(kle, residual, sol, param_vals):
    mesh = residual.mesh
    dmats = [residual.mesh._dmat(dd) for dd in range(mesh.nphys_vars)]
    if kle.use_log:
        # compute gradient of diffusivity with respect to KLE coeff
        assert param_vals.ndim == 1
        kle_vals = kle(param_vals[:, None])
        assert kle_vals.ndim == 2
        dkdp = kle_vals*kle.eig_vecs
    else:
        dkdp = kle.eig_vecs
    Du = [torch.linalg.multi_dot((dmats[dd], sol))
          for dd in range(mesh.nphys_vars)]
    kDu = [Du[dd][:, None]*dkdp for dd in range(mesh.nphys_vars)]
    dRdp = sum([torch.linalg.multi_dot((dmats[dd], kDu[dd]))
               for dd in range(mesh.nphys_vars)])
    return dRdp


def advection_diffusion_reaction_kle_dRdp(
        mesh, kle, bndry_conds, residual, sol, param_vals):
    dRdp = raw_advection_diffusion_reaction_kle_dRdp(
        kle, residual, sol, param_vals)
    for ii, bndry_cond in enumerate(bndry_conds):
        idx = mesh._bndry_indices[ii]
        if bndry_cond[1] == "D":
            dRdp[idx] = 0.0
        elif bndry_cond[1] == "R":
            mesh_pts_idx = mesh._bndry_slice(mesh.mesh_pts, idx, 1)
            normal_vals = mesh._bndrys[ii].normals(mesh_pts_idx)
            if kle.use_log:
                kle_vals = kle(param_vals[:, None])
                dkdp = kle_vals*kle.eig_vecs
            else:
                dkdp = torch.as_tensor(kle.eig_vecs)
            flux_vals = [
                (torch.linalg.multi_dot(
                    (mesh._bndry_slice(mesh._dmat(dd), idx, 0), sol))[:, None]
                 * mesh._bndry_slice(dkdp, idx, 0))
                for dd in range(mesh.nphys_vars)]
            flux_normal_vals = [
                normal_vals[:, dd:dd+1]*flux_vals[dd]
                for dd in range(mesh.nphys_vars)]
            dRdp[idx] = sum(flux_normal_vals)
        else:
            raise NotImplementedError()
    return dRdp


class AdvectionDiffusionReactionKLEModel():
    def __init__(self, mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
                 functional, functional_deriv_funs=[None, None],
                 newton_kwargs={}):

        self._newton_kwargs = newton_kwargs
        self._kle = kle
        # TODO pass in parameterized functions for diffusiviy and forcing and
        # reaction and use same process as used for KLE currently

        self._fwd_solver = self._set_forward_solver(
            mesh, bndry_conds, vel_fun, react_funs, forc_fun)

        import inspect
        if "mesh" == inspect.getfullargspec(functional).args[0]:
            if "physics" == inspect.getfullargspec(functional).args[1]:
                functional = partial(
                    functional, mesh, self._fwd_solver.physics)
            else:
                functional = partial(functional, mesh)
        for ii in range(len(functional_deriv_funs)):
            if (functional_deriv_funs[ii] is not None and
                "mesh" == inspect.getfullargspec(
                    functional_deriv_funs[ii]).args[0]):
                functional_deriv_funs[ii] = partial(
                    functional_deriv_funs[ii], mesh)

        self._functional = functional

        self._mesh_basis_mat = mesh._get_lagrange_basis_mat(
            mesh._canonical_mesh_pts_1d,
            mesh._map_samples_to_canonical_domain(mesh.mesh_pts))

        if issubclass(type(self._fwd_solver), SteadyStatePDE):
            dqdu, dqdp = functional_deriv_funs
            dRdp = partial(
                advection_diffusion_reaction_kle_dRdp,
                mesh, self._kle, self._fwd_solver.physics._bndry_conds)
            # dRdp must be after boundary conditions are applied.
            # For now assume that parameters do not effect boundary conditions
            # so dRdp at boundary indices is zero
            self._adj_solver = SteadyStateAdjointPDE(
                self._fwd_solver, self._functional, dqdu, dqdp, dRdp)

    def _set_forward_solver(self, mesh, bndry_conds, vel_fun, react_funs,
                            forc_fun):
        if react_funs is None:
            react_funs = [None, None]
        return SteadyStatePDE(AdvectionDiffusionReaction(
            mesh, bndry_conds, partial(full_fun_axis_1, 1), vel_fun,
            react_funs[0], forc_fun, react_funs[1]))

    def _fast_interpolate(self, values, xx):
        # interpolate assuming need to evaluate all mesh points
        mesh = self._fwd_solver.physics.mesh
        assert xx.shape[1] == mesh.mesh_pts.shape[1]
        assert np.allclose(xx, mesh.mesh_pts)
        interp_vals = torch.linalg.multi_dot((self._mesh_basis_mat, values))
        # assert np.allclose(interp_vals, mesh.interpolate(values, xx))
        return interp_vals

    def _set_random_sample(self, sample):
        self._fwd_solver.physics._diff_fun = partial(
            self._fast_interpolate,
            self._kle(sample[:, None]))

    def _eval(self, sample, return_grad=False):
        sample_copy = torch.as_tensor(sample.copy(), dtype=torch.double)
        self._set_random_sample(sample_copy)
        sol = self._fwd_solver.solve(**self._newton_kwargs)
        qoi = self._functional(sol, sample_copy).numpy()
        if not return_grad:
            return qoi
        grad = self._adj_solver.compute_gradient(
            lambda r, p: None, sample_copy, **self._newton_kwargs)
        return qoi, grad.detach().numpy().squeeze()

    def __call__(self, samples, return_grad=False):
        return evaluate_1darray_function_on_2d_array(
            self._eval, samples, return_grad=return_grad)

    def get_num_degrees_of_freedom_cost(self, config_vals):
        if (len(config_vals) !=
                self._fwd_solver.physics.mesh.mesh_pts.shape[0]):
            msg = "config_vals provided has an incorrect shape"
            raise ValueError(msg)
        orders = config_vals
        return int(np.prod(np.asarray(orders)+1))**3

    def __repr__(self):
        return "{0}(kle={1}, mesh={2})".format(
            self.__class__.__name__, self._kle,
            self._fwd_solver.physics.mesh)


class TransientAdvectionDiffusionReactionKLEModel(
        AdvectionDiffusionReactionKLEModel):
    def __init__(self, mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
                 functional, init_sol_fun, init_time, final_time,
                 deltat, butcher_tableau, newton_kwargs={}):
        if callable(init_sol_fun):
            self._init_sol = torch.as_tensor(
                init_sol_fun(mesh.mesh_pts), dtype=torch.double)
            if self._init_sol.ndim == 2:
                self._init_sol = self._init_sol[:, 0]
        else:
            assert init_sol_fun is None
            self._init_sol = None
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        self._butcher_tableau = butcher_tableau
        super().__init__(mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
                         functional, newton_kwargs=newton_kwargs)
        self._steady_state_fwd_solver = super()._set_forward_solver(
            mesh, bndry_conds, vel_fun, react_funs, forc_fun)

    def _get_init_sol(self, sample):
        if self._init_sol is None:
            self._steady_state_fwd_solver.physics._diff_fun = partial(
                self._fast_interpolate,
                self._kle(sample[:, None]))
            self._fwd_solver.physics._set_time(self._init_time)
            init_sol = self._steady_state_fwd_solver.solve(
                **self._newton_kwargs)
        else:
            init_sol = self._init_sol
        return init_sol

    def _eval(self, sample, return_grad=False):
        if return_grad:
            raise ValueError("return_grad=True is not supported")
        sample_copy = torch.as_tensor(sample.copy(), dtype=torch.double)
        self._set_random_sample(sample_copy)
        init_sol = self._get_init_sol(sample_copy)
        sols, times = self._fwd_solver.solve(
            init_sol, 0, self._final_time,
            newton_kwargs=self._newton_kwargs, verbosity=0)
        qoi = self._functional(sols, sample_copy).numpy()
        return qoi

    def _set_forward_solver(self, mesh, bndry_conds, vel_fun, react_funs,
                            forc_fun):
        if react_funs is None:
            react_funs = [None, None]
        return TransientPDE(
            AdvectionDiffusionReaction(
                mesh, bndry_conds, partial(full_fun_axis_1, 1), vel_fun,
                react_funs[0], forc_fun, react_funs[1]),
            self._deltat, self._butcher_tableau)

    def get_num_degrees_of_freedom_cost(self, config_vals):
        if (len(config_vals) !=
                self._fwd_solver.physics.mesh.mesh_pts.shape[0]+1):
            msg = "config_vals provided has an incorrect shape"
            raise ValueError(msg)
        ntsteps = int(self._final_time/config_vals[-1])
        # valid for implicit backward euler and crank nicolson
        assert (self._butcher_tableau == "im_beuler1" or
                self._butcher_tableau == "im_crank2")
        return super().get_num_degrees_of_freedom_cost(
            config_vals[:-1])*ntsteps

    def __repr__(self):
        return "{0}(kle={1}, T0={2}, T={3}, dt={4}, tableau={5}, mesh={6})".format(
            self.__class__.__name__, self._kle, self._init_time,
            self._final_time, self._deltat, self._butcher_tableau,
            self._fwd_solver.physics.mesh)


def _setup_advection_diffusion_benchmark(
        amp, scale, loc, length_scale, sigma, nvars, orders, functional,
        functional_deriv_funs=[None, None], kle_args=None,
        newton_kwargs={}, time_scenario=None, vel_vec=[1., 0.],
        kle_mean_field=0.):
    variable = IndependentMarginalsVariable([stats.norm(0, 1)]*nvars)
    orders = np.asarray(orders, dtype=int)

    domain_bounds = [0, 1, 0, 1]
    mesh = CartesianProductCollocationMesh(domain_bounds, orders)
    # bndry_conds = [
    #     [partial(full_fun_axis_1, 0, oned=False), "D"],
    #     [partial(full_fun_axis_1, 0, oned=False), "D"],
    #     [partial(full_fun_axis_1, 0, oned=False), "D"],
    #     [partial(full_fun_axis_1, 0, oned=False), "D"]]
    alpha, nominal_value = 0.1, 0
    bndry_conds = [
        [lambda x: torch.full(
            (x.shape[1], 1), alpha*nominal_value), "R", alpha],
        [lambda x: torch.full(
            (x.shape[1], 1), alpha*nominal_value), "R", alpha],
        [lambda x: torch.full(
            (x.shape[1], 1), alpha*nominal_value), "R", alpha],
        [lambda x: torch.full(
            (x.shape[1], 1), alpha*nominal_value), "R", alpha]]
    react_funs = None
    vel_fun = partial(constant_vel_fun, vel_vec)

    if kle_args is None:
        kle = MeshKLE(
            mesh.mesh_pts, use_log=True, use_torch=True,
            mean_field=kle_mean_field)
        kle.compute_basis(
            length_scale, sigma=sigma, nterms=nvars)
    else:
        kle = InterpolatedMeshKLE(kle_args[0], kle_args[1], mesh)

    if time_scenario is None:
        forc_fun = partial(gauss_forc_fun, amp, scale, loc)
        model = AdvectionDiffusionReactionKLEModel(
            mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
            functional, functional_deriv_funs, newton_kwargs)
    else:
        assert (time_scenario["init_sol_fun"] is None or
                callable(time_scenario["init_sol_fun"]))
        init_sol_fun, final_time, deltat, butcher_tableau = (
            time_scenario["init_sol_fun"], time_scenario["final_time"],
            time_scenario["deltat"], time_scenario["butcher_tableau"])
        forc_fun = partial(
            transient_multi_index_forcing,
            [amp, scale, loc], source2_args=time_scenario["sink"])
        forc_fun = TransientFunction(forc_fun)
        model = TransientAdvectionDiffusionReactionKLEModel(
            mesh, bndry_conds, kle, vel_fun, react_funs, forc_fun,
            functional, init_sol_fun, 0, final_time, deltat, butcher_tableau,
            newton_kwargs)

    return model, variable


class InterpolatedMeshKLE(MeshKLE):
    def __init__(self, kle_mesh, kle, mesh):
        self._kle_mesh = kle_mesh
        self._kle = kle
        self._mesh = mesh

        self.matern_nu = self._kle.matern_nu
        self.nterms = self._kle.nterms
        self.lenscale = self._kle.lenscale

        self._basis_mat = self._kle_mesh._get_lagrange_basis_mat(
            self._kle_mesh._canonical_mesh_pts_1d,
            mesh._map_samples_to_canonical_domain(self._mesh.mesh_pts))

    def _fast_interpolate(self, values, xx):
        assert xx.shape[1] == self._mesh.mesh_pts.shape[1]
        assert np.allclose(xx, self._mesh.mesh_pts)
        interp_vals = torch.linalg.multi_dot((self._basis_mat, values))
        # assert np.allclose(
        #     interp_vals, self._kle_mesh.interpolate(values, xx))
        return interp_vals

    def __call__(self, coef):
        use_log = self._kle.use_log
        self._kle.use_log = False
        vals = self._kle(coef)
        interp_vals = self._fast_interpolate(vals, self._mesh.mesh_pts)
        mean_field = self._fast_interpolate(
            self._kle.mean_field[:, None], self._mesh.mesh_pts)
        if use_log:
            interp_vals = np.exp(mean_field+interp_vals)
        self._kle.use_log = use_log
        return interp_vals


def _setup_inverse_advection_diffusion_benchmark(
        amp, scale, loc, nobs, noise_std, length_scale, sigma, nvars, orders,
        obs_indices=None):

    loc = torch.as_tensor(loc, dtype=torch.double)
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
        negloglike_functional,
        torch.as_tensor(obs, dtype=torch.double), obs_indices,
        noise_std)
    dqdu = partial(negloglike_functional_dqdu,
                   torch.as_tensor(obs, dtype=torch.double),
                   obs_indices, noise_std)
    dqdp = partial(loglike_functional_dqdp,
                   torch.as_tensor(obs, dtype=torch.double),
                   obs_indices, noise_std)
    inv_functional_deriv_funs = [dqdu, dqdp]

    newton_kwargs = {"maxiters": 1, "rtol": 1e-7, "verbosity": 0}
    inv_model, variable = _setup_advection_diffusion_benchmark(
        amp, scale, loc, length_scale, sigma, nvars, orders,
        inv_functional, inv_functional_deriv_funs, newton_kwargs=newton_kwargs)

    return (inv_model, variable, true_kle_params, noiseless_obs, obs,
            obs_indices, obs_model, obs_model._kle,
            obs_model._fwd_solver.physics.mesh)


def _setup_multi_index_advection_diffusion_benchmark(
        length_scale, sigma, nvars, time_scenario=None,
        functional=None, config_values=None,
        source_loc=[0.25, 0.75], source_scale=0.1,
        source_amp=100.0, vel_vec=[1., 0.], kle_mean_field=0.):

    if time_scenario is True:
        time_scenario = {
            "final_time": 0.2,
            "butcher_tableau": "im_crank2",
            "deltat": 0.1,  # default will be overwritten
            "init_sol_fun": None,
            # "sink": [50, 0.1, [0.75, 0.75]]
            "sink": None
        }

    source_loc = np.asarray(source_loc)
    if source_loc.ndim == 1:
        source_loc = source_loc[:, None]
    assert source_loc.shape[1] == 1

    newton_kwargs = {"maxiters": 1, "rtol": 0}
    if config_values is None:
        nlevels = 2
        config_values = [
            2+4*np.arange(1, 1+nlevels)+1,
            2+4*np.arange(1, 1+nlevels)+1]
        if time_scenario is not None:
            config_values += [
                time_scenario["final_time"]/(4**np.arange(2, 2+nlevels))]

    if functional is None:
        subdomain_bounds = np.array([0.75, 1, 0, 0.25])
        functional = partial(subdomain_integral_functional, subdomain_bounds)
        if time_scenario is not None:
            functional = partial(final_time_functional, functional)
    hf_orders = np.array([config_values[0][-1], config_values[1][-1]])
    if time_scenario is None and len(config_values) != 2:
        msg = "Steady state scenario specified so must provide config_values "
        msg += "for each physical dimension"
        raise ValueError(msg)
    if time_scenario is not None and len(config_values) != 3:
        msg = "Transient scenario specified so must provide config_values "
        msg += "for each physical dimension and time-stepping"
        raise ValueError(msg)
    if time_scenario is not None:
        time_scenario["deltat"] = config_values[2][-1]

    hf_model, variable = _setup_advection_diffusion_benchmark(
        source_amp, source_scale, source_loc,
        length_scale, sigma, nvars, hf_orders, functional,
        newton_kwargs=newton_kwargs, time_scenario=time_scenario,
        vel_vec=vel_vec, kle_mean_field=kle_mean_field)
    kle_args = [hf_model._fwd_solver.physics.mesh, hf_model._kle]

    def setup_model(config_vals):
        orders = config_vals[:2]
        if time_scenario is not None:
            time_scenario["deltat"] = config_vals[2]
        return _setup_advection_diffusion_benchmark(
            source_amp, source_scale, source_loc, length_scale,
            sigma, nvars, orders, functional,
            kle_args=kle_args, newton_kwargs=newton_kwargs,
            time_scenario=time_scenario, vel_vec=vel_vec,
            kle_mean_field=kle_mean_field)[0]
    multi_index_model = MultiIndexModel(setup_model, config_values)
    config_var_trans = ConfigureVariableTransformation(config_values)
    # order models from highest to lowest fidelity to match ACV MC convention
    model_ensemble = ModelEnsemble(
        multi_index_model._model_ensemble.functions[::-1])
    return (multi_index_model, variable, config_var_trans,
            model_ensemble)
