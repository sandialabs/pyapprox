import numpy as np
from scipy.linalg import qr as qr_factorization
from abc import ABC, abstractmethod

from pyapprox.util.linalg import qr_solve
from pyapprox.interface.wrappers import (
    evaluate_1darray_function_on_2d_array
)
from pyapprox.util.sys_utilities import get_num_args
from pyapprox.pde.spectralcollocation.spectral_collocation import (
    OneDCollocationMesh, RectangularCollocationMesh
)


def zeros_fun_axis_1(x):
    # axis_1 used when x is mesh points
    return np.zeros((x.shape[1], 1))


def ones_fun_axis_0(x):
    # axis_1 used when x is solution like quantity
    return np.ones((x.shape[0], 1))


class AbstractAdvectionDiffusion(ABC):
    r"""
    Solve the advection diffusion equation
    .. math:: u_t(x)-(a(x)*u_x(x))_x+b(x)*u_x(x) = f
    """
    def __init__(self):
        self.diffusivity_fun = None
        self.forcing_fun = None
        self.velocity_fun = None
        self.domain = None
        self.qoi_functional = None
        self.mesh = None

    def set_qoi_functional(self, qoi_functional, qoi_functional_deriv=None):
        self.qoi_functional = qoi_functional
        self.qoi_functional_deriv = qoi_functional_deriv

    def set_domain(self, domain, order):
        if len(domain) == 2:
            self.mesh = OneDCollocationMesh(domain, order)
        elif len(domain) == 4:
            self.mesh = RectangularCollocationMesh(domain, order)
        else:
            raise ValueError("Only 1D and 2D domains supported")

    def initialize(self, bndry_conds, diffusivity_fun, forcing_fun,
                   velocity_fun, order, domain):
        self.set_domain(domain, order)
        self.mesh.set_boundary_conditions(bndry_conds)
        self.set_diffusivity_function(diffusivity_fun)
        self.set_forcing_function(forcing_fun)
        self.set_velocity_function(velocity_fun)
        # default qoi functional is integral of solution over entire domain
        # at the final time-step or for steady-state problems the only solution
        self.set_qoi_functional(self.mesh.integrate, ones_fun_axis_0)

    def set_diffusivity_function(self, fun):
        assert callable(fun)
        assert get_num_args(fun) == 2
        self.diffusivity_fun = fun

    def set_velocity_function(self, fun):
        assert callable(fun)
        assert get_num_args(fun) == 2
        self.velocity_fun = fun

    def set_forcing_function(self, fun):
        assert callable(fun)
        assert get_num_args(fun) == 2
        self.forcing_fun = fun

    def form_collocation_matrix(self, diffusivity_vals, advection_vals):
        assert diffusivity_vals.ndim == 2 and diffusivity_vals.shape[1] == 1
        assert (advection_vals.ndim == 2 and
                advection_vals.shape[1] == self.mesh.nphys_vars)

        matrix = 0
        for dd in range(self.mesh.nphys_vars):
            matrix += np.dot(
                self.mesh.derivative_matrices[dd],
                diffusivity_vals*self.mesh.derivative_matrices[dd])
            matrix -= self.mesh.derivative_matrices[dd]*advection_vals[:, dd]
        # want to solve -u_xx-u_yy+au_x+bu_y=f so add negative
        return -matrix

    def apply_boundary_conditions_to_matrix(self, matrix):
        return self.mesh._apply_boundary_conditions_to_matrix(
             matrix)

    def apply_boundary_conditions_to_rhs(self, rhs):
        return self.mesh._apply_boundary_conditions_to_rhs(
            rhs, self.mesh.bndry_conds)

    def apply_boundary_conditions(self, matrix, forcing):
        matrix = self.apply_boundary_conditions_to_matrix(matrix)
        forcing = self.apply_boundary_conditions_to_rhs(forcing)
        return matrix, forcing

    @abstractmethod
    def solve(self, sample):
        raise NotImplementedError()

    def get_collocation_points(self):
        return self.mesh.mesh_pts

    def get_derivative_matrices(self):
        return self.mesh.derivative_matrices

    def value(self, sample):
        assert sample.ndim == 1
        solutions = self.solve(sample)
        # from pyapprox.util.configure_plots import plt
        # fig, axs = plt.subplots(1, 2)
        # self.mesh.plot(self.initial_sol, ax=axs[0])
        # self.mesh.plot(solutions[:, -1:], ax=axs[1])
        # print(self.initial_sol.shape, solutions[:, -1:].shape)
        # print(np.linalg.norm(self.initial_sol-solutions[:, -1:]))
        # plt.show()
        qoi = self.qoi_functional(solutions)
        if np.isscalar(qoi) or qoi.ndim == 0:
            qoi = np.array([qoi])
        return qoi

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.value, samples, None)

    @abstractmethod
    def get_num_degrees_of_freedom(self, config_sample):
        raise NotImplementedError()

    def compute_bndry_fluxes(self, sol, bndry_indices, sample):
        normal_fluxes = []
        normals = self.mesh._get_bndry_normals(bndry_indices)
        for ii, idx in enumerate(bndry_indices):
            normal_flux = 0
            diff_vals = self.diffusivity_fun(
                self.mesh.mesh_pts[:, self.mesh.boundary_indices[idx]], sample)
            for dd in range(self.mesh.nphys_vars):
                deriv_dd = self.mesh.derivative_matrices[dd][
                    self.mesh.boundary_indices[idx], :]
                flux_1d = diff_vals*(deriv_dd.dot(sol))
                normal_flux += flux_1d*normals[ii, dd]
            normal_fluxes.append(normal_flux[:, 0])
        return normal_fluxes


class SteadyStateAdvectionDiffusion(AbstractAdvectionDiffusion):
    def __init__(self):
        super().__init__()
        self.adjoint_derivative_matrix = None
        self.adjoint_mesh_pts = None

    def solve(self, sample):
        assert sample.ndim == 1
        diffusivity = self.diffusivity_fun(self.mesh.mesh_pts, sample[:, None])
        forcing = self.forcing_fun(self.mesh.mesh_pts, sample[:, None])
        advection = self.velocity_fun(self.mesh.mesh_pts, sample[:, None])

        assert diffusivity.ndim == 2 and diffusivity.shape[1] == 1
        assert forcing.ndim == 2 and forcing.shape[1] == 1
        assert advection.ndim == 2 and (
            advection.shape[1] == self.mesh.mesh_pts.shape[0])
        # forcing will be overwritten with bounary values so must take a
        # deep copy
        forcing = forcing.copy()
        # we need another copy so that forcing can be used when solving adjoint
        self.forcing_vals = forcing.copy()
        assert not np.any(diffusivity <= 0.)
        self.collocation_matrix = self.form_collocation_matrix(
            diffusivity, advection)
        matrix, forcing = self.apply_boundary_conditions(
            self.collocation_matrix.copy(), forcing)
        solution = np.linalg.solve(matrix, forcing)
        # store solution for use with adjoints
        self.fwd_solution = solution.copy()
        return solution

    def solve_adjoint(self, sample, order):
        """
        Typically with FEM we solve Ax=b and the discrete adjoint equation
        is A'y=z. But with collocation this does not work. Instead of
        taking the adjoint of the discrete system as the aforementioned
        approach does. We discretize the continuous adjoint equation. Which for
        the ellipic diffusion equation is just Ay=z. That is the adjoint
        of A is A.
        """
        if np.allclose(order, self.mesh.order):
            # used when computing gradient from adjoint solution
            matrix = self.collocation_matrix.copy()
        else:
            # used when computing error estimate from adjoint solution
            if self.adjoint_derivative_matrix is None:
                adjoint_mesh_pts, self.adjoint_derivative_matrix = \
                    chebyshev_derivative_matrix(order)
                self.adjoint_mesh_pts = self.map_samples_from_canonical_domain(
                    adjoint_mesh_pts)
                # scale derivative matrix from [-1,1] to [a,b]
                self.adjoint_derivative_matrix *= 2. / \
                    (self.domain[1]-self.domain[0])
                # TODO: THIS will not yet work for 2D

            diffusivity = self.diffusivity_fun(self.adjoint_mesh_pts, sample)
            advection = self.velocity_fun(self.adjoint_mesh_pts, sample)
            matrix = self._form_collocation_matrix(
                self.adjoint_derivative_matrix, diffusivity, advection)
            self.adjoint_collocation_matrix = matrix.copy()

        # regardless of whether computing error estimate or
        # computing gradient, rhs is always derivative (with respect to the
        # solution) of the qoi_functional
        qoi_deriv = self.qoi_functional_deriv(self.fwd_solution)
        print(qoi_deriv, 'd')
        print(self.fwd_solution, 'f')

        matrix = self.mesh._apply_dirichlet_boundary_conditions_to_matrix(
            matrix, self.mesh.adjoint_dirichlet_bndry_indices)
        qoi_deriv = self.mesh._apply_boundary_conditions_to_rhs(
            qoi_deriv, self.mesh.adjoint_bndry_conds)
        adj_solution = np.linalg.solve(matrix, qoi_deriv)
        return adj_solution

    def compute_residual(self, matrix, solution, forcing):
        matrix, forcing = self.apply_boundary_conditions(matrix, forcing)
        return forcing - np.dot(matrix, solution)

    def compute_residual_derivative(self, solution, diffusivity_deriv,
                                    forcing_deriv, advection_deriv):
        matrix = self.form_collocation_matrix(
            diffusivity_deriv, advection_deriv)
        matrix = self.mesh._apply_dirichlet_boundary_conditions_to_matrix(
            matrix, self.mesh.adjoint_dirichlet_bndry_indices)
        # the values here are the derivative of the boundary conditions
        # with respect to the random parameters. I assume that
        # this is always zero
        forcing_deriv = self.mesh._apply_boundary_conditions_to_rhs(
            forcing_deriv, self.mesh.adjoint_bndry_conds)
        return forcing_deriv.squeeze() - np.dot(matrix, solution)

    def compute_error_estimate(self, sample):
        raise NotImplementedError("Not passing tests")
        # must solve adjoint with a higher order grid
        adj_solution = self.solve_adjoint(sample, self.order*2)

        # interpolate forward solution onto higher-order grid
        interp_fwd_solution = self.interpolate(
            self.fwd_solution, self.adjoint_mesh_pts)

        # compute residual of forward solution using higher-order grid
        forcing_vals = self.forcing_fun(self.adjoint_mesh_pts, sample)

        # compute residual
        residual = self.compute_residual(self.adjoint_collocation_matrix,
                                         interp_fwd_solution, forcing_vals)

        # self.plot(interp_fwd_solution+adj_solution,
        #           plot_mesh_coords=self.adjoint_mesh_pts )
        # self.plot(residual, plot_mesh_coords=self.adjoint_mesh_pts,
        #           color='r')
        # pylab.show()
        # print self.integrate((adj_solution+interp_fwd_solution )**2)

        # print(np.dot(residual, adj_solution )/self.integrate(
        #    residual * adj_solution)
        # print('cond', np.linalg.cond(self.adjoint_collocation_matrix))
        error_estimate = self.integrate(residual * adj_solution, self.order*2)

        return error_estimate

    def evaluate_gradient(self, sample):
        assert sample.ndim == 1
        num_stoch_dims = sample.shape[0]
        # qoi_deriv = self.qoi_functional_deriv(self.mesh.mesh_pts)
        adj_solution = self.solve_adjoint(sample, self.mesh.order)
        gradient = np.empty((num_stoch_dims), float)
        for ii in range(num_stoch_dims):
            diffusivity_deriv_vals_i = self.diffusivity_derivs_fun(
                self.mesh.mesh_pts, sample, ii)
            forcing_deriv_vals_i = self.forcing_derivs_fun(
                self.mesh.mesh_pts, sample, ii)
            advection_deriv_vals_i = self.advection_derivs_fun(
                self.mesh.mesh_pts, sample, ii)
            residual_deriv = self.compute_residual_derivative(
                self.fwd_solution, diffusivity_deriv_vals_i,
                forcing_deriv_vals_i, advection_deriv_vals_i)
            gradient[ii] = self.mesh.integrate(residual_deriv * adj_solution)
        return gradient

    def get_num_degrees_of_freedom(self, config_sample):
        return np.prod(config_sample)


class TransientAdvectionDiffusion(SteadyStateAdvectionDiffusion):
    def __init__(self):
        super().__init__()
        self.num_time_steps = None
        self.time_step_size = None
        self.final_time = None
        self.times = None
        self.initial_sol = None
        self.time_step_method = None
        self.implicit_matrix_factors = None
        # currently do not support time dependent boundary conditions,
        # diffusivity or advection

    def initialize(self, bndry_conds, diffusivity_fun, forcing_fun,
                   velocity_fun, order, domain, final_time, time_step_size,
                   initial_sol_fun, time_step_method="crank-nicholson"):
        super().initialize(bndry_conds, diffusivity_fun, forcing_fun,
                           velocity_fun, order, domain)
        self.set_time_step_size(final_time, time_step_size)
        self.set_time_step_method(time_step_method)
        if get_num_args(initial_sol_fun) != 1:
            msg = "initial_sol_fun must be a callable function with 1 argument"
            raise ValueError(msg)
        self.initial_sol_fun = initial_sol_fun
        self.initial_sol = self.initial_sol_fun(self.mesh.mesh_pts)

    def set_forcing_function(self, fun):
        """
        Set time dependent forcing
        """
        assert callable(fun)
        assert get_num_args(fun) == 3
        self.forcing_fun = fun

    def set_time_step_size(self, final_time, time_step_size):
        self.final_time = final_time
        self.time_step_size = time_step_size

    def set_time_step_method(self, time_step_method):
        self.time_step_method = time_step_method

    def explicit_runge_kutta(self, rhs, sol, time, time_step_size):
        assert callable(rhs)
        dt2 = time_step_size/2.
        k1 = rhs(time, sol)
        k2 = rhs(time+dt2, sol+dt2*k1)
        k3 = rhs(time+dt2, sol+dt2*k2)
        k4 = rhs(time+time_step_size, sol+time_step_size*k3)
        new_sol = sol+time_step_size/6.*(k1+2.*k2+2.*k3+k4)
        self.apply_boundary_conditions_to_rhs(new_sol)
        return new_sol

    def forward_euler(self, rhs, sol, time, time_step_size):
        k1 = rhs(time, sol)
        new_sol = sol+time_step_size*k1
        self.apply_boundary_conditions_to_rhs(new_sol)
        return new_sol

    def form_adams_moulton_3rd_order_system(self, matrix, current_sol,
                                            current_forcing, future_forcing,
                                            prev_forcing, prev_sol,
                                            time_step_size):
        """ 3rd order Adams-Moultobn method
        WARNING: seems to be unstable (at least my implementation)
        y_{n+2} = y_{n+1}+h(c_0y_{n+2}+c_1y_{n+1}+c_3y_{n})
        c = (5/12,2/3,-1./12)
        """

        dt12 = time_step_size/12.
        dt12matrix = dt12*matrix
        identity = np.eye(matrix.shape[0])
        matrix = identity-5.*dt12matrix
        rhs = np.dot(identity+8.*dt12matrix, current_sol)
        rhs += dt12*(5.*future_forcing+8.*current_forcing-prev_forcing)
        rhs -= np.dot(dt12matrix, prev_sol)
        # currently I do not support time varying boundary conditions
        return self.apply_boundary_conditions_to_rhs(rhs)

    def get_implicit_time_step_rhs(self, current_sol, time, sample):
        assert current_sol.ndim == 2 and current_sol.shape[1] == 1
        future_forcing = self.forcing_fun(
            self.mesh.mesh_pts, sample, time+self.time_step_size)
        if (self.time_step_method == "backward-euler"):
            rhs = current_sol + self.time_step_size*future_forcing
        elif (self.time_step_method == "crank-nicholson"):
            identity = np.eye(self.collocation_matrix.shape[0])
            rhs = np.dot(
                identity+0.5*self.time_step_size*self.collocation_matrix,
                current_sol)
            current_forcing = self.forcing_fun(
                self.mesh.mesh_pts, sample, time)
            rhs += 0.5*self.time_step_size*(current_forcing+future_forcing)
        else:
            raise Exception('incorrect timestepping method specified')
        # apply boundary conditions
        return self.apply_boundary_conditions_to_rhs(rhs)

    def get_implicit_timestep_matrix_inverse_factors(self, matrix):
        identity = np.eye(matrix.shape[0])
        if (self.time_step_method == "backward-euler"):
            matrix = identity-self.time_step_size*matrix
        elif (self.time_step_method == "crank-nicholson"):
            matrix = identity-self.time_step_size/2.*matrix
        else:
            raise Exception('incorrect timestepping method specified')
        self.apply_boundary_conditions_to_matrix(matrix)
        return qr_factorization(matrix)

    def time_step(self, current_sol, time, sample):
        if (self.time_step_method == 'RK4' or
                self.time_step_method == 'forward-euler'):
            def rhs_fun(t, u): return np.dot(
                self.collocation_matrix, u) +\
                self.forcing_fun(self.mesh.mesh_pts, sample, t)
            if self.time_step_method == 'RK4':
                current_sol = self.explicit_runge_kutta(
                    rhs_fun, current_sol, time, self.time_step_size)
            else:
                current_sol = self.forward_euler(
                    rhs_fun, current_sol, time, self.time_step_size)
        else:
            rhs = self.get_implicit_time_step_rhs(current_sol, time, sample)
            current_sol = qr_solve(
                self.implicit_matrix_factors[0],
                self.implicit_matrix_factors[1],
                rhs)
            # current_sol = np.linalg.solve( matrix, rhs )

        return current_sol

    def solve(self, sample):
        assert sample.ndim == 1
        # in future consider supporting time varying diffusivity. This would
        # require updating collocation matrix at each time-step
        # for now make diffusivity time-independent

        # consider replacing time = 0 with time = self.initial_time
        time = 0.
        diffusivity = self.diffusivity_fun(self.mesh.mesh_pts, sample[:, None])
        advection = self.velocity_fun(self.mesh.mesh_pts, sample[:, None])
        # negate collocation matrix because it has moved from
        # lhs for steady state to rhs for transient
        self.collocation_matrix = -self.form_collocation_matrix(
            diffusivity, advection)
        assert (self.initial_sol is not None and
                self.initial_sol.shape[1] == 1)
        # make sure that if mesh has been updated initial sol has also been
        # updated
        assert self.initial_sol.shape[0] == self.collocation_matrix.shape[0]
        assert self.time_step_size is not None
        current_sol = self.initial_sol.copy()
        # num_time_steps is number of steps taken after initial time
        self.num_time_steps = int(
            np.ceil(self.final_time/self.time_step_size))
        self.times = np.empty((self.num_time_steps+1), float)
        sols = np.empty((self.initial_sol.shape[0],
                         self.num_time_steps+1), float)
        self.times[0] = time
        sols[:, 0] = self.initial_sol[:, 0]
        sol_cntr = 1
        if self.time_step_method not in ['RK4', 'forward-euler']:
            self.implicit_matrix_factors = \
                self.get_implicit_timestep_matrix_inverse_factors(
                    self.collocation_matrix)
        dt_tol = 1e-12
        while time < self.final_time-dt_tol:
            # Construct linear system
            current_sol = self.time_step(current_sol, time, sample)
            time += self.time_step_size
            time = min(time, self.final_time)

            # Store history
            sols[:, sol_cntr] = current_sol[:, 0].copy()
            self.times[sol_cntr] = time
            sol_cntr += 1
        return sols[:, :sol_cntr]

    def get_num_degrees_of_freedom(self, config_sample):
        order = config_sample[:-1]
        time_step_size = config_sample[-1]
        ntime_steps = self.final_time/time_step_size+1
        return np.prod(order)*ntime_steps



# Wrappers to maintain backwards compatability
class TransientAdvectionDiffusionEquation1D(TransientAdvectionDiffusion):
    pass


class TransientAdvectionDiffusionEquation2D(TransientAdvectionDiffusion):
    pass


class SteadyStateAdvectionDiffusionEquation1D(SteadyStateAdvectionDiffusion):
    pass


class SteadyStateAdvectionDiffusionEquation2D(SteadyStateAdvectionDiffusion):
    pass
