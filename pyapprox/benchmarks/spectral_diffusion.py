import numpy as np
from scipy.linalg import qr as qr_factorization
from abc import ABC, abstractmethod
from functools import partial
import copy

from pyapprox.utilities import (
    cartesian_product, outer_product, get_tensor_product_quadrature_rule
)
from pyapprox.univariate_polynomials.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.barycentric_interpolation import (
    compute_barycentric_weights_1d,
    multivariate_barycentric_lagrange_interpolation
)
from pyapprox.variable_transformations import map_hypercube_samples
from pyapprox.models.wrappers import (
    evaluate_1darray_function_on_2d_array
)
from pyapprox.utilities import qr_solve
from pyapprox.sys_utilities import get_num_args
from pyapprox.variable_transformations import ConfigureVariableTransformation


def zeros_fun_axis_1(x):
    # axis_1 used when x is mesh points
    return np.zeros((x.shape[1], 1))


def ones_fun_axis_0(x):
    # axis_1 used when x is solution like quantity
    return np.zeros((x.shape[0], 1))


def kronecker_product_2d(matrix1, matrix2):
    """
    TODO: I can store kroneker as a sparse matrix see ( scipy.kron )
    """
    assert matrix1.shape == matrix2.shape
    assert matrix1.ndim == 2
    block_num_rows = matrix1.shape[0]
    matrix_num_rows = block_num_rows**2
    matrix = np.empty((matrix_num_rows, matrix_num_rows), float)

    # loop through blocks
    start_col = 0
    for jj in range(block_num_rows):
        start_row = 0
        for ii in range(block_num_rows):
            matrix[start_row:start_row+block_num_rows,
                   start_col:start_col+block_num_rows] = \
                matrix2*matrix1[ii, jj]
            start_row += block_num_rows
        start_col += block_num_rows
    return matrix


def chebyshev_derivative_matrix(order):
    if order == 0:
        pts = np.array([1], float)
        derivative_matrix = np.array([0], float)
    else:
        # this is reverse order used by matlab cheb function
        pts = -np.cos(np.linspace(0., np.pi, order+1))
        scalars = np.ones((order+1), float)
        scalars[0] = 2.
        scalars[order] = 2.
        scalars[1:order+1:2] *= -1
        derivative_matrix = np.empty((order+1, order+1), float)
        for ii in range(order+1):
            row_sum = 0.
            for jj in range(order+1):
                if (ii == jj):
                    denominator = 1.
                else:
                    denominator = pts[ii]-pts[jj]
                numerator = scalars[ii] / scalars[jj]
                derivative_matrix[ii, jj] = numerator / denominator
                row_sum += derivative_matrix[ii, jj]
            derivative_matrix[ii, ii] -= row_sum

    # I return points and calculate derivatives using reverse order of points
    # compared to what is used by Matlab cheb function thus the
    # derivative matrix I return will be the negative of the matlab version
    return pts, derivative_matrix


class AbstractAdvectionDiffusion(ABC):
    r"""
    Solve the advection diffusion equation
    .. math:: u_t(x)-(a(x)*u_x(x))_x+b(x)*u_x(x) = f
    """
    def __init__(self):
        self.diffusivity_fun = None
        self.forcing_fun = None
        self.advection_fun = None
        self.domain = None
        self.qoi_functional = None

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
                   advection_fun, order, domain):
        self.set_domain(domain, order)
        self.mesh.set_boundary_conditions(bndry_conds)
        self.set_diffusivity_function(diffusivity_fun)
        self.set_forcing_function(forcing_fun)
        self.set_advection_function(advection_fun)
        # default qoi functional is integral of solution over entire domain
        # at the final time-step or for steady-state problems the only solution
        self.set_qoi_functional(self.mesh.integrate, ones_fun_axis_0)

    def set_diffusivity_function(self, fun):
        assert callable(fun)
        assert get_num_args(fun) == 2
        self.diffusivity_fun = fun

    def set_advection_function(self, fun):
        assert callable(fun)
        assert get_num_args(fun) == 2
        self.advection_fun = fun

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
        # from pyapprox.configure_plots import plt
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


class SteadyStateAdvectionDiffusion(AbstractAdvectionDiffusion):
    def __init__(self):
        super().__init__()
        self.adjoint_derivative_matrix = None
        self.adjoint_mesh_pts = None

    def solve(self, sample):
        assert sample.ndim == 1
        diffusivity = self.diffusivity_fun(self.mesh.mesh_pts, sample[:, None])
        forcing = self.forcing_fun(self.mesh.mesh_pts, sample[:, None])
        advection = self.advection_fun(self.mesh.mesh_pts, sample[:, None])

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
            advection = self.advection_fun(self.adjoint_mesh_pts, sample)
            matrix = self._form_collocation_matrix(
                self.adjoint_derivative_matrix, diffusivity, advection)
            self.adjoint_collocation_matrix = matrix.copy()

        # regardless of whether computing error estimate or
        # computing gradient, rhs is always derivative (with respect to the
        # solution) of the qoi_functional
        qoi_deriv = self.qoi_functional_deriv(self.fwd_solution)

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


class TransientAdvectionDiffusion(
        SteadyStateAdvectionDiffusion):
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
                   advection_fun, order, domain, final_time, time_step_size,
                   initial_sol_fun, time_step_method="crank-nicholson"):
        super().initialize(bndry_conds, diffusivity_fun, forcing_fun,
                           advection_fun, order, domain)
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
        advection = self.advection_fun(self.mesh.mesh_pts, sample[:, None])
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


def integrate_subdomain(mesh_values, subdomain, order, interpolate,
                        quad_rule=None):
    # Keep separate from model class so that pre-computed so
    # it can also be used by QoI functions
    if quad_rule is None:
        quad_pts, quad_wts = get_2d_sub_domain_quadrature_rule(
            subdomain, order)
    else:
        quad_pts, quad_wts = quad_rule
    quad_vals = interpolate(mesh_values, quad_pts)
    # Compute and return integral
    return np.dot(quad_vals[:, 0], quad_wts)


def get_2d_sub_domain_quadrature_rule(subdomain, order):
    pts_1d, wts_1d = [], []
    for ii in range(2):
        gl_pts, gl_wts = gauss_jacobi_pts_wts_1D(order[ii], 0, 0)
        # Scale points from [-1,1] to to physical domain
        x_range = subdomain[2*ii+1]-subdomain[2*ii]
        # Remove factor of 0.5 from weights and shift to [a,b]
        wts_1d.append(gl_wts*x_range)
        pts_1d.append(x_range*(gl_pts+1.)/2.+subdomain[2*ii])
    # Interpolate mesh values onto quadrature nodes
    pts = cartesian_product(pts_1d)
    wts = outer_product(wts_1d)
    return pts, wts


class AbstractCartesianProductCollocationMesh(ABC):
    def __init__(self, domain, order):
        self.bndry_conds = None
        self.dirichlet_bndry_indices = []
        self.neumann_bndry_indices = []
        self.set_domain(domain, order)

    def _expand_order(self, order):
        order = np.atleast_1d(order).astype(int)
        assert order.ndim == 1
        if order.shape[0] == 1 and self.nphys_vars > 1:
            order = np.array([order[0]]*self.nphys_vars, dtype=int)
        if order.shape[0] != self.nphys_vars:
            msg = "order must be a scalar or an array_like with an entry for"
            msg += " each physical dimension"
            raise ValueError(msg)
        return order

    def set_domain(self, domain, order):
        self.domain = np.asarray(domain)
        self.nphys_vars = self.domain.shape[0]//2
        self.order = self._expand_order(order)
        self.canonical_domain = np.ones(2*self.nphys_vars)
        self.canonical_domain[::2] = -1.
        self.form_derivative_matrices()
        self.quad_pts, self.quad_wts = self.form_quadrature_rule(
            self.order, self.domain)
        self._determine_boundary_indices()

    def map_samples_from_canonical_domain(self, canonical_pts):
        pts = map_hypercube_samples(
            canonical_pts, self.canonical_domain, self.domain)
        return pts

    def form_quadrature_rule(self, order, domain):
        univariate_quad_rules = [
            partial(gauss_jacobi_pts_wts_1D, alpha_poly=0, beta_poly=0)
        ]*self.nphys_vars
        quad_pts, quad_wts = \
            get_tensor_product_quadrature_rule(
                order, self.nphys_vars, univariate_quad_rules,
                transform_samples=self.map_samples_from_canonical_domain)
        quad_wts *= np.prod(domain[1::2]-domain[::2])
        return quad_pts, quad_wts

    def form_derivative_matrices(self):
        self.mesh_pts_1d, self.derivative_matrices_1d = [], []
        for ii in range(self.nphys_vars):
            mpts, der_mat = chebyshev_derivative_matrix(self.order[ii])
            self.mesh_pts_1d.append(mpts)
            self.derivative_matrices_1d.append(der_mat)

        self.mesh_pts = self.map_samples_from_canonical_domain(
            cartesian_product(self.mesh_pts_1d))
        self._form_derivative_matrices()

    @abstractmethod
    def _form_derivative_matrices(self):
        raise NotImplementedError()

    @abstractmethod
    def _determine_boundary_indices(self):
        raise NotImplementedError()

    def integrate(self, mesh_values, order=None, domain=None, qoi=None):
        if order is None:
            order = self.order
            quad_pts, quad_wts = self.quad_pts, self.quad_wts
        else:
            order = self._expand_order(order, domain)
            quad_pts, quad_wts = self.form_quadrature_rule(order)
        if domain is None:
            domain = self.domain
            assert np.all(domain[::2] >= self.domain[::2])
            assert np.all(domain[1::2] <= self.domain[1::2])
        if qoi is None:
            qoi = mesh_values.shape[1]-1

        quad_vals = self.interpolate(mesh_values[:, qoi:qoi+1], quad_pts)
        # Compute and return integral
        return np.dot(quad_vals[:, 0], quad_wts)

    def interpolate(self, mesh_values, eval_samples):
        if eval_samples.ndim == 1:
            eval_samples = eval_samples[None, :]
        assert eval_samples.shape[0] == self.mesh_pts.shape[0]
        if mesh_values.ndim == 1:
            mesh_values = mesh_values[:, None]
        assert mesh_values.ndim == 2
        num_dims = eval_samples.shape[0]  # map samples to canonical domain
        eval_samples = map_hypercube_samples(
            eval_samples, self.domain, self.canonical_domain)
        # mesh_pts_1d are in canonical domain
        abscissa_1d = self.mesh_pts_1d
        weights_1d = [compute_barycentric_weights_1d(xx) for xx in abscissa_1d]
        interp_vals = multivariate_barycentric_lagrange_interpolation(
            eval_samples,
            abscissa_1d,
            weights_1d,
            mesh_values,
            np.arange(num_dims))
        return interp_vals

    def set_boundary_conditions(self, bndry_conds):
        """
        Set time independent boundary conditions. If time dependent BCs
        are needed. Then this function must be called at every timestep
        """
        self.bndry_conds = bndry_conds
        self.dirichlet_bndry_indices = []
        self.neumann_bndry_indices = []

        if len(bndry_conds) != len(self.boundary_indices):
            msg = "Incorrect number of boundary conditions provided.\n"
            msg += f"\tNum boundary edges {len(self.boundary_indices)}\n"
            msg += f"\tNum boundary conditions provided {len(bndry_conds)}\n"
            raise ValueError(msg)

        # increase this if make boundary time dependent and/or
        # parameter dependent
        num_args = 1
        self.adjoint_bndry_conds = []
        for ii, bndry_cond in enumerate(bndry_conds):
            assert len(bndry_cond) == 2
            assert callable(bndry_cond[0])
            if get_num_args(bndry_cond[0]) != num_args:
                msg = f"Boundary condition function must have {num_args} "
                msg += "arguments"
                raise ValueError(msg)
            idx = self.boundary_indices[ii]
            if bndry_cond[1] == "D":
                self.dirichlet_bndry_indices.append(idx)
            elif bndry_cond[1] == "N":
                self.neumann_bndry_indices.append(idx)
            else:
                raise ValueError("Incorrect boundary type")
            # adjoint boundary conditions are always dirichlet zero
            self.adjoint_bndry_conds.append([zeros_fun_axis_1, "D"])

        if len(self.dirichlet_bndry_indices) > 0:
            self.dirichlet_bndry_indices = np.concatenate(
                self.dirichlet_bndry_indices)
        else:
            self.dirichlet_bndry_indices = np.empty((0), dtype=int)
        if len(self.neumann_bndry_indices) > 0:
            self.neumann_bndry_indices = np.concatenate(
                self.neumann_bndry_indices)
        else:
            self.neumann_bndry_indices = np.empty((0), dtype=int)

        if (np.sum([len(idx) for idx in self.boundary_indices]) !=
                len(self.dirichlet_bndry_indices) +
                len(self.neumann_bndry_indices)):
            raise ValueError("Boundary indices mismatch")

        self.adjoint_dirichlet_bndry_indices = np.concatenate((
            self.dirichlet_bndry_indices, self.neumann_bndry_indices))
        self.adjoint_neumann_bndry_indices = np.empty((0), dtype=int)

    def _apply_dirichlet_boundary_conditions_to_matrix(
            self, matrix, dirichlet_bndry_indices):
        # needs to have indices as argument so this fucntion can be used
        # when setting boundary conditions for forward and adjoint solves
        matrix[dirichlet_bndry_indices, :] = 0.0
        matrix[dirichlet_bndry_indices,
               dirichlet_bndry_indices] = 1.0
        return matrix

    @abstractmethod
    def _apply_neumann_boundary_conditions_to_matrix(self, matrix):
        raise NotImplementedError()

    def _apply_boundary_conditions_to_matrix(self, matrix):
        matrix = self._apply_dirichlet_boundary_conditions_to_matrix(
            matrix, self.dirichlet_bndry_indices)
        return self._apply_neumann_boundary_conditions_to_matrix(matrix)

    def _apply_boundary_conditions_to_rhs(self, rhs, bndry_conds):
        for ii, bndry_cond in enumerate(bndry_conds):
            idx = self.boundary_indices[ii]
            rhs[idx] = bndry_cond[0](self.mesh_pts[:, idx])
        return rhs


class OneDCollocationMesh(AbstractCartesianProductCollocationMesh):
    def _form_derivative_matrices(self):
        if self.domain.shape[0] != 2:
            raise ValueError("Domain must be 1D")
        self.derivative_matrices = [
            self.derivative_matrices_1d[0]*2./(self.domain[1]-self.domain[0])]

    def _determine_boundary_indices(self):
        self.boundary_indices = [
            np.array([0]), np.array(
                [self.derivative_matrices[0].shape[0]-1])]

    def _apply_neumann_boundary_conditions_to_matrix(self, matrix):
        matrix[self.neumann_bndry_indices, :] = \
            self.derivative_matrices[0][self.neumann_bndry_indices, :]
        return matrix

    def plot(self, mesh_values, num_plot_pts_1d=None, plot_mesh_coords=None,
             color='k'):
        import pylab as plt
        if num_plot_pts_1d is not None:
            # interpolate values onto plot points
            plot_mesh = np.linspace(
                self.domain[0], self.domain[1], num_plot_pts_1d)
            interp_vals = self.interpolate(mesh_values, plot_mesh)
            plt.plot(plot_mesh, interp_vals, color+'-')

        elif plot_mesh_coords is not None:
            assert mesh_values.shape[0] == plot_mesh_coords.squeeze().shape[0]
            plt.plot(plot_mesh_coords, mesh_values, 'o-'+color)
        else:
            # just plot values on mesh points
            plt.plot(self.mesh_pts, mesh_values, color)


class RectangularCollocationMesh(AbstractCartesianProductCollocationMesh):
    def _form_derivative_matrices(self):
        if self.domain.shape[0] != 4:
            raise ValueError("Domain must be 2D")
        ident1 = np.eye(self.order[1]+1)
        ident2 = np.eye(self.order[0]+1)
        self.derivative_matrices = []
        self.derivative_matrices.append(np.kron(
            ident1,
            self.derivative_matrices_1d[0]*2./(self.domain[1]-self.domain[0])))
        # form derivative (in x2-direction) matrix of a 2d polynomial
        # this assumes that 2d-mesh_pts varies in x1 faster than x2,
        # e.g. points are
        # [[x11,x21],[x12,x21],[x13,x12],[x11,x22],[x12,x22],...]
        self.derivative_matrices.append(np.kron(
            self.derivative_matrices_1d[1],
            ident2*2./(self.domain[3]-self.domain[2])))

    def _determine_boundary_indices(self):
        # boundary edges are stored with the following order,
        # left, right, bottom, top
        self.boundary_indices = [[] for ii in range(4)]
        self.boundary_indices_to_edges_map = -np.ones(self.mesh_pts.shape[1])
        # Todo avoid double counting the bottom and upper boundaries
        for ii in range(self.mesh_pts.shape[1]):
            if (self.mesh_pts[0, ii] == self.domain[0]):
                self.boundary_indices[0].append(ii)
                self.boundary_indices_to_edges_map[ii] = 0
            elif (self.mesh_pts[0, ii] == self.domain[1]):
                self.boundary_indices[1].append(ii)
                self.boundary_indices_to_edges_map[ii] = 1
            elif (self.mesh_pts[1, ii] == self.domain[2]):
                self.boundary_indices[2].append(ii)
                self.boundary_indices_to_edges_map[ii] = 2
            elif (self.mesh_pts[1, ii] == self.domain[3]):
                self.boundary_indices[3].append(ii)
                self.boundary_indices_to_edges_map[ii] = 3

        self.boundary_indices = [
            np.array(idx, dtype=int) for idx in self.boundary_indices]

    def set_boundary_conditions(self, bndry_conds):
        super().set_boundary_conditions(bndry_conds)
        II = np.where(
            self.boundary_indices_to_edges_map[self.neumann_bndry_indices] < 2)
        self.lr_neumann_bndry_indices = self.neumann_bndry_indices[II]
        JJ = np.where(
            self.boundary_indices_to_edges_map[self.neumann_bndry_indices] > 1)
        self.bt_neumann_bndry_indices = self.neumann_bndry_indices[JJ]

    def _apply_neumann_boundary_conditions_to_matrix(self, matrix):
        # left and right boundaries
        matrix[self.lr_neumann_bndry_indices, :] = \
            self.derivative_matrices[0][self.lr_neumann_bndry_indices, :]
        # bottom and top boundaries
        matrix[self.bt_neumann_bndry_indices, :] = \
            self.derivative_matrices[1][self.bt_neumann_bndry_indices, :]
        return matrix

    def plot(self, mesh_values, num_pts_1d=100, ncontour_levels=20, ax=None):
        from pyapprox.visualization import get_meshgrid_function_data, plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # interpolate values onto plot points
        def fun(x): return self.interpolate(mesh_values, x)
        X, Y, Z = get_meshgrid_function_data(
            fun, self.domain, num_pts_1d, qoi=0)
        return ax.contourf(
            X, Y, Z, levels=np.linspace(Z.min(), Z.max(), ncontour_levels))


# Wrappers to maintain backwards compatability
class TransientAdvectionDiffusionEquation1D(TransientAdvectionDiffusion):
    pass


class TransientAdvectionDiffusionEquation2D(TransientAdvectionDiffusion):
    pass


class SteadyStateAdvectionDiffusionEquation1D(SteadyStateAdvectionDiffusion):
    pass


class SteadyStateAdvectionDiffusionEquation2D(SteadyStateAdvectionDiffusion):
    pass


class NobileBenchmarkFunctions(object):
    def __init__(self, nvars, corr_len):
        self.bndry_conds = [[zeros_fun_axis_1, "D"],
                            [zeros_fun_axis_1, "D"],
                            [zeros_fun_axis_1, "D"],
                            [zeros_fun_axis_1, "D"]]

        domain_len = 1
        self.corr_len, self.nvars = corr_len, nvars
        self.Lp = max(domain_len, 2*corr_len)
        self.L = self.corr_len/self.Lp
        self.II = np.arange(2, 2+nvars-1)//2
        self.eigenvalues = np.sqrt(np.sqrt(np.pi)*self.L)*(
            np.exp(-(self.II*np.pi*self.L)**2/8.0))

    def diffusivity_fun(self, mesh_pts, sample):
        assert sample.ndim == 2
        values = np.ones((mesh_pts.shape[1], 1))
        values += sample[0]*np.sqrt(np.sqrt(np.pi)*self.L/2.0)
        basis_vals = np.sin(self.II[:, None]*np.pi*mesh_pts[:1, :]/self.Lp)
        values += (self.eigenvalues[:, None]*basis_vals).T.dot(
            sample[1:])
        return np.exp(values)

    def forcing_fun(self, mesh_pts, sample):
        return self.time_dependent_forcing_fun(mesh_pts, sample, 0.0)

    def time_dependent_forcing_fun(self, mesh_pts, sample, time):
        return (1.5 + np.cos(2*np.pi*time))*np.cos(mesh_pts[0, :])[:, None]

    def initial_condition_fun(self, mesh_pts):
        # return np.zeros((mesh_pts.shape[1], 1))
        return np.prod(1-(2*mesh_pts-1)**2, axis=0)[:, None]

    def velocity_fun(self, mesh_pts, sample):
        return np.zeros((mesh_pts.shape[1], 2))


class SpectralPDEMultiIndexWrapper(object):
    def __init__(self, model, nvars):
        self.model = model
        self.nvars = nvars
        self.num_config_vars = model.mesh.nphys_vars
        if hasattr(self.model, "time_step_size"):
            self.num_config_vars += 1

    def value(self, sample):
        if sample.shape[0] != self.nvars+self.num_config_vars:
            msg = f"sample has shape {sample.shape} but expected "
            msg += f"{(self.nvars+self.num_config_vars, 1)}"
            raise ValueError(msg)
        if hasattr(self.model, "time_step_size"):
            order = sample[self.nvars:-1]
            time_step_size = sample[-1]
        else:
            order = sample[self.nvars:-1]
            time_step_size = None
        assert np.allclose(order.astype(int), order)
        order = order.astype(int)
        # must extract bndry_conds before set_domain which
        # creates new mesh and thus does not know about bndry_conds
        bndry_conds = self.model.mesh.bndry_conds
        # must use mesh.set_domain and not model.set_subdomain or
        # internal variables will sometimes be inconsistent
        self.model.mesh.set_domain(self.model.mesh.domain, order)
        self.model.mesh.set_boundary_conditions(bndry_conds)
        if time_step_size is not None:
            self.model.initial_sol = self.model.initial_sol_fun(
                self.model.mesh.mesh_pts)
            self.model.set_time_step_size(
                self.model.final_time, time_step_size)
        random_sample = sample[:self.nvars]
        value = self.model.value(random_sample)
        return value

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.value, samples, None)


def setup_multi_index_advection_diffusion_benchmark(
        nvars, corr_len, degree, final_time=1, time_step_size=1e-2,
        max_eval_concurrency=1):
    r"""
    Compute functionals of the following model of transient advection-diffusion (with 3 configure variables which control the two spatial mesh resolutions and the timestep)

    .. math::

       \frac{\partial u}{\partial t}(x,t,\rv) + \nabla u(x,t,\rv)-\nabla\cdot\left[k(x,\rv) \nabla u(x,t,\rv)\right] &=g(x,t) \qquad (x,t,\rv)\in D\times [0,1]\times\rvdom\\
       \mathcal{B}(x,t,\rv)&=0 \qquad\qquad (x,t,\rv)\in \partial D\times[0,1]\times\rvdom\\
       u(x,t,\rv)&=u_0(x,\rv) \qquad (x,t,\rv)\in D\times\{t=0\}\times\rvdom

    Following [NTWSIAMNA2008]_, [JEGGIJNME2020]_ we set

    .. math:: g(x,t)=(1.5+\cos(2\pi t))\cos(x_1),

    the initial condition as :math:`u(x,z)=0`, :math:`B(x,t,z)` to be zero dirichlet boundary conditions.

    and we model the diffusivity :math:`k` as a random field represented by the
    Karhunen-Loeve (like) expansion (KLE)

    .. math::

       \log(k(x,\rv)-0.5)=1+\rv_1\left(\frac{\sqrt{\pi L}}{2}\right)^{1/2}+\sum_{k=2}^d \lambda_k\phi(x)\rv_k,

    with

    .. math::

       \lambda_k=\left(\sqrt{\pi L}\right)^{1/2}\exp\left(-\frac{(\lfloor\frac{k}{2}\rfloor\pi L)^2}{4}\right) k>1,  \qquad\qquad  \phi(x)=
       \begin{cases}
       \sin\left(\frac{(\lfloor\frac{k}{2}\rfloor\pi x_1)}{L_p}\right) & k \text{ even}\,,\\
       \cos\left(\frac{(\lfloor\frac{k}{2}\rfloor\pi x_1)}{L_p}\right) & k \text{ odd}\,.
       \end{cases}

    where :math:`L_p=\max(1,2L_c)`, :math:`L=\frac{L_c}{L_p}`.

    The quantity of interest :math:`f(z)` is the measurement of the solution at a location :math:`x_k` at the final time :math:`T=1` obtained via the linear functional

    .. math:: f(z)=\int_D u(x,T,z)\frac{1}{2\pi\sigma^2}\exp\left(-\frac{\lVert x-x_k \rVert^2_2}{\sigma^2}\right) dx


    Parameters
    ----------
    nvars : integer
        The number of variables of the KLE

    corr_len : float
        The correlation length :math:`L_c` of the covariance kernel

    max_eval_concurrency : integer
        The maximum number of simulations that can be run in parallel. Should be         no more than the maximum number of cores on the computer being used

    Returns
    -------
    benchmark : pya.Benchmark
       Object containing the benchmark attributes documented below

    fun : callable

        The quantity of interest :math:`f(w)` with signature

        ``fun(w) -> np.ndarray``

        where ``w`` is a 2D np.ndarray with shape (nvars+3,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1). The first ``nvars``
        rows of ``w`` are realizations of the random variables. The last 3 rows
        are configuration variables specifying the numerical discretization of
        the PDE model. Specifically the first and second configuration variables
        specify the levels :math:`l_{x_1}` and :math:`l_{x_2}` which dictate
        the resolution of the FEM mesh in the directions :math:`{x_1}` and
        :math:`{x_2}` respectively. The number of cells in the :math:`{x_i}`
        direction is given by :math:`2^{l_{x_i}+2}`. The third configuration
        variable specifies the level :math:`l_t` of the temporal discretization.
        The number of timesteps satisfies :math:`2^{l_{t}+2}` so the timestep
        size is and :math:`T/2^{l_{t}+2}`.

    variable : pya.IndependentMultivariateRandomVariable
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        uniform variables on :math:`[-\sqrt{3},\sqrt{3}]`.

    Examples
    --------
    >>> from pyapprox_dev.benchmarks.benchmarks import setup_benchmark
    >>> benchmark = setup_benchmark('advection-diffusion', nvars=2)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'variable'])
    """
    from scipy import stats
    from pyapprox.models.wrappers import TimerModelWrapper, PoolModel, \
        WorkTrackingModel
    from pyapprox.variables import IndependentMultivariateRandomVariable
    from pyapprox.benchmarks.benchmarks import Benchmark
    univariate_variables = [stats.uniform(-np.sqrt(3), 2*np.sqrt(3))]*nvars
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    pde_funs = NobileBenchmarkFunctions(nvars, corr_len)
    config_values = [2*np.arange(1, 11)+1, 2*np.arange(1, 11)+1]
    if final_time is not None:
        base_model = TransientAdvectionDiffusion()
        base_model.initialize(
            pde_funs.bndry_conds, pde_funs.diffusivity_fun,
            pde_funs.time_dependent_forcing_fun, pde_funs.velocity_fun,
            degree, [0, 1, 0, 1], final_time, time_step_size,
            pde_funs.initial_condition_fun)
        config_values.append(final_time/(2*(2*np.arange(1, 11))**2))
    else:
        base_model = SteadyStateAdvectionDiffusion()
        base_model.initialize(pde_funs.bndry_conds, pde_funs.diffusivity_fun,
                              pde_funs.forcing_fun, pde_funs.velocity_fun,
                              degree, [0, 1, 0, 1])
    multi_index_model = SpectralPDEMultiIndexWrapper(
        base_model, variable.num_vars())
    config_var_trans = ConfigureVariableTransformation(config_values)

    # add wrapper to allow execution times to be captured
    timer_model = TimerModelWrapper(multi_index_model, multi_index_model)
    #pool_model = PoolModel(
    #    timer_model, max_eval_concurrency, base_model=base_model)
    pool_model = timer_model

    # add wrapper that tracks execution times.
    model = WorkTrackingModel(pool_model, multi_index_model,
                              multi_index_model.num_config_vars)
    attributes = {
        'fun': model, 'variable': variable,
        "get_num_degrees_of_freedom": base_model.get_num_degrees_of_freedom,
        "config_var_trans": config_var_trans}
    return Benchmark(attributes)
