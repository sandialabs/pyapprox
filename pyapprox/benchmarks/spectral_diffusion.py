import numpy as np
import inspect
from scipy.linalg import qr as qr_factorization
from copy import deepcopy

from pyapprox.utilities import cartesian_product, outer_product
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


class SteadyStateAdvectionDiffusionEquation1D(object):
    """
    Solve  -(a(x)*u_x)_x+b(x)*u_x = f
    """

    def __init__(self):
        self.diffusivity_fun = None
        self.forcing_fun = None
        self.advection_fun = None
        self.domain = None
        self.adjoint_derivative_matrix = None
        self.adjoint_mesh_pts = None
        self.bndry_conds = None
        self.dirichlet_bndry_indices = None
        self.neumann_bndry_indices = None
        self.adjoint_bndry_conds = None
        self.adjoint_dirichlet_bndry_indices = None
        self.adjoint_neumann_bndry_indices = None

        # default qoi functional is integral of solution over entire domain
        self.qoi_functional = self.integrate
        self.qoi_functional_deriv = lambda x: x*0.+1.

    def map_samples_from_canonical_domain(self, canonical_pts):
        pts = map_hypercube_samples(canonical_pts, self.canonical_domain, self.domain)
        return pts

    def initialize(self, bndry_conds, diffusivity_fun, forcing_fun,
                   advection_fun, order, domain):
        self.set_domain(domain, order)
        self.set_boundary_conditions(bndry_conds)
        self.set_diffusivity_function(diffusivity_fun)
        self.set_forcing_function(forcing_fun)
        self.set_advection_function(advection_fun)

        self.gl_pts, self.gl_wts = gauss_jacobi_pts_wts_1D(order, 0, 0)

    def set_domain(self, domain, order):
        self.order = order
        assert len(domain) == 2
        self.domain = domain
        self.canonical_domain = [-1, 1]
        mesh_pts, self.derivative_matrix = chebyshev_derivative_matrix(order)
        # scale derivative matrix from [-1,1] to [a,b]
        self.derivative_matrix *= 2./(self.domain[1]-self.domain[0])
        self.mesh_pts_1d = mesh_pts
        self.mesh_pts = self.map_samples_from_canonical_domain(mesh_pts[None, :])
        # define boundary indices of each segment
        self.boundary_indices = [
            np.array([0]), np.array([self.derivative_matrix.shape[0]-1])]

    def set_boundary_conditions(self, bndry_conds):
        """
        Set time independent boundary conditions
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
            self.adjoint_bndry_conds.append(
                [lambda x: np.zeros((x.shape[1], 1)), "D"])

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
        return self.form_collocation_matrix_engine(
            self.derivative_matrix, diffusivity_vals, advection_vals)

    def form_collocation_matrix_engine(self, derivative_matrix,
                                       diffusivity_vals, advection_vals):
        assert diffusivity_vals.ndim == 2 and diffusivity_vals.shape[1] == 1
        assert advection_vals.ndim == 2 and advection_vals.shape[1] == 1
        # scaled_matrix = np.empty(derivative_matrix.shape)
        # for ii in range(scaled_matrix.shape[0]):
        #     scaled_matrix[ii, :]=derivative_matrix[ii, :]*diffusivity_vals[ii]
        scaled_matrix = diffusivity_vals*derivative_matrix
        matrix = np.dot(derivative_matrix, scaled_matrix)

        # matrix -= np.dot(derivative_matrix, np.diag(advection_vals[:, 0]))
        matrix -= derivative_matrix*advection_vals[:, 0]
        # want to solve u_xx-au_x+f=0 i.e. -u_xx+au_x=f so add negative
        return -matrix

    def apply_boundary_conditions_to_matrix(self, matrix):
        return self.apply_boundary_conditions_to_matrix_engine(
            matrix, self.dirichlet_bndry_indices, self.neumann_bndry_indices)

    def apply_boundary_conditions_to_matrix_engine(
            self, matrix, dirichlet_bndry_indices, neumann_bndry_indices):
        matrix[dirichlet_bndry_indices, :] = 0.0
        matrix[dirichlet_bndry_indices,
               dirichlet_bndry_indices] = 1.0
        matrix[neumann_bndry_indices, :] = \
            self.derivative_matrix[neumann_bndry_indices, :]
        return matrix

    def apply_boundary_conditions_to_rhs(self, rhs):
        return self.__apply_boundary_conditions_to_rhs(rhs, self.bndry_conds)

    def __apply_boundary_conditions_to_rhs(self, rhs, bndry_conds):
        for ii, bndry_cond in enumerate(bndry_conds):
            idx = self.boundary_indices[ii]
            rhs[idx] = bndry_cond[0](self.mesh_pts[:, idx])
        return rhs

    def apply_boundary_conditions(self, matrix, forcing):
        matrix = self.apply_boundary_conditions_to_matrix(matrix)
        forcing = self.apply_boundary_conditions_to_rhs(forcing)
        return matrix, forcing

    def solve(self, diffusivity, forcing, advection):
        assert diffusivity.ndim == 2 and diffusivity.shape[1] == 1
        assert forcing.ndim == 2 and forcing.shape[1] == 1
        assert advection.ndim == 2 and advection.shape[1] == self.mesh_pts.shape[0]
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

    def run(self, sample):
        assert sample.ndim == 1
        diffusivity = self.diffusivity_fun(self.mesh_pts, sample[:, None])
        forcing = self.forcing_fun(self.mesh_pts, sample[:, None])
        advection = self.advection_fun(self.mesh_pts, sample[:, None])
        solution = self.solve(diffusivity, forcing, advection)
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

        if order == self.order:
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
            matrix = self.form_collocation_matrix_engine(
                self.adjoint_derivative_matrix, diffusivity, advection)
            self.adjoint_collocation_matrix = matrix.copy()

        # regardless of whether computing error estimate or
        # computing gradient, rhs is always derivative (with respect to the
        # solution) of the qoi_functional
        qoi_deriv = self.qoi_functional_deriv(self.fwd_solution)

        matrix = self.apply_boundary_conditions_to_matrix_engine(
            matrix, self.adjoint_dirichlet_bndry_indices,
            self.adjoint_neumann_bndry_indices)
        qoi_deriv = self.__apply_boundary_conditions_to_rhs(
            qoi_deriv, self.adjoint_bndry_conds)
        adj_solution = np.linalg.solve(matrix, qoi_deriv)
        return adj_solution

    def compute_residual(self, matrix, solution, forcing):
        matrix, forcing = self.apply_boundary_conditions(matrix, forcing)
        return forcing - np.dot(matrix, solution)

    def compute_residual_derivative(self, solution, diffusivity_deriv,
                                    forcing_deriv, advection_deriv):
        matrix = self.form_collocation_matrix(
            diffusivity_deriv, advection_deriv)
        matrix = self.apply_boundary_conditions_to_matrix_engine(
            matrix, self.adjoint_dirichlet_bndry_indices,
            self.adjoint_neumann_bndry_indices)
        # the values here are the derivative of the boundary conditions
        # with respect to the random parameters. I assume that
        # this is always zero
        forcing_deriv = self.__apply_boundary_conditions_to_rhs(
            forcing_deriv, self.adjoint_bndry_conds)
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
        # qoi_deriv = self.qoi_functional_deriv(self.mesh_pts)
        adj_solution = self.solve_adjoint(sample, self.order)
        gradient = np.empty((num_stoch_dims), float)
        for ii in range(num_stoch_dims):
            diffusivity_deriv_vals_i = self.diffusivity_derivs_fun(
                self.mesh_pts, sample, ii)
            forcing_deriv_vals_i = self.forcing_derivs_fun(
                self.mesh_pts, sample, ii)
            advection_deriv_vals_i = self.advection_derivs_fun(
                self.mesh_pts, sample, ii)
            residual_deriv = self.compute_residual_derivative(
                self.fwd_solution, diffusivity_deriv_vals_i,
                forcing_deriv_vals_i, advection_deriv_vals_i)
            gradient[ii] = self.integrate(residual_deriv * adj_solution)
        return gradient

    def value(self, sample):
        assert sample.ndim == 1
        solution = self.run(sample)
        qoi = self.qoi_functional(solution)
        if np.isscalar(qoi) or qoi.ndim == 0:
            qoi = np.array([qoi])
        return qoi

    def integrate(self, mesh_values, order=None):
        if order is None:
            order = self.order
        # Scale points from [-1,1] to to physical domain
        x_range = self.domain[1]-self.domain[0]
        gl_pts = x_range*(self.gl_pts+1.)/2.+self.domain[0]
        # Remove factor of 0.5 from weights
        gl_wts = self.gl_wts*x_range
        # Interpolate mesh values onto quadrature nodes
        gl_vals = self.interpolate(mesh_values, gl_pts)
        # Compute and return integral
        return np.dot(gl_vals[:, 0], gl_wts)

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
        abscissa_1d = [self.mesh_pts_1d]*num_dims
        weights_1d = [compute_barycentric_weights_1d(xx) for xx in abscissa_1d]
        interp_vals = multivariate_barycentric_lagrange_interpolation(
            eval_samples,
            abscissa_1d,
            weights_1d,
            mesh_values,
            np.arange(num_dims))
        return interp_vals

    def plot(self, mesh_values, num_plot_pts_1d=None, plot_mesh_coords=None,
             color='k'):
        import pylab
        if num_plot_pts_1d is not None:
            # interpolate values onto plot points
            plot_mesh = np.linspace(
                self.domain[0], self.domain[1], num_plot_pts_1d)
            interp_vals = self.interpolate(mesh_values, plot_mesh)
            pylab.plot(plot_mesh, interp_vals, color+'-')

        elif plot_mesh_coords is not None:
            assert mesh_values.shape[0] == plot_mesh_coords.squeeze().shape[0]
            pylab.plot(plot_mesh_coords, mesh_values, 'o-'+color)
        else:
            # just plot values on mesh points
            pylab.plot(self.mesh_pts, mesh_values, color)

    def get_collocation_points(self):
        return self.mesh_pts

    def get_derivative_matrix(self):
        return self.derivative_matrix

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.value, samples, None)


class TransientAdvectionDiffusionEquation1D(
        SteadyStateAdvectionDiffusionEquation1D):
    """
    Solve  u_t=(a(x)*u_x)_x-b(x)*u_x + f
    """
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
        self.initial_sol = initial_sol_fun(self.mesh_pts)

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
            self.mesh_pts, sample, time+self.time_step_size)
        if (self.time_step_method == "backward-euler"):
            rhs = current_sol + self.time_step_size*future_forcing
        elif (self.time_step_method == "crank-nicholson"):
            identity = np.eye(self.collocation_matrix.shape[0])
            rhs = np.dot(
                identity+0.5*self.time_step_size*self.collocation_matrix,
                current_sol)
            current_forcing = self.forcing_fun(
                self.mesh_pts, sample, time)
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
                self.forcing_fun(self.mesh_pts, sample, t)
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

    def transient_solve(self, sample):
        # in future consider supporting time varying diffusivity. This would
        # require updating collocation matrix at each time-step
        # for now make diffusivity time-independent

        # consider replacing time = 0 with time = self.initial_time
        time = 0.
        diffusivity = self.diffusivity_fun(self.mesh_pts, sample)
        advection = self.advection_fun(self.mesh_pts, sample)
        # negate collocation matrix because it has moved from
        # lhs for steady state to rhs for transient
        self.collocation_matrix = -self.form_collocation_matrix(
            diffusivity, advection)

        assert (self.initial_sol is not None and
                self.initial_sol.shape[1] == 1)
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


class SteadyStateAdvectionDiffusionEquation2D(
        SteadyStateAdvectionDiffusionEquation1D):
    """
    solve  -(a(x)*u_x)_x + b(x)*u_x = f

    boundary edges are stored with the following order,
    left, right, bottom, top
    """
    def set_domain(self, domain, order):
        # boundary edges are stored with the following order,
        # left, right, bottom, top
        self.order = order
        assert len(domain) == 4
        self.domain = domain
        self.canonical_domain = [-1, 1, -1, 1]
        mesh_pts, derivative_matrix_1d = chebyshev_derivative_matrix(order)

        self.mesh_pts_1d = mesh_pts
        self.mesh_pts = self.map_samples_from_canonical_domain(
            cartesian_product([self.mesh_pts_1d]*2))

        # define boundary indices of each segment
        self.determine_boundary_indices()

        ident = np.eye(self.order+1)
        self.derivative_matrix_1 = np.kron(
            ident, derivative_matrix_1d*2./(self.domain[1]-self.domain[0]))
        # form derivative (in x2-direction) matrix of a 2d polynomial
        # this assumes that 2d-mesh_pts varies in x1 faster than x2,
        # e.g. points are
        # [[x11,x21],[x12,x21],[x13,x12],[x11,x22],[x12,x22],...]
        self.derivative_matrix_2 = np.kron(
            derivative_matrix_1d, ident*2./(self.domain[3]-self.domain[2]))

    def determine_boundary_indices(self):
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

    def form_collocation_matrix(self, diffusivity_vals, advection_vals):
        assert diffusivity_vals.ndim == 2 and diffusivity_vals.shape[1] == 1
        assert advection_vals.ndim == 2 and advection_vals.shape[1] == 2
        matrix_1 = np.dot(self.derivative_matrix_1,
                          diffusivity_vals*self.derivative_matrix_1)
        matrix_2 = np.dot(self.derivative_matrix_2,
                          diffusivity_vals*self.derivative_matrix_2)

        matrix_1 -= self.derivative_matrix_1*advection_vals[:, 0]
        matrix_2 -= self.derivative_matrix_2*advection_vals[:, 1]

        # want to solve -u_xx-u_yy+au_x+bu_y=f so add negative
        return -matrix_1 - matrix_2

    def apply_boundary_conditions_to_matrix_engine(
            self, matrix, dirichlet_bndry_indices, neumann_bndry_indices):
        matrix[dirichlet_bndry_indices, :] = 0.0
        matrix[dirichlet_bndry_indices,
               dirichlet_bndry_indices] = 1.0
        # left and right boundaries
        matrix[self.lr_neumann_bndry_indices, :] = \
            self.derivative_matrix_1[self.lr_neumann_bndry_indices, :]
        # bottom and top boundaries
        matrix[self.bt_neumann_bndry_indices, :] = \
            self.derivative_matrix_2[self.bt_neumann_bndry_indices, :]
        return matrix

    def plot(self, mesh_values, num_plot_pts_1d=100):
        if num_plot_pts_1d is not None:
            # interpolate values onto plot points
            def fun(x): return self.interpolate(mesh_values, x)
            from utilities.visualisation import plot_surface_from_function
            plot_surface_from_function(
                fun, self.domain, num_plot_pts_1d, False)

    def apply_adjoint_boundary_conditions_to_rhs(self, qoi_deriv):
        # adjoint always has zero Dirichlet BC
        # apply left boundary condition
        for ii in range(4):
            indices = self.boundary_indices[self.boundary_edges[ii]]
            qoi_deriv[indices] = 0
        return qoi_deriv

    def integrate(self, mesh_values, order=None):
        return self.integrate_subdomain(mesh_values, self.domain, order)

    def integrate_subdomain(self, mesh_values, subdomain, order=None):
        if order is None:
            order = self.order

        pts_1d, wts_1d = [], []
        for ii in range(2):
            # Scale points from [-1,1] to to physical domain
            x_range = subdomain[2*ii+1]-subdomain[2*ii]
            # Remove factor of 0.5 from weights and shift to [a,b]
            wts_1d.append(self.gl_wts*x_range)
            pts_1d.append(x_range*(self.gl_pts+1.)/2.+subdomain[2*ii])
        # Interpolate mesh values onto quadrature nodes
        pts = cartesian_product(pts_1d)
        wts = outer_product(wts_1d)
        gl_vals = self.interpolate(mesh_values, pts)
        # Compute and return integral
        return np.dot(gl_vals[:, 0], wts)
