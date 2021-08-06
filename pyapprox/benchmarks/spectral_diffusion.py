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
from pyapprox.models.wrappers import (
    evaluate_1darray_function_on_2d_array
)
from pyapprox.utilities import qr_solve


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


class SteadyStateDiffusionEquation1D(object):
    """
    solve  (a(x)*u_x)_x = f; x in [0,1]; subject to u(0)=a; u(1)=b
    """

    def __init__(self):
        self.diffusivity = None
        self.forcing_function = None
        self.bndry_cond = [0., 0.]
        self.xlim = [0, 1]
        self.adjoint_derivative_matrix = None
        self.adjoint_mesh_pts = None
        self.num_time_steps = 0
        self.time_step_size = None
        self.initial_sol = None
        self.num_stored_timesteps = 1
        self.time_step_method = 'crank-nicholson'

        # default qoi functional is integral of solution over entire domain
        self.qoi_functional = self.integrate
        self.qoi_functional_deriv = lambda x: x*0.+1.

    def scale_canonical_pts(self, pts):
        return (self.xlim[1]-self.xlim[0])*(pts+1.)/2.+self.xlim[0]

    def initialize(self, order, bndry_cond=None, xlim=None):
        self.order = order
        if xlim is not None:
            self.xlim = xlim
        if bndry_cond is not None:
            self.bndry_cond = bndry_cond

        mesh_pts, self.derivative_matrix = chebyshev_derivative_matrix(order)
        # scale mesh points to from [-1,1] to [a,b]
        self.mesh_pts_1d = self.scale_canonical_pts(mesh_pts)
        self.mesh_pts = self.mesh_pts_1d
        # scale derivative matrix from [-1,1] to [a,b]
        self.derivative_matrix *= 2./(self.xlim[1]-self.xlim[0])

    def set_diffusivity(self, func):
        assert callable(func)
        assert len(inspect.getargspec(func)[0]) == 2
        self.diffusivity = func

    def set_forcing(self, func):
        assert callable(func)
        assert len(inspect.getargspec(func)[0]) == 2
        self.forcing_function = func

    def form_collocation_matrix(self, derivative_matrix, diagonal):
        scaled_matrix = np.empty(derivative_matrix.shape)
        for i in range(scaled_matrix.shape[0]):
            scaled_matrix[i, :] = derivative_matrix[i, :] * diagonal[i]
        matrix = np.dot(derivative_matrix, scaled_matrix)
        return matrix

    def apply_boundary_conditions_to_matrix(self, matrix):
        matrix[0, :] = 0
        matrix[-1, :] = 0
        matrix[0, 0] = 1
        matrix[-1, -1] = 1
        return matrix

    def apply_boundary_conditions_to_rhs(self, rhs):
        rhs[0] = self.bndry_cond[0]
        rhs[-1] = self.bndry_cond[1]
        return rhs

    def apply_boundary_conditions(self, matrix, forcing):
        assert len(self.bndry_cond) == 2
        matrix = self.apply_boundary_conditions_to_matrix(matrix)
        forcing = self.apply_boundary_conditions_to_rhs(forcing)
        return matrix, forcing

    def explicit_runge_kutta(self, rhs, sol, time, time_step_size):
        assert callable(rhs)
        dt2 = time_step_size/2.
        k1 = rhs(time, sol)
        k2 = rhs(time+dt2, sol+dt2*k1)
        k3 = rhs(time+dt2, sol+dt2*k2)
        k4 = rhs(time+time_step_size, sol+time_step_size*k3)
        new_sol = sol+time_step_size/6.*(k1+2.*k2+2.*k3+k4)
        new_sol[0] = self.bndry_cond[0]
        new_sol[-1] = self.bndry_cond[1]
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
        forcing = np.dot(identity+8.*dt12matrix, current_sol)
        forcing += dt12*(5.*future_forcing+8.*current_forcing-prev_forcing)
        forcing -= np.dot(dt12matrix, prev_sol)
        # currently I do not support time varying boundary conditions
        return self.apply_boundary_conditions(matrix, forcing)

    def get_implicit_time_step_rhs(self, current_sol, time, sample):
        future_forcing = self.forcing_function(
            self.mesh_pts, time+self.time_step_size, sample)
        if (self.time_step_method == "backward-euler"):
            forcing = current_sol + self.time_step_size*future_forcing
        elif (self.time_step_method == "crank-nicholson"):
            identity = np.eye(self.collocation_matrix.shape[0])
            forcing = np.dot(
                identity+0.5*self.time_step_size*self.collocation_matrix, current_sol)
            current_forcing = self.forcing_function(
                self.mesh_pts, time, sample)
            forcing += 0.5*self.time_step_size*(current_forcing+future_forcing)
        else:
            raise Exception('incorrect timestepping method specified')

        # apply boundary conditions
        forcing[0] = self.bndry_cond[0]
        forcing[-1] = self.bndry_cond[1]
        return forcing

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
        if self.time_step_method == 'RK4':
            def rhs_func(t, u): return np.dot(
                self.collocation_matrix, u) +\
                self.forcing_function(self.mesh_pts, t, sample)
            current_sol = self.explicit_runge_kutta(
                rhs_func, current_sol, time, self.time_step_size)
        else:
            rhs = self.get_implicit_time_step_rhs(current_sol, time, sample)
            current_sol = qr_solve(
                self.implicit_matrix_factors[0], self.implicit_matrix_factors[1],
                rhs[:, None])[:, 0]
            #current_sol = np.linalg.solve( matrix, rhs )

        return current_sol

    def transient_solve(self, sample):
        # in future consider supporting time varying diffusivity. This would
        # require updating collocation matrix at each time-step
        # for now make diffusivity time-independent
        # assert self.diffusivity_function.__code__.co_argcount == 3
        diffusivity = self.diffusivity_function(self.mesh_pts, sample)
        self.collocation_matrix = self.form_collocation_matrix(
            self.derivative_matrix, diffusivity)

        # consider replacing time = 0 with time = self.initial_time
        time = 0.

        assert self.forcing_function.__code__.co_argcount == 3
        current_forcing = self.forcing_function(self.mesh_pts, time, sample)
        if self.num_time_steps > 0:
            assert self.initial_sol is not None
            assert self.time_step_size is not None
            current_sol = self.initial_sol.copy()
            assert self.num_stored_timesteps <= self.num_time_steps
            # num_time_steps is number of steps taken after initial time
            self.times = np.empty((self.num_stored_timesteps), float)
            sols = np.empty((self.initial_sol.shape[0],
                             self.num_stored_timesteps), float)
            sol_cntr = 0
            sol_storage_stride = self.num_time_steps/self.num_stored_timesteps
            if self.time_step_method != 'RK4':
                self.implicit_matrix_factors = \
                    self.get_implicit_timestep_matrix_inverse_factors(
                        self.collocation_matrix)
            for i in range(1, self.num_time_steps+1):
                # Construct linear system
                current_sol = self.time_step(current_sol, time, sample)
                time += self.time_step_size

                # Store history if requested
                if i % sol_storage_stride == 0:
                    sols[:, sol_cntr] = current_sol
                    self.times[sol_cntr] = time
                    sol_cntr += 1
            assert sol_cntr == self.num_stored_timesteps
            return sols
        else:
            current_forcing = self.forcing_function(
                self.mesh_pts, time, sample)
            matrix, rhs = self.apply_boundary_conditions(
                self.collocation_matrix.copy(), current_forcing)
            return np.linalg.solve(matrix, rhs)

    def solve(self, diffusivity, forcing):
        assert diffusivity.ndim == 1
        assert forcing.ndim == 1
        # forcing will be overwritten with bounary values so must take a
        # deep copy
        forcing = forcing.copy()
        # we need another copy so that forcing can be used when solving adjoint
        self.forcing_vals = forcing.copy()
        assert not np.any(diffusivity <= 0.)
        self.collocation_matrix = self.form_collocation_matrix(
            self.derivative_matrix, diffusivity)
        matrix, forcing = self.apply_boundary_conditions(
            self.collocation_matrix.copy(), forcing)
        solution = np.linalg.solve(matrix, forcing)
        # store solution for use with adjoints
        self.fwd_solution = solution.copy()
        return solution

    def run(self, sample):
        assert sample.ndim == 1
        diffusivity = self.diffusivity_function(self.mesh_pts, sample)
        forcing = self.forcing_function(self.mesh_pts, sample)
        solution = self.solve(diffusivity, forcing)
        return solution

    def solve_adjoint(self, sample, order):
        """
        Typically with FEM we solve Ax=b and the discrete adjoint equation
        is A'y=z. But with collocation this does not work. Instead of
        taking the adjoint of the discrete system as the aforemntioned
        approach does. We discretize continuous adjoint equation. Which for
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
                self.adjoint_mesh_pts = self.scale_canonical_pts(
                    adjoint_mesh_pts)
                # scale derivative matrix from [-1,1] to [a,b]
                self.adjoint_derivative_matrix *= 2. / \
                    (self.xlim[1]-self.xlim[0])

            diffusivity = self.diffusivity_function(
                self.adjoint_mesh_pts, sample)
            matrix = self.form_collocation_matrix(
                self.adjoint_derivative_matrix, diffusivity)
            self.adjoint_collocation_matrix = matrix.copy()

        # regardless of whether computing error estimate or
        # computing gradient, rhs is always derivative (with respect to the
        # solution) of the qoi_functional
        qoi_deriv = self.qoi_functional_deriv(self.fwd_solution)

        matrix = self.apply_boundary_conditions_to_matrix(matrix)
        qoi_deriv = self.apply_adjoint_boundary_conditions_to_rhs(qoi_deriv)
        adj_solution = np.linalg.solve(matrix, qoi_deriv)
        return adj_solution

    def apply_adjoint_boundary_conditions_to_rhs(self, qoi_deriv):
        # adjoint always has zero Dirichlet BC
        qoi_deriv[0] = 0
        qoi_deriv[-1] = 0
        return qoi_deriv

    def compute_residual(self, matrix, solution, forcing):
        matrix, forcing = self.apply_boundary_conditions(matrix, forcing)
        return forcing - np.dot(matrix, solution)

    def compute_residual_derivative(self, solution, diagonal,
                                    forcing_deriv):
        matrix = self.form_collocation_matrix(self.derivative_matrix,
                                              diagonal)
        # Todo: check if boundary conditions need to be applied to both
        # matrix and forcing_derivs or just matrix. If the former
        # what boundary conditions to I impose on the focing deriv
        matrix = self.apply_boundary_conditions_to_matrix(
            matrix)
        # the values here are the derivative of the boundary conditions
        # with respect to the random parameters. I assume that
        # this is always zero
        forcing_deriv[0] = 0
        forcing_deriv[-1] = 0
        return forcing_deriv.squeeze() - np.dot(matrix, solution)

    def compute_error_estimate(self, sample):
        raise NotImplementedError("Not passing tests")
        # must solve adjoint with a higher order grid
        adj_solution = self.solve_adjoint(sample, self.order*2)

        # interpolate forward solution onto higher-order grid
        interp_fwd_solution = self.interpolate(
            self.fwd_solution, self.adjoint_mesh_pts)

        # compute residual of forward solution using higher-order grid
        forcing_vals = self.forcing_function(self.adjoint_mesh_pts,
                                             sample)

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
        print('cond', np.linalg.cond(self.adjoint_collocation_matrix))
        error_estimate = self.integrate(residual * adj_solution, self.order*2)

        return error_estimate

    def evaluate_gradient(self, sample):
        assert sample.ndim == 1
        num_stoch_dims = sample.shape[0]
        # qoi_deriv = self.qoi_functional_deriv(self.mesh_pts)
        adj_solution = self.solve_adjoint(sample, self.order)
        gradient = np.empty((num_stoch_dims), float)
        for i in range(num_stoch_dims):
            diffusivity_deriv_vals_i = self.diffusivity_derivs_function(
                self.mesh_pts.squeeze(), sample, i)
            forcing_deriv_vals_i = self.forcing_derivs_function(
                self.mesh_pts.squeeze(), sample, i)
            residual_deriv = self.compute_residual_derivative(
                self.fwd_solution, diffusivity_deriv_vals_i,
                forcing_deriv_vals_i)
            gradient[i] = self.integrate(residual_deriv * adj_solution)
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
        # Get Gauss-Legendre rule
        gl_pts, gl_wts = gauss_jacobi_pts_wts_1D(order, 0, 0)
        # Scale points from [-1,1] to to physical domain
        x_range = self.xlim[1]-self.xlim[0]
        gl_pts = x_range*(gl_pts+1.)/2.+self.xlim[0]
        # Remove factor of 0.5 from weights
        gl_wts *= x_range
        # Interpolate mesh values onto quadrature nodes
        gl_vals = self.interpolate(mesh_values, gl_pts)
        # Compute and return integral
        return np.dot(gl_vals[:, 0], gl_wts)

    def interpolate(self, mesh_values, eval_samples):
        if eval_samples.ndim == 1:
            eval_samples = eval_samples[None, :]
        if mesh_values.ndim == 1:
            mesh_values = mesh_values[:, None]
        assert mesh_values.ndim == 2
        num_dims = eval_samples.shape[0]
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
                self.xlim[0], self.xlim[1], num_plot_pts_1d)
            interp_vals = self.interpolate(mesh_values, plot_mesh)
            pylab.plot(plot_mesh, interp_vals, color+'-')

        elif plot_mesh_coords is not None:
            assert mesh_values.shape[0] == plot_mesh_coords.squeeze().shape[0]
            pylab.plot(plot_mesh_coords, mesh_values, 'o-'+color)
        else:
            # just plot values on mesh points
            pylab.plot(self.mesh_pts, mesh_values, color)

    def get_collocation_points(self):
        return np.atleast_2d(self.mesh_pts)

    def get_derivative_matrix(self):
        return self.derivative_matrix

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.value, samples, None)



class SteadyStateDiffusionEquation2D(SteadyStateDiffusionEquation1D):
    """
    solve  (a(x)*u_x)_x = f; x in [0,1]x[0,1];
    subject to u(0,:)=a(x); u(:,0)=b(x), u(1,:)=c(x), u(:,1)=d(x)
    """

    def __init__(self):
        self.diffusivity = None
        self.forcing_function = None
        self.bndry_cond = [0., 0., 0., 0.]
        self.xlim = [0, 1]
        self.ylim = [0, 1]
        self.left_bc, self.right_bc = None, None
        self.top_bc, self.bottom_bc = None, None

        # default qoi functional is integral of solution over entire domain
        self.qoi_functional = self.integrate
        self.qoi_functional_deriv = lambda x: x*0.+1.

    def determine_boundary_indices(self):
        # boundary edges are stored with the following order,
        # left, right, bottom, top
        self.boundary_edges = [[], [], [], []]
        self.boundary_indices = np.empty((4*self.order), int)
        # To avoid double counting the bottom and upper boundaries
        # will not include the edge indices
        cntr = 0
        for i in range(self.mesh_pts.shape[1]):
            if (self.mesh_pts[0, i] == self.xlim[0]):
                self.boundary_indices[cntr] = i
                self.boundary_edges[0].append(cntr)
                cntr += 1
            elif (self.mesh_pts[0, i] == self.xlim[1]):
                self.boundary_indices[cntr] = i
                self.boundary_edges[1].append(cntr)
                cntr += 1
            elif (self.mesh_pts[1, i] == self.ylim[0]):
                self.boundary_indices[cntr] = i
                self.boundary_edges[2].append(cntr)
                cntr += 1
            elif (self.mesh_pts[1, i] == self.ylim[1]):
                self.boundary_indices[cntr] = i
                self.boundary_edges[3].append(cntr)
                cntr += 1

    def initialize(self, order, bndry_cond=None, lims=None):
        # 1d model transforms mesh pts 1d from are on [-1,1] to [a,b]
        # I will asssume that second physical dimension is also [a,b]
        super(SteadyStateDiffusionEquation2D, self).initialize(order,
                                                               bndry_cond[:2],
                                                               lims[:2])
        self.ylim = lims[2:]
        self.bndry_cond = bndry_cond
        self.order = order
        self.mesh_pts_1d = self.mesh_pts
        self.mesh_pts = cartesian_product([self.mesh_pts_1d]*2, 1)

        # note scaling of self.derivative_matrix to [a,b] happens at base class
        self.determine_boundary_indices()
        # form derivative (in x1-direction) matrix of a 2d polynomial
        # this assumes that 2d-mesh_pts varies in x1 faster than x2,
        # e.g. points are
        # [[x11,x21],[x12,x21],[x13,x12],[x11,x22],[x12,x22],...]
        Ident = np.eye(self.order+1)
        derivative_matrix_1d = self.get_derivative_matrix()
        self.derivative_matrix_1 = np.kron(Ident, derivative_matrix_1d)
        # form derivative (in x2-direction) matrix of a 2d polynomial
        self.derivative_matrix_2 = np.kron(derivative_matrix_1d, Ident)

    def form_collocation_matrix(self, derivative_matrix, diagonal):
        scaled_matrix_1 = np.empty(self.derivative_matrix_1.shape)
        scaled_matrix_2 = np.empty(self.derivative_matrix_2.shape)
        for i in range(scaled_matrix_1.shape[0]):
            scaled_matrix_1[i, :] = self.derivative_matrix_1[i, :]*diagonal[i]
            scaled_matrix_2[i, :] = self.derivative_matrix_2[i, :]*diagonal[i]
        matrix_1 = np.dot(self.derivative_matrix_1, scaled_matrix_1)
        matrix_2 = np.dot(self.derivative_matrix_2, scaled_matrix_2)
        return matrix_1 + matrix_2

    def apply_boundary_conditions_to_matrix(self, matrix):
        # apply default homogeenous zero value direchlet conditions if
        # necessary
        if self.left_bc is None:
            self.left_bc = lambda x: 0.
        if self.right_bc is None:
            self.right_bc = lambda x: 0.
        if self.bottom_bc is None:
            self.bottom_bc = lambda x: 0.
        if self.top_bc is None:
            self.top_bc = lambda x: 0.

        # adjust collocation matrix
        matrix[self.boundary_indices, :] = 0.
        for i in range(self.boundary_indices.shape[0]):
            index = self.boundary_indices[i]
            matrix[index, index] = 1.
        return matrix

    def apply_boundary_conditions_to_rhs(self, forcing):
        # apply left boundary condition
        indices = self.boundary_indices[self.boundary_edges[0]]
        forcing[indices] = self.left_bc(self.mesh_pts[0, indices])
        # apply right boundary condition
        indices = self.boundary_indices[self.boundary_edges[1]]
        forcing[indices] = self.right_bc(self.mesh_pts[0, indices])
        # apply bottom boundary condition
        indices = self.boundary_indices[self.boundary_edges[2]]
        forcing[indices] = self.bottom_bc(self.mesh_pts[1, indices])
        # apply top boundary condition
        indices = self.boundary_indices[self.boundary_edges[3]]
        forcing[indices] = self.top_bc(self.mesh_pts[1, indices])
        return forcing

    def plot(self, mesh_values, num_plot_pts_1d=100):
        if num_plot_pts_1d is not None:
            # interpolate values onto plot points
            def func(x): return self.interpolate(mesh_values, x)
            from utilities.visualisation import plot_surface_from_function
            plot_surface_from_function(func, [self.xlim[0], self.xlim[1],
                                              self.ylim[0], self.ylim[1]],
                                       num_plot_pts_1d, False)

    def apply_adjoint_boundary_conditions_to_rhs(self, qoi_deriv):
        # adjoint always has zero Dirichlet BC
        # apply left boundary condition
        for ii in range(4):
            indices = self.boundary_indices[self.boundary_edges[ii]]
            qoi_deriv[indices] = 0
        return qoi_deriv

    def integrate(self, mesh_values, order=None):
        if order is None:
            order = self.order
        # Get Gauss-Legendre rule
        gl_pts, gl_wts = gauss_jacobi_pts_wts_1D(order, 0, 0)
        pts_1d, wts_1d = [], []
        lims = self.xlim+self.ylim
        for ii in range(2):
            # Scale points from [-1,1] to to physical domain
            x_range = lims[2*ii+1]-lims[2*ii]
            # Remove factor of 0.5 from weights and shift to [a,b]
            wts_1d.append(gl_wts*x_range)
            pts_1d.append(x_range*(gl_pts+1.)/2.+lims[2*ii])
        # Interpolate mesh values onto quadrature nodes
        pts = cartesian_product(pts_1d)
        wts = outer_product(wts_1d)
        gl_vals = self.interpolate(mesh_values, pts)
        # Compute and return integral
        return np.dot(gl_vals[:, 0], wts)
