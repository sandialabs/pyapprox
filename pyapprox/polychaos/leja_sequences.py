from scipy.optimize import fmin_l_bfgs_b
import numpy as np


def get_quadrature_weights_from_samples(compute_basis_matrix, indices, samples):
    """
    Get the quadrature weights from a set of samples. The number os samples
    must equal the number of terms in the polynomial basis.

    Parameters
    ----------
    compute_basis_matrix : callable compute_basis_matrix(samples,indices)
        Function used to construct the basis matrix eavluate at samples.

    indices : np.ndarray (num_vars, num_indices)
        The mutivariate indices definining the polynomial basis

    samples : np.ndarray (num_vars, num_samples)
        The samples for which quadrature weights are desired.

    Return
    ------
    weights : np.ndarray (num_samples)
        The quadrature weights.
    """
    assert samples.shape[1] == indices.shape[1]
    basis_matrix = compute_basis_matrix(indices, samples)
    basis_matrix_inv = np.linalg.inv(basis_matrix)
    weights = basis_matrix_inv[0, :]
    return weights


def leja_objective_and_gradient(samples, leja_sequence, poly, new_indices,
                                coeff, weight_function, weight_function_deriv,
                                deriv_order=0):
    """
    Evaluate the Leja objective at a set of samples.



    Parameters
    ----------
    samples : np.ndarray (num_vars, num_samples)
        The sample at which to evaluate the leja_objective

    leja_sequence : np.ndarray (num_vars, num_leja_samples)
        The sample already in the Leja sequence

    deriv_order : integer
        Flag specifiying whether to compute gradients of the objective

    new_indices : np.ndarray (num_vars, num_new_indices)
        The new indices that are considered when choosing next sample
        in the Leja sequence

    coeff : np.ndarray (num_indices, num_new_indices)
        The coefficient of the approximation that interpolates the polynomial
        terms specified by new_indices

    Return
    ------
    residuals : np.ndarray(num_new_indices,num_samples):

    objective_vals : np.ndarray (num_samples)
        The values of the objective at samples

    objective_grads : np.ndarray (num_vars,num_samples)
        The gradient of the objective at samples. Return only
        if deriv_order==1
    """
    assert samples.ndim == 2
    num_vars, num_samples = samples.shape
    assert num_samples == 1

    indices = poly.indices.copy()
    poly.set_indices(new_indices)
    basis_matrix_for_new_indices_at_samples = poly.basis_matrix(
        samples, {'deriv_order': deriv_order})
    if deriv_order == 1:
        basis_deriv_matrix_for_new_indices_at_samples = \
            basis_matrix_for_new_indices_at_samples[1:, :]
    basis_matrix_for_new_indices_at_samples = \
        basis_matrix_for_new_indices_at_samples[:1, :]
    poly.set_indices(indices)

    basis_matrix_at_samples = poly.basis_matrix(
        samples[:, :1], {'deriv_order': deriv_order})
    if deriv_order == 1:
        basis_deriv_matrix_at_samples = basis_matrix_at_samples[1:, :]
    basis_matrix_at_samples = basis_matrix_at_samples[:1, :]

    weights = weight_function(samples)
    # to avoid division by zero
    weights = np.maximum(weights, 0)
    assert weights.ndim == 1
    sqrt_weights = np.sqrt(weights)

    poly_vals = np.dot(basis_matrix_at_samples, coeff)

    unweighted_residual = basis_matrix_for_new_indices_at_samples-poly_vals
    residual = sqrt_weights*unweighted_residual

    num_residual_entries = residual.shape[1]

    if deriv_order == 0:
        return (residual,)

    poly_derivs = np.dot(basis_deriv_matrix_at_samples, coeff)
    weight_derivs = weight_function_deriv(samples)

    unweighted_residual_derivs = \
        poly_derivs-basis_deriv_matrix_for_new_indices_at_samples

    jacobian = np.zeros((num_residual_entries, num_vars), dtype=float)
    I = np.where(weights > 0)[0]
    for dd in range(num_vars):
        jacobian[I, dd] = (
            unweighted_residual[0, I]*weight_derivs[dd, I]/(2.0*sqrt_weights[I]) -
            unweighted_residual_derivs[dd, I]*sqrt_weights[I])
    assert residual.ndim == 2
    return residual, jacobian


def compute_coefficients_of_leja_interpolant(leja_sequence, poly, new_indices,
                                             weight_function):

    weights = weight_function(leja_sequence)
    # to avoid division by zero
    weights = np.maximum(weights, 0)
    assert weights.ndim == 1
    sqrt_weights = np.sqrt(weights)

    indices = poly.indices.copy()
    poly.set_indices(new_indices)
    #basis_matrix_for_new_indices_at_leja = poly.basis_matrix(leja_sequence)
    basis_matrix_for_new_indices_at_leja = (
        poly.basis_matrix(leja_sequence).T*sqrt_weights).T
    poly.set_indices(indices)

    # replace with more efficient procedure that just updates LU and
    # uses backwards subtitution
    #basis_matrix_at_leja = poly.basis_matrix(leja_sequence)
    basis_matrix_at_leja = (poly.basis_matrix(leja_sequence).T*sqrt_weights).T
    out = np.linalg.lstsq(
        basis_matrix_at_leja, basis_matrix_for_new_indices_at_leja, rcond=None)
    coeffs = out[0]
    return coeffs


def leja_objective(samples, leja_sequence, poly, new_indices, coeff,
                   weight_function, weight_function_deriv):
    objective_vals = np.empty((samples.shape[1]), dtype=float)
    for ii in range(samples.shape[1]):
        residual = leja_objective_and_gradient(
            samples[:, ii:ii+1], leja_sequence, poly, new_indices, coeff,
            weight_function, weight_function_deriv)[0]
        objective_vals[ii] = 0.5*np.dot(residual.squeeze(), residual.squeeze())
    return objective_vals


def compute_finite_difference_derivative(func, sample, fd_eps=1e-6):
    assert sample.ndim == 2
    num_vars = sample.shape[0]
    fd_samples = np.empty((num_vars, num_vars+1))
    fd_samples[:, 0] = sample[:, 0].copy()

    for dd in range(num_vars):
        fd_samples[:, dd+1] = sample[:, 0].copy()
        fd_samples[dd, dd+1] += fd_eps

    objective_at_fd_samples = func(fd_samples)

    fd_deriv = np.empty((num_vars, 1))
    for dd in range(num_vars):
        fd_deriv[dd] =\
            (objective_at_fd_samples[dd+1]-objective_at_fd_samples[0])/(fd_eps)
    return fd_deriv


class LejaObjective():
    def __init__(self, poly, weight_function, weight_function_deriv):
        self.poly = poly
        self.unscaled_weight_function = weight_function
        self.unscaled_weight_function_deriv = weight_function_deriv
        self.set_scale(1)

    def set_scale(self, scale):
        """
        scale objective function by a scalar. This can make finding values of
        small objectives easier.
        """
        assert scale > 0
        self.scale = max(scale, 1e-8)
        self.weight_function = \
            lambda x: self.unscaled_weight_function(x)/self.scale
        self.weight_function_deriv = \
            lambda x: self.unscaled_weight_function_deriv(x)/self.scale

    def precompute_residual_and_jacobian(self, sample, leja_sequence,
                                         new_indices, coeffs):
        self.sample = sample
        if sample.ndim == 1:
            sample = sample[:, np.newaxis]
        self.residual, self.jacobian = leja_objective_and_gradient(
            sample, leja_sequence, self.poly, new_indices, coeffs,
            self.weight_function, self.weight_function_deriv, deriv_order=1)

    def gradient(self, sample, leja_sequence, new_indices, coeffs):
        assert np.allclose(sample, self.sample)
        if sample.ndim == 1:
            sample = sample[:, np.newaxis]
        # from functools import partial
        # func = partial(leja_objective,leja_sequence=leja_sequence,
        #                poly=self.poly,
        #                new_indices=new_indices, coeff=coeffs,
        #                weight_function=self.weight_function,
        #                weight_function_deriv=self.weight_function_deriv)
        # fd_eps=1e-7
        # fd_deriv = -compute_finite_difference_derivative(
        #     func,sample,fd_eps=fd_eps)
        gradient = -np.dot(self.jacobian.T, self.residual)
        # print('gk',sample,gradient)
        return gradient

    def __call__(self, sample, leja_sequence, new_indices, coeffs):
        self.precompute_residual_and_jacobian(
            sample, leja_sequence, new_indices, coeffs)
        val = -0.5*np.dot(self.residual, self.residual)
        # print('val',sample,val)
        return val

    def jacobian(self, sample, leja_sequence, new_indices, coeffs):
        assert np.allclose(sample, self.sample)
        return self.jacobian

    def plot(self, leja_sequence, poly, new_indices, coeffs, ranges):
        import matplotlib.pyplot as plt
        if leja_sequence.shape[0] == 1:
            num_samples = 400
            samples = np.linspace(
                ranges[0], ranges[1], num_samples).reshape(1, num_samples)
            objective_vals = -leja_objective(
                samples, leja_sequence, poly, new_indices, coeffs,
                self.weight_function, self.weight_function_deriv)
            # print(self.weight_function(samples))
            # unweighted_objective_vals = -leja_objective(
            #     samples, leja_sequence, poly, new_indices, coeffs,
            #     lambda x: np.ones(x.shape[1]), lambda x: np.zeros(x.shape[1]))
            # print(unweighted_objective_vals)
            #objective_vals = np.array([self(samples[:,ii],leja_sequence,new_indices,coeffs) for ii in range(num_samples)]).squeeze()
            plt.plot(samples[0, :], objective_vals, lw=3)
            plt.plot(leja_sequence[0, :], leja_sequence[0, :]*0.0, 'o',
                     label='Leja sequence')
            # plt.ylim(-1,1)


def optimize(obj, initial_guess, ranges, objective_args, scale):
    bounds = []
    for i in range(len(ranges)//2):
        bounds.append([ranges[2*i], ranges[2*i+1]])
    obj.set_scale(scale)
    tol = 1e-8
    callback = None
    out = fmin_l_bfgs_b(
        func=obj, x0=initial_guess, bounds=bounds,
        args=objective_args,
        factr=tol/np.finfo(float).eps, pgtol=tol, maxiter=1000,
        iprint=0,
        # approx_grad=True)
        fprime=obj.gradient, callback=callback)
    optimal_sample = out[0]
    obj_val = out[1]*obj.scale
    num_fn_evals = out[2]['funcalls']
    #print ('initial_guess',initial_guess)
    #print ('\tFunction evaluations: %d'%(num_fn_evals))
    #print (optimal_sample)

    obj.set_scale(1)
    return optimal_sample, obj_val


def get_initial_guesses_1d(leja_sequence, ranges):
    eps = 1e-6  # must be larger than optimization tolerance
    intervals = np.sort(leja_sequence)
    if ranges[0] != None and (leja_sequence.min() > ranges[0]+eps):
        intervals = np.hstack(([[ranges[0]]], intervals))
    if ranges[1] != None and (leja_sequence.max() < ranges[1]-eps):
        intervals = np.hstack((intervals, [[ranges[1]]]))

    if ranges[0] is None:
        intervals = np.hstack((
            [[min(1.1*leja_sequence.min(), -0.1)]], intervals))
    if ranges[1] is None:
        intervals = np.hstack((
            intervals, [[max(1.1*leja_sequence.max(), 0.1)]]))

    diff = np.diff(intervals)
    initial_guesses = intervals[:, :-1]+np.diff(intervals)/2.0

    # put intervals in form useful for bounding 1d optimization problems
    intervals = [intervals[0, ii] for ii in range(intervals.shape[1])]
    if ranges[0] is None:
        intervals[0] = None
    if ranges[1] is None:
        intervals[-1] = None

    return initial_guesses, intervals


def get_leja_sequence_1d(num_leja_samples, initial_points, poly,
                         weight_function, weight_function_deriv, ranges,
                         plot=False):
    num_vars = initial_points.shape[0]
    leja_sequence = initial_points.copy()
    indices = np.arange(leja_sequence.shape[1])[np.newaxis, :]
    new_indices = np.asarray([[leja_sequence.shape[1]]]).T

    ii = leja_sequence.shape[1]
    while ii < num_leja_samples:
        poly.set_indices(indices)
        coeffs = compute_coefficients_of_leja_interpolant(
            leja_sequence, poly, new_indices, weight_function)

        obj = LejaObjective(poly, weight_function, weight_function_deriv)
        objective_args = (leja_sequence, new_indices, coeffs)
        initial_guesses, intervals = get_initial_guesses_1d(
            leja_sequence, ranges)
        obj_vals = np.empty((initial_guesses.shape[1]))
        new_samples = np.empty((num_vars, initial_guesses.shape[1]))
        # scales = np.array(
        #    [abs(np.asscalar(obj(initial_guesses[:,jj],*objective_args))) for jj in range(initial_guesses.shape[1])])
        #scale = scales.max()
        scale = 1
        for jj in range(initial_guesses.shape[1]):
            initial_guess = initial_guesses[:, jj]
            sub_ranges = [intervals[jj], intervals[jj+1]]
            new_samples[:, jj], obj_vals[jj] = optimize(
                obj, initial_guess, sub_ranges, objective_args, scale)

            # if ii==17:
            #     import matplotlib.pyplot as plt
            #     #plot_ranges=[min(leja_sequence.min(),new_samples.min()),
            #     #             max(leja_sequence.max(),new_samples.max())]
            #     plot_ranges=[leja_sequence.min(),
            #                  leja_sequence.max()]
            #     plot_ranges[0] = plot_ranges[0]-abs(plot_ranges[0])*1
            #     plot_ranges[1] = plot_ranges[1]+abs(plot_ranges[1])*1
            #     plot_ranges=[-8.5,8.5]
            #     obj.plot(leja_sequence,poly,new_indices,coeffs,plot_ranges)
            #     if num_vars==1:
            #         print(new_samples[:,jj],obj_vals[jj])
            #         initial_obj_val = obj(initial_guess,leja_sequence,new_indices,coeffs)
            #         plt.plot(initial_guess[0],initial_obj_val,'s',
            #                 label='init guess')
            #         plt.plot(new_samples[:,jj],obj_vals[jj],'s',label='local minima')
            #         plt.title(r'$N=%d$'%leja_sequence.shape[1])
            #         plt.legend()
            #         plt.show()

        I = np.argmin(obj_vals)
        new_sample = new_samples[:, I]

        # if (plot and ii == num_leja_samples-1):
        #     import matplotlib.pyplot as plt
        #     #plot_ranges=[min(leja_sequence.min(),new_samples.min()),
        #     #             max(leja_sequence.max(),new_samples.max())]
        #     plot_ranges=[leja_sequence.min(),
        #                  leja_sequence.max()]
        #     plot_ranges[0] = plot_ranges[0]-abs(plot_ranges[0])*1
        #     plot_ranges[1] = plot_ranges[1]+abs(plot_ranges[1])*1
        #     plot_ranges=[-6,6]
        #     obj.plot(leja_sequence,poly,new_indices,coeffs,plot_ranges)
        #     if num_vars==1:
        #         plt.plot(new_sample[0],obj_vals[I],'o',label='new sample',ms=10)
        #         initial_obj_vals = np.array([obj(initial_guesses[:,ii],leja_sequence,new_indices,coeffs) for ii in range(initial_guesses.shape[1])]).squeeze()
        #         plt.plot(initial_guesses[0,:],initial_obj_vals,'s',
        #                 label='init guesses')
        #         plt.plot(new_samples[0,:],obj_vals,'s',label='local minima')
        #         plt.title(r'$N=%d$'%leja_sequence.shape[1])
        #         plt.legend()
        #         plt.show()

        leja_sequence = np.hstack((leja_sequence, new_sample[:, np.newaxis]))
        indices = np.hstack((indices, new_indices))  # only works in 1D
        # in nd only increment when number of points equal to number of
        # cardinaltiy of basis of current degree
        new_indices = np.asarray([indices[:, -1]+1]).T

        ii += 1

    return leja_sequence
