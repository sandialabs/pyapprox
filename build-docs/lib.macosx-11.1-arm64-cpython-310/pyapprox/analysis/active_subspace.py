import os
import numpy as np
from scipy.optimize import linprog as scipy_linprog
from scipy.spatial import ConvexHull
from scipy import stats

from pyapprox.variables.transforms import map_hypercube_samples
from pyapprox.surrogates.interp.manipulate_polynomials import (
    multiply_multivariate_polynomials,
    coeffs_of_power_of_nd_linear_monomial,
)
from pyapprox.util.utilities import cartesian_product
from pyapprox.util.visualization import (
    plt, plot_2d_polygon, get_meshgrid_function_data
)


def get_random_active_subspace_eigenvecs(
        num_vars, num_active_vars, filename=None):
    """
    Eigenvectors will be zero for all inactive variables
    """
    if filename is None or not os.path.exists(filename):
        A = np.random.normal(0, 1, (num_vars, num_vars))
        W, R = np.linalg.qr(A)
        W1 = W[:, :num_active_vars]
        W2 = W[:, num_active_vars:]*0.
        if filename is not None:
            np.savez(filename, W1=W1, W2=W2, W=W)
    else:
        W1 = np.load(filename)['W1']
        W2 = np.load(filename)['W2']
        W = np.load(filename)['W']
    return W, W1, W2


def scipy_solve_linear_program_ineq(c, A, b):

    c = c.reshape((c.size,))
    b = b.reshape((b.size,))

    # make unbounded bounds
    bounds = []
    for i in range(c.size):
        bounds.append((None, None))

    A_ub, b_ub = -A, -b
    res = scipy_linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                        options={"disp": False})
    if res.success:
        return res.x.reshape((c.size, 1))
    else:
        raise Exception('Scipy failed to solve the linear program')


def get_chebyshev_center_of_inactive_subspace(W1, W2, active_sample):
    """
    Find the chebhsev sample of the inactive variables z so that
    -1 <= W1*y + W2*z <= 1,
    where y is the given value of the active variables. In other words, we need
    to sample z such that it respects the linear equalities
    W2*z <= 1 - W1*y, -W2*z <= 1 + W1*y.
    """
    num_vars = W1.shape[0]
    num_active_vars = W1.shape[1]
    assert active_sample.ndim == 1
    assert active_sample.shape[0] == num_active_vars

    s = np.dot(W1, active_sample).reshape((num_vars, 1))
    normW2 = np.sqrt(
        np.sum(np.power(W2, 2), axis=1)).reshape((num_vars, 1))
    A = np.hstack(
        (np.vstack((W2, -W2.copy())),
         np.vstack((normW2, normW2.copy()))))
    b = np.vstack((1-s, 1+s)).reshape((2*num_vars, 1))
    c = np.zeros((num_vars-num_active_vars+1, 1))
    c[-1] = -1.0

    zc = scipy_solve_linear_program_ineq(c, -A, -b)
    z0 = zc[:-1].reshape((num_vars-num_active_vars, 1))
    return z0


def transform_active_subspace_samples_to_original_coordinates(
        active_samples, W):
    num_active_vars = active_samples.shape[0]
    num_active_samples = active_samples.shape[1]
    num_vars = W.shape[0]
    W1 = W[:, :num_active_vars]
    W2 = W[:, num_active_vars:]
    samples = np.empty((num_vars, num_active_samples), float)
    for i in range(num_active_samples):
        # Get a sample in the active subspace zonotope
        active_sample = active_samples[:, i]

        # get a sample (chebyshev center) in the inactive
        # subspace zonotope
        inactive_sample = get_chebyshev_center_of_inactive_subspace(
            W1, W2, active_sample)

        # get coordinates of the active and inactive samples
        # in the original space
        rotated_sample = np.hstack(
            (active_sample.squeeze(), inactive_sample.squeeze()))
        samples[:, i] = np.dot(W, rotated_sample)
    return samples


def get_zonotope_vertices_and_bounds(active_subspace_eigvecs):
    """
    active_subspace_eigvecs : np.ndarray (num_vars,num_active_vars)
    """
    import active_subspaces as asub
    # requires
    # PYTHONPATH=$PYTHONPATH:~/tpl/active_subspaces/
    Y, X = asub.domains.zonotope_vertices(
        active_subspace_eigvecs)
    zonotope_max = Y.max(axis=0)
    zonotope_min = Y.min(axis=0)
    assert zonotope_min.shape[0] == active_subspace_eigvecs.shape[1]
    return Y.T, zonotope_min, zonotope_max


def get_uniform_samples_on_zonotope(num_samples, zonotope_vertices):
    return get_uniform_samples_on_polygon(num_samples, zonotope_vertices)


def get_uniform_samples_on_polygon(num_samples, vertices, batch_size=100):
    num_vars = vertices.shape[0]
    ranges = np.empty((2*num_vars), float)
    ranges[::2] = vertices.min(axis=1)
    ranges[1::2] = vertices.max(axis=1)

    convhull = ConvexHull(vertices.T)
    Aeq = convhull.equations[:, :num_vars]
    beq = convhull.equations[:, num_vars]
    def constraints(x): return np.dot(Aeq, x) + beq

    num_accepted_samples = 0
    current_ranges = np.ones((2*num_vars), float)
    current_ranges[::2] = 0.

    accepted_samples = np.empty((num_vars, num_samples), float)
    while (num_accepted_samples < num_samples):
        batch_samples = np.random.uniform(0., 1., (num_vars, batch_size))
        batch_samples = map_hypercube_samples(
            batch_samples, current_ranges, ranges)
        # constraints = np.dot(Aeq, batch_samples)+beq[:, np.newaxis]
        accepted_idx = np.arange(batch_size)[np.all(constraints <= 0, axis=0)]
        num_batch_samples_accepted = min(
            num_samples-num_accepted_samples, accepted_idx.shape[0])
        accepted_samples[:, num_accepted_samples:num_accepted_samples +
                         num_batch_samples_accepted] = batch_samples[:, accepted_idx[:num_batch_samples_accepted]]

        num_accepted_samples += num_batch_samples_accepted

    assert accepted_samples.shape[1] == num_samples
    # from pyapprox.util.visualization import plot_2d_polygon
    # plot_2d_polygon(vertices)
    # import matplotlib.pyplot as pl
    # plt.plot(accepted_samples[0,:],accepted_samples[1,:],'ro')
    # plt.show()
    return accepted_samples


def coeffs_of_active_subspace_polynomial(W1_trans, as_index):
    """
    Evaluate the coefficients of a polynomial in an
    active subspace in terms of a polynomial in the original variable space.

    In active subspace with a monomial basis we may have a polynomial with
    terms
        [1,y1,y2,y1**2,y1*y2,y2**2,y1**3,y1**2*y2,y1*y2**2,y2**3]
    We want to compute a polynomial in terms of the original variables x. E.g.
       y1**2*y2 = W_11*x1+W_12*x2+W_13*x3)**2*(W_21*x1+W_22*x2+W_23*x3)
    where the native function space (x) has dimension num_vars=3


    Parameters
    ----------
    W1_trans : np.ndarray (num_active_vars, num_vars)
        The active subspace active variable transformation such that
        the active variable y is obtained from the full dimension variable
        x via y = W1_trans*x

    as_index : np.ndarray (num_active_vars)
        The multivariate index of the polynomial in terms of the active
        subspace variables y

    Returns
    -------
    coeff : np.ndarray (num_moments)
        The coefficients of the polynomial in terms of the original variables x

    indices : np.ndarray (num_terms)
        The set of multivariate indices that define equivalent polynomial
        (to the active subapce polynomial) in terms of the original variables x
    """
    num_vars = W1_trans.shape[1]
    activated_vars = np.where(as_index > 0)[0]
    num_activated_vars = activated_vars.shape[0]
    if num_activated_vars > 0:
        var_num = activated_vars[0]
        degree = as_index[var_num]
    else:
        degree = 0
        var_num = 0
    coeffs, indices = coeffs_of_power_of_nd_linear_monomial(
        num_vars, degree, W1_trans[var_num, :])
    for dd in range(1, num_activated_vars):
        var_num = activated_vars[dd]
        degree = as_index[var_num]
        coeffs_d, indices_d = coeffs_of_power_of_nd_linear_monomial(
            num_vars, degree, W1_trans[var_num, :])
        indices, coeffs = multiply_multivariate_polynomials(
            indices, coeffs[:, None], indices_d, coeffs_d[:, None])
    return coeffs, indices


def moments_of_active_subspace(
        W1_trans, as_poly_indices, integrate_polynomial):
    """Evaluate the moments of a total-degree polynomial in an
    active subspace.

    In active subspace with a monomial basis we may have a polynomial with
    terms
        [1,y1,y2,y1**2,y1*y2,y2**2,y1**3,y1**2*y2,y1*y2**2,y2**3]
    We want moments in active subspace so we need to compute
    integrals like
       int_{as_domain} y1**2*y2 dp(y) =
       int_{as_domain} (W_11*x1+W_12*x2+W_13*x3)**2*(W_21*x1+W_22*x2+W_23*x3)dp(x)
    where in this example the native function space (x) has dimension num_vars=3

    Parameters
    ----------
    W1_trans : np.ndarray (num_active_vars,num_vars)
        The active subspace active variable transformation such that
        the active variable y is obtained from the full dimension variable
        x via y = W1_trans*x

    as_poly_indices :  np.ndarray (num_active_vars,num_active_poly_indices)
        The degree of the total-degree polynomial defined over the
        active subspace (dimension=num_active_vars)

    integrate_polynomial : callable function f(indices,coeffs):
        Compute the integral of a polynomial in the native model parameter
        space (dimension=num_vars). The function must have the signature
        ``val = integrate_polynomial(indices,coeffs)``
        where index index is a np.ndarray (num_vars,num_indices) of
        multivariate indices and coeffs are the coefficients associated
        with each index

    Returns
    -------
    moments : np.ndarray (num_moments)
        The moments of each polynomial term in the active subspace

    Comments
    --------
    NOTE: only works when computing moments of monomials. Will not
    currently work when computing moments using orthogonal polynomials.
    This is because of reliance on function coeffs_of_active_subspace_polynomial

    E.g. When not using monomials y1**2 can not be expanded as
    (W_11*x1+W_12*x2+W_13*x3)**2 instead if y1 = 0.5*(3*x^2-1)
    then we must expand as
    0.5*(3*(W_11*x1+W_12*x2+W_13*x3)^2-1)
    """

    num_active_vars, num_vars = W1_trans.shape
    num_as_moments = as_poly_indices.shape[1]
    moments = np.zeros((num_as_moments), float)
    # for ii in range(as_poly_indices.shape[1]):
    #     as_index = as_poly_indices[:,ii]
    #     coeffs, indices = coeffs_of_active_subspace_polynomial(
    #         W1_trans,as_index)
    #     moments[ii] = integrate_polynomial(indices,coeffs)

    # the following is faster because it precomputes
    # coeffs_of_power_of_nd_linear_monomial for each variable dimension
    indices_list, coeffs_list = coeffs_of_active_subspace_polynomials(
        W1_trans, as_poly_indices)
    for ii in range(as_poly_indices.shape[1]):
        moments[ii] = integrate_polynomial(
            indices_list[ii], coeffs_list[ii][:, None])

    return moments


def coeffs_of_active_subspace_polynomials(W1_trans, as_poly_indices):
    num_active_vars, num_vars = W1_trans.shape

    monomial_power_indices = [[] for ii in range(num_active_vars)]
    monomial_power_coeffs = [[] for ii in range(num_active_vars)]
    for var_num in range(num_active_vars):
        for degree in range(as_poly_indices[var_num, :].max()+1):
            coeffs, indices = coeffs_of_power_of_nd_linear_monomial(
                num_vars, degree, W1_trans[var_num, :])
            monomial_power_coeffs[var_num].append(coeffs[:, None].copy())
            monomial_power_indices[var_num].append(indices)
    indices_list = []
    coeffs_list = []
    for ii in range(as_poly_indices.shape[1]):
        as_index = as_poly_indices[:, ii]
        activated_vars = np.where(as_index > 0)[0]
        num_activated_vars = activated_vars.shape[0]
        if num_activated_vars > 0:
            var_num = activated_vars[0]
            degree = as_index[var_num]
        else:
            degree = 0
            var_num = 0
        coeffs = monomial_power_coeffs[var_num][degree]
        indices = monomial_power_indices[var_num][degree]
        for dd in range(1, num_activated_vars):
            var_num = activated_vars[dd]
            degree = as_index[var_num]
            coeffs_dd = monomial_power_coeffs[var_num][degree]
            indices_dd = monomial_power_indices[var_num][degree]

            # # TODO Extension does not group like terms,
            # # but pure python function
            # # multiply_multivariate_polynomials does.
            # # This is only a valid substitution if this function is
            # # only called by intergrate moments
            # try:
            #     from pyapprox.cython.manipulate_polynomials import \
            #         multiply_multivariate_polynomials_pyx
            #     indices, coeffs = multiply_multivariate_polynomials_pyx(
            #         indices, coeffs, indices_dd, coeffs_dd)
            # except (ImportError, TypeError) as e:
            #     from pyapprox.util.sys_utilities import trace_error_with_msg
            #     trace_error_with_msg('multiply_multivariate_polynomials extension failed', e)

            #     print(coeffs.shape, coeffs_dd.shape)
            indices, coeffs = multiply_multivariate_polynomials(
                indices, coeffs, indices_dd, coeffs_dd)

        indices_list.append(indices)
        coeffs_list.append(coeffs[:, 0])

    return indices_list, coeffs_list


def inner_products_on_active_subspace(W1_trans, as_poly_indices,
                                      integrate_polynomial):
    num_active_vars, num_vars = W1_trans.shape
    num_indices = as_poly_indices.shape[1]

    # This actually produces more multiply_multivariate_polynomials
    # than implementation below
    # indices_list,coeffs_list = coeffs_of_active_subspace_polynomials(
    #     W1_trans,as_poly_indices)
    # inner_products = np.zeros((num_indices,num_indices),float)
    # for ii in range(num_indices):
    #     for jj in range(ii,num_indices):
    #         indices,coeffs = multiply_multivariate_polynomials(
    #             indices_list[ii],coeffs_list[ii],
    #             indices_list[jj],coeffs_list[jj])
    #         inner_products[ii,jj] = integrate_polynomial(indices,coeffs)
    #         inner_products[jj,ii] = inner_products[ii,jj]

    doubled_as_poly_indices = []
    for ii in range(num_indices):
        for jj in range(ii, num_indices):
            doubled_as_poly_indices.append(
                as_poly_indices[:, ii]+as_poly_indices[:, jj])

    doubled_as_poly_indices = np.asarray(doubled_as_poly_indices).T

    indices_list, coeffs_list = coeffs_of_active_subspace_polynomials(
        W1_trans, doubled_as_poly_indices)

    kk = 0
    inner_products = np.zeros((num_indices, num_indices), float)
    for ii in range(num_indices):
        for jj in range(ii, num_indices):
            inner_products[ii, jj] = integrate_polynomial(
                indices_list[kk], coeffs_list[kk][:, None])
            inner_products[jj, ii] = inner_products[ii, jj]
            kk += 1

    # inner_products = np.zeros((num_indices,num_indices),float)
    # for ii in range(num_indices):
    #     for jj in range(ii,num_indices):
    #         index = as_poly_indices[:,ii]+as_poly_indices[:,jj]
    #         coeffs, indices = coeffs_of_active_subspace_polynomial(
    #             W1_trans,index)
    #         inner_products[ii,jj] = integrate_polynomial(indices,coeffs)
    #         inner_products[jj,ii]=inner_products[ii,jj]
    return inner_products


def sample_based_inner_products_on_active_subspace(W1, basis_matrix_func,
                                                   as_indices,
                                                   num_samples,
                                                   generate_samples):
    """
    """
    num_active_vars = as_indices.shape[0]
    assert num_active_vars == W1.shape[1]
    inner_product_indices = np.empty(
        (num_active_vars, as_indices.shape[1]**2), dtype=int)
    for ii in range(as_indices.shape[1]):
        for jj in range(as_indices.shape[1]):
            inner_product_indices[:, ii*as_indices.shape[1]+jj] =\
                as_indices[:, ii]+as_indices[:, jj]

    samples = generate_samples(int(num_samples))
    active_samples = np.dot(W1.T, samples)

    chunk_size = num_samples//10
    vandermonde = basis_matrix_func(
        inner_product_indices, active_samples[:, :chunk_size])
    moments = np.sum(vandermonde, axis=0)
    ii = 1
    while True:
        vandermonde = basis_matrix_func(
            inner_product_indices,
            active_samples[:, chunk_size*ii:min(
                chunk_size*(ii+1), num_samples)])
        moments += np.sum(vandermonde, axis=0)
        if chunk_size*(ii+1) > num_samples:
            break
        ii += 1

    moments /= float(num_samples)
    moments = moments.reshape(as_indices.shape[1], as_indices.shape[1])
    return moments


def sort_2d_vertices_by_polar_angle(vertices):
    # compute centroid
    cent = (sum([p[0] for p in vertices.T])/len(vertices.T),
            sum([p[1] for p in vertices.T])/len(vertices))
    # sort by polar angle
    sorted_vertices = np.array(
        sorted(vertices.T,
               key=lambda p: np.math.atan2(p[1]-cent[1], p[0]-cent[0]))).T
    return sorted_vertices


def find_line_perpendicular_to_active_subspace(W, point):
    """
    Find a line perpenidicular to a one dimensional active subspace that passes
    through a specified point in that 1d subspace and that does not extend
    outside the bounds of the 2d polygon defined by the original hyperbube
    coordinates mapped to the rorated (but not truncated) active subspace
    coordinate system

    Assume we have hypercube on [-1,1]^d, d=2
    W: matrix (d x d)
       rotation matrix usually the transpose of activesubspace eigenvectors
    point: vector (d)
       the point on the active subspace the line must pass through

    return line: list [lb,up]
           the upper and lower limits of the line
    """
    W1 = W[0, :]
    cnt = 0
    line = np.zeros((2, 2))
    # top edge x2==1
    x1 = (point-W1[1])/W1[0]
    if abs(x1) <= 1 and W1[0] != 0:
        line[:, cnt] = np.array([x1, 1])
        cnt += 1
    # bottom edge x2==-1
    x1 = (point+W1[1])/W1[0]
    if abs(x1) <= 1 and W1[0] != 0:
        line[:, cnt] = np.array([x1, -1])
        cnt += 1
    # right edge x1==1
    x2 = (point-W1[0])/W1[1]
    if abs(x2) <= 1 and W1[1] != 0:
        line[:, cnt] = np.array([1, x2])
        cnt += 1
    # left edge x1==-1
    x2 = (point+W1[0])/W1[1]
    if abs(x2) <= 1 and W1[1] != 0:
        line[:, cnt] = np.array([-1, x2])
        cnt += 1
    line = np.dot(W, line)
    assert cnt == 2
    return line


def map_m11_to_ab(x, a, b):
    """
    map points in [-1,1] to [a,b]
    """
    assert x.ndim == 1
    return 0.5*((b-a)*x+(b+a))


def integrate_density_a_2d_function_long_a_line(W, line, point, density_fn,
                                                num_quad_points=100):
    """
    Integrate density along line perpendicular to active subspace line

    W: matrix (d x d)
       rotation matrix usually the transpose of activesubspace eigenvectors
    point: vector (d)
       the point on the active subspace the line must pass through
    line: list [lb,ub]
       the upper and lower limits of the line
    """
    lb = line[1, :].min()
    ub = line[1, :].max()

    # get gauss quadrature points and weights in [-1,1]
    x, w = np.polynomial.legendre.leggauss(num_quad_points)
    w *= (ub-lb)/2
    # map points from [-1,1] to [lb,ub]
    x = map_m11_to_ab(x, lb, ub)

    # 2d quadrature points along the line with the first dimesion fixed at
    # the value of point
    x = np.vstack((np.ones((1, x.shape[0]))*point, (x.reshape(1, x.shape[0]))))

    # quadrature points in full space coordinates
    samples = np.dot(W.T, x)  # W.T = inv(W) because W is orthogonal
    vals = density_fn(samples)
    integral = np.dot(vals, w)
    return integral


def plot_evaluate_active_subspace_density_1d_step(
        line, points_for_eval_in_interval, rotated_vertices, density_fn,
        points_for_eval, cnt_all, W, mapped_vertices, density_vals):
    f, axs = plt.subplots(1, 2, sharey=False, figsize=(16, 6))
    axs[0].plot(line[0, :], line[1, :], '^b-', ms=10)
    axs[0].plot(points_for_eval_in_interval, 0, 'rs')
    axs[0].plot(rotated_vertices[0, :], rotated_vertices[1, :], 'o-k')
    axs[0].plot(rotated_vertices[0, :], rotated_vertices[1, :], 'o-k')
    ss_samples = np.vstack(
        (points_for_eval[:cnt_all], np.zeros(cnt_all, float)))
    axs[0].plot(ss_samples[0, :], ss_samples[1, :], 'r-')
    II = [0, 3]
    axs[0].plot(rotated_vertices[0, II], rotated_vertices[1, II], 'o-k')

    def rotated_density_fn(x): return density_fn(np.dot(W.T, x))
    limits = [
        rotated_vertices[0, :].min(), rotated_vertices[0, :].max(),
        rotated_vertices[1, :].min(), rotated_vertices[1, :].max()]
    X, Y, Z = get_meshgrid_function_data(
        rotated_density_fn, limits, 51)
    xx = np.vstack(
        (X.flatten()[np.newaxis, :], Y.flatten()[np.newaxis, :]))
    II = np.where(np.absolute(np.dot(W.T, xx)) > 1)[1]
    Z = Z.flatten()
    levels = np.linspace(Z.min(), Z.max(), 30)
    Z[II] = np.nan
    cmap = plt.cm.coolwarm
    cmap.set_bad('white', 1.)
    Z = Z.reshape(X.shape[0], X.shape[1])
    # axs[0].imshow(Z[::-1,:],extent=limits,cmap=cmap)
    axs[0].contourf(
        X, Y, Z, extent=limits, cmap=cmap, levels=levels)
    axs[1].set_xlim([mapped_vertices.min(), mapped_vertices.max()])
    axs[1].set_ylim([0, 2])
    axs[1].plot(
        points_for_eval[:cnt_all], density_vals[:cnt_all], 'k')


def evaluate_active_subspace_density_1d(W, density_fn, points_for_eval=None,
                                        num_quad_points=100, plot_steps=True):
    """
    Assume we have hypercube on [-1,1]^d, d=2
    W: matrix (d x d)
       rotation matrix usually the transpose of activesubspace eigenvectors
    num_quad_points: int
       number of quadrature points used to integrate 2d density along a line
    points_for_eval: vector (d)
       points at which to evalute the 1d active subspace density

    """
    assert W.shape[0] == W.shape[1] == 2

    num_dims = 2
    # Split rotation matrix into active subspace eigenvector
    W1 = W[0, :]
    # and inactive subspace eigenvector
    # W2 = W[1, :]

    # define hypercube vertices
    hypercube_vertices_1d = np.array([-1., 1.])
    hypercube_vertices = cartesian_product([hypercube_vertices_1d]*num_dims, 1)

    # compute location of each hypercube vertices in the 1d active subspace.
    # multiple vertices may map to the same point
    mapped_vertices = np.dot(W1, hypercube_vertices).squeeze()
    mapped_vertices = np.sort(mapped_vertices)

    subspace_1d_bounds = mapped_vertices.min(), mapped_vertices.max()

    # compute location of each hypercube vertices in the rotated
    # (but still full dimensional) active subspace coordinates.
    rotated_vertices = np.dot(W, hypercube_vertices)
    # sort vertices for plotting
    rotated_vertices = sort_2d_vertices_by_polar_angle(rotated_vertices)

    # define points for plotting in 1d active subspace
    if points_for_eval is None:
        num_points_for_eval = 102
        delta = (subspace_1d_bounds[1]-subspace_1d_bounds[0])/(
            num_points_for_eval)
        points_for_eval = np.linspace(
            subspace_1d_bounds[0]+delta/2.,
            subspace_1d_bounds[1]-delta/2., num_points_for_eval)
    else:
        # map subspace eval points from [-1,1] to [lb,ub]
        assert points_for_eval.min() >= -1 and points_for_eval.max() <= 1
        points_for_eval = map_m11_to_ab(
            points_for_eval, subspace_1d_bounds[0], subspace_1d_bounds[1])

    # Initialize memory and counters
    cnt_all = 0
    density_vals = np.zeros(points_for_eval.shape[0], float)
    for i in range(mapped_vertices.shape[0]-1):
        # Find points in 1d active subspace that map to the
        # current edge of the two dimensional hypercube
        II = np.where(
            (points_for_eval >= mapped_vertices[i]) &
            (points_for_eval < mapped_vertices[i+1]))[0]
        points_for_eval_in_interval = points_for_eval[II]
        num_points_for_eval_in_interval = points_for_eval_in_interval.shape[0]

        # Check that no points for eval coincide with the location of
        # the mappeed hypercube vertices. Such points will mess up search
        # for perpendicular line
        assert np.all(points_for_eval_in_interval != mapped_vertices[i])
        assert np.all(points_for_eval_in_interval != mapped_vertices[i+1])

        # integrate desity on zonotope along the line perpendicular to the
        # active subspace that runs through the point
        # points_for_eval_in_interval[i] and extends to edges on original
        # hypercube. At some locations line may only touch
        # the boundary of the hyercube on one side of the active subspace
        for j in range(num_points_for_eval_in_interval):
            line = find_line_perpendicular_to_active_subspace(
                W, points_for_eval_in_interval[j])

            density = integrate_density_a_2d_function_long_a_line(
                W, line, points_for_eval_in_interval[j], density_fn,
                num_quad_points=num_quad_points)

            points_for_eval[cnt_all] = points_for_eval_in_interval[j]
            density_vals[cnt_all] = density

            cnt_all += 1

            if plot_steps:
                plot_evaluate_active_subspace_density_1d_step(
                    line, points_for_eval_in_interval[j], rotated_vertices,
                    density_fn, points_for_eval, cnt_all, W, mapped_vertices,
                    density_vals)
                plt.show()
    assert cnt_all == points_for_eval.shape[0]

    return density_vals, subspace_1d_bounds


def evaluate_active_subspace_density_1d_example(density_fn, tol, test=False):

    num_quad_samples = 100
    points_for_eval, quad_weights = np.polynomial.legendre.leggauss(
        num_quad_samples)

    # for varing rotations make sure density on 1d active subspace
    # integrates to 1
    num_rotations = 23
    angles = np.linspace(0., np.pi*2., num_rotations)
    import matplotlib.pyplot as plt
    if not test:
        f, axs = plt.subplots(1, 2, sharey=False, figsize=(16, 6))
    for i in range(1, angles.shape[0]):
        W = np.array([[np.cos(angles[i]), -np.sin(angles[i])],
                      [np.sin(angles[i]), np.cos(angles[i])]])
        density_vals, subspace_1d_bounds = \
            evaluate_active_subspace_density_1d(
                W, density_fn, points_for_eval, num_quad_samples, False)
        integral_density = np.dot(density_vals, quad_weights)
        # Gauss legendre weights must be scaled to correct for length of
        # subspace interval. If integrating
        # int_a^b f(t)dt but gauss points are x in [-1,1] then dt=(b-a)/2 dx
        integral_density *= (subspace_1d_bounds[1]-subspace_1d_bounds[0])/2
        # print abs(1-integral_density)
        assert np.allclose(integral_density, 1., atol=tol)

        if not test:
            axs[0].plot(map_m11_to_ab(points_for_eval, subspace_1d_bounds[0],
                                      subspace_1d_bounds[1]),
                        density_vals, 'k', lw=2)
        hypercube_vertices_1d = np.array([-1., 1.])
        hypercube_vertices = cartesian_product(
            [hypercube_vertices_1d]*2, 1)
        mapped_vertices = np.dot(W, hypercube_vertices)
        if not test:
            plot_2d_polygon(mapped_vertices, axs[1])
    if not test:
        plt.show()


def plot_evaluate_active_subspace_density_1d_steps():
    # niform_density_fn = lambda x: np.ones(x.shape[1])*0.25
    alpha = 2
    beta = 5

    def beta_density_function_1d(x):
        return stats.beta.pdf((x+1.)/2., alpha, beta)/2

    def beta_density_fn(x): return beta_density_function_1d(
        x[0, :])*beta_density_function_1d(x[1, :])
    angle = 0.25*np.pi
    W = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    points_for_eval = np.linspace(-.99, .99, 10)
    evaluate_active_subspace_density_1d(
        W, beta_density_fn, points_for_eval=points_for_eval,
        num_quad_points=100, plot_steps=True)
