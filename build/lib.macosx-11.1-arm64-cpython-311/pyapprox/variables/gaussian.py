import numpy as np
import copy


def get_operator_diagonal(
        operator, num_vars, eval_concurrency, transpose=False,
        active_indices=None):
    r"""
    Dont want to solve for all vectors at once because this will
    likely be to large to fit in memory.

    active indices: np.ndarray
       only do some entries of diagonal
    """
    if active_indices is None:
        active_indices = np.arange(num_vars)
    else:
        assert active_indices.shape[0] <= num_vars

    num_active_indices = active_indices.shape[0]

    # diagonal = np.empty((num_vars),float)
    diagonal = np.empty((num_active_indices), float)
    cnt = 0
    # while cnt<num_vars:
    while cnt < num_active_indices:
        # num_vectors = min(eval_concurrency, num_vars-cnt)
        num_vectors = min(eval_concurrency, num_active_indices-cnt)
        vectors = np.zeros((num_vars, num_vectors), dtype=float)
        for j in range(num_vectors):
            # vectors[cnt+j,j]=1.0
            vectors[active_indices[cnt+j], j] = 1.0
        tmp = operator(vectors, transpose=transpose)
        for j in range(num_vectors):
            diagonal[cnt+j] = tmp[active_indices[cnt+j], j]
        cnt += num_vectors
    return diagonal


class GaussianSqrtCovarianceOperator(object):
    def apply(self, vectors):
        r"""
        Apply the sqrt of the covariance to a set of vectors x.
        If the vectors are standard normal samples then the result
        will be a samples from the Gaussian.

        Parameters
        ----------
        vector : np.ndarray (num_vars x num_vectors)
            The vectors x
        """
        raise Exception('Derived classes must implement this function')

    def num_vars(self):
        raise Exception('Derived classes must implement this function')

    def __call__(self, vectors, transpose):
        return self.apply(vectors, transpose)


class CholeskySqrtCovarianceOperator(GaussianSqrtCovarianceOperator):
    def __init__(self, covariance, eval_concurrency=1, fault_percentage=0):
        super(CholeskySqrtCovarianceOperator, self).__init__()

        assert fault_percentage < 100
        self.fault_percentage = fault_percentage

        self.covariance = covariance
        self.chol_factor = np.linalg.cholesky(self.covariance)
        self.eval_concurrency = eval_concurrency

    def apply(self, vectors, transpose):
        r"""

        Apply the sqrt of the covariance to a st of vectors x.
        If a vector is a standard normal sample then the result
        will be a sample from the prior.

        C_sqrt*x

        Parameters
        ----------
        vectors : np.ndarray (num_vars x num_vectors)
           The vectors x

        transpose : boolean
            Whether to apply the action of the sqrt of the covariance
            or its tranpose.

        Returns
        -------
        result : np.ndarray (num_vars x num_vectors)
            The vectors C_sqrt*x
        """
        assert vectors.ndim == 2
        num_vectors = vectors.shape[1]
        if transpose:
            result = np.dot(self.chol_factor.T, vectors)
        else:
            result = np.dot(self.chol_factor, vectors)
        u = np.random.uniform(0., 1., (num_vectors))
        I = np.where(u < self.fault_percentage/100.)[0]
        result[:, I] = np.nan
        return result

    def num_vars(self):
        r"""
        Return the number of variables of the multivariate Gaussian
        """
        return self.covariance.shape[0]


class CovarianceOperator(object):
    r"""
    Class to compute the action of the a covariance opearator on a
    vector.

    We can compute the matrix rerpesentation of the operator C = C_op(I)
    where I is the identity matrix.
    """

    def __init__(self, sqrt_covariance_operator):
        self.sqrt_covariance_operator = sqrt_covariance_operator

    def apply(self, vectors, transpose):
        r"""
        Compute action of covariance C by applying action of sqrt_covariance L
        twice, where C=L*L.T. E.g.
        C(x) = L(L.T(x))
        Tranpose is ignored because covariance operators are symmetric
        """
        tmp = self.sqrt_covariance_operator.apply(vectors, True)
        result = self.sqrt_covariance_operator.apply(tmp, False)
        return result

    def __call__(self, vectors, transpose):
        r"""
        Tranpose is ignored because covariance operators are symmetric
        """
        return self.apply(vectors, None)

    def num_rows(self):
        return self.sqrt_covariance_operator.num_vars()

    def num_cols(self):
        return self.num_rows()


class MultivariateGaussian(object):
    def __init__(self, sqrt_covariance_operator, mean=0.):
        self.sqrt_covariance_operator = sqrt_covariance_operator
        if np.isscalar(mean):
            mean = mean*np.ones((self.num_vars()))
        assert mean.ndim == 1 and mean.shape[0] == self.num_vars()
        self.mean = mean

    def num_vars(self):
        r"""
        Return the number of variables of the multivariate Gaussian
        """
        return self.sqrt_covariance_operator.num_vars()

    def apply_covariance_sqrt(self, vectors, transpose):
        return self.sqrt_covariance_operator(vectors, transpose)

    def generate_samples(self, nsamples):
        std_normal_samples = np.random.normal(
            0., 1., (self.num_vars(), nsamples))
        samples = self.apply_covariance_sqrt(std_normal_samples, False)
        samples += self.mean[:, np.newaxis]
        return samples

    def pointwise_variance(self, active_indices=None):
        r"""
        Get the diagonal of the Gaussian covariance matrix.
        Default implementation is two apply the sqrt operator twice.
        """
        covariance_operator = CovarianceOperator(self.sqrt_covariance_operator)
        return get_operator_diagonal(
            covariance_operator, self.num_vars(),
            self.sqrt_covariance_operator.eval_concurrency,
            active_indices=active_indices)


def subselect_matrix_blocks(selected_block_indices, nentries_per_block):
    r"""
    Return the vector indices of a subset of blocks from a block vector from
    the block indices.

    Parameters
    ----------
    selected_block_indices : np.ndarray (nselected_blocks)
        Integer indentifier of selected row blocks.

    nentries_per_block : np.ndarray (nblocks)
        The number of consecutive rows associated with each block

    Returns
    -------
    selected_vector_indices : np.ndarray (nselected_vector_indices)
        The indices of the individual vector entries which where selected
    """
    assert selected_block_indices.shape[0] <= nentries_per_block.shape[0]
    nrows, nblocks = np.sum(nentries_per_block), len(nentries_per_block)
    block_starts = np.append(0, np.cumsum(nentries_per_block))
    matrix_indices = np.arange(nrows, dtype=int)
    selected_matrix_indices = np.concatenate(
        [matrix_indices[block_starts[ii]:block_starts[ii+1]]
         for ii in selected_block_indices])
    return selected_matrix_indices


def get_matrix_partition_indices(selected_block_ids, block_ids,
                                 nentries_per_block, issubset=True):
    r"""
    Select a subset of blocks from a block vector from unique block ids.

    Warning: keep_rows and leave_rows will be returned in order they appear in
    in all_block_ids and not in the order they appear in selected_block_ids.

    Parameters
    ----------
    selected_block_ids : np.ndarray (nselected_blocks)
        Integer indentifier of selected blocks. Must be a subset of
        all_block_ids.

    block_ids : np.ndarray (nblocks)
        Integer identifier of all blocks. Ids must be unique but
        can be negative and do not need to be consecutive or
        montonically increasing or decreasing

    nentries_per_block :  np.ndarray (nblocks)
        The number of consecutive rows associated with each block

    Returns
    -------
    selected_vector_indices : np.ndarray (nselected_vector_indices)
        The indices of the individual vector entries selected

    selected_vector_indices_reduced : np.ndarray (numselected_indices)
        The indices, of the individual vector entries selected,
        into a vector consisting of only the selected blocks
    """
    nvector_entries, nblocks = np.sum(
        nentries_per_block), len(nentries_per_block)
    nselected_blocks = len(selected_block_ids)
    assert len(nentries_per_block) == len(block_ids)
    nentries_per_block = np.asarray(nentries_per_block)

    __, relevant_block_indices, relevant_selected_block_indices =\
        np.intersect1d(block_ids, selected_block_ids, return_indices=True)

    selected_vector_indices = subselect_matrix_blocks(
        relevant_block_indices, nentries_per_block)

    selected_vector_indices_reduced = subselect_matrix_blocks(
        relevant_selected_block_indices, nentries_per_block[relevant_block_indices][np.argsort(relevant_selected_block_indices)])

    return selected_vector_indices, selected_vector_indices_reduced


def condition_gaussian_on_data(mean, covariance, fixed_indices, values):
    r"""
    Compute conditional density of a multivariate Gaussian

    p(x1|x2)

    with covariance and mean

    C = [C11 C12],  m = [m1]
        [C21 C22]       [22]

    and

    x=[x1]
      [x2].

    The conditional mean and variance are

    m_1|2 = m1+C12*C22^{-1}*(x2-m2)
    C_1|2 = C11-C12*C22^{-1}*C21


    Parameters
    ----------
    mean : np.ndarray (num_vars)
        The mean m

    covariance : np.ndarray (num_vars, num_vars)
        The covariance matrix C

    fixed_indices : np.ndarray (num_fixed_vars)
        Indices of variables x2 to condition on (in the order they enter
        precision matrix)

    values : np.ndarray (num_fixed_vars)
        The values of the conditioned variables

    Returns
    -------
    new_mean : np.ndarray (num_remain_vars)
        The mean of the conditional Gaussian


    new_covariance : np.ndarray (num_remain_vars, num_remain_vars)
        The matrix C of the conditional Gaussian

    """
    indices = set(np.arange(mean.shape[0]))
    ind_remain = list(indices.difference(set(fixed_indices)))
    diff = values - mean[fixed_indices]
    sigma_11 = np.array(covariance[np.ix_(ind_remain, ind_remain)], ndmin=2)
    sigma_12 = np.array(
        covariance[np.ix_(ind_remain, fixed_indices)], ndmin=2)
    sigma_22 = np.array(
        covariance[np.ix_(fixed_indices, fixed_indices)], ndmin=2)
    update = np.dot(sigma_12, np.linalg.solve(sigma_22, diff)).flatten()
    new_mean = mean[ind_remain] + update
    new_cov = sigma_11 - \
        np.dot(sigma_12, np.linalg.solve(sigma_22, sigma_12.T))
    return new_mean, new_cov


def multiply_gaussian_densities(mean1, covariance1, mean2, covariance2):
    r"""
    Multiply two multivariate Gaussians with mean and covariance given by
    m1, m2 and C1,C2.

    Note it is much easier to multiply Gaussian densities using the canonical
    form. See multiply gaussians in canonical form.

    Parameters
    ----------
    mean1 : np.ndarray (num_vars)
        The mean m1

    covariance1 : np.ndarray (num_vars, num_vars)
        The covariance matrix C1


    mean2 : np.ndarray (num_vars)
        The mean m1

    covariance2 : np.ndarray (num_vars, num_vars)
        The covariance matrix C1

    Returns
    -------
    new_mean : np.ndarray (num_vars)
        The mean of the conditional Gaussian


    new_covariance : np.ndarray (num_vars, num_vars)
        The matrix C of the new Gaussian
    """
    assert covariance1.shape == covariance2.shape
    assert mean1.shape == mean2.shape
    assert mean1.shape[0] == covariance1.shape[0]

    C1pC2_inv = np.linalg.inv(covariance1+covariance2)     # inv(C1+C2)
    covariance = covariance1.dot(C1pC2_inv).dot(covariance2)
    mean = covariance2.dot(C1pC2_inv).dot(mean1) +\
        covariance1.dot(C1pC2_inv).dot(mean2)

    # Above is faster as it only requires one matrix inversion instead of the
    # three used below.
    # C1_inv = np.linalg.inv(covariance1)
    # C2_inv = np.linalg.inv(covariance2)
    # covariance = np.linalg.inv(C1_inv+C2_inv)
    # mean = covariance.dot(C1_inv).dot(mean1)+covariance.dot(C2_inv).dot(mean2)
    return mean, covariance


def get_unique_variable_blocks(var1_ids, nvars_per_var1, var2_ids, nvars_per_var2):
    r"""
    Get the unique variable blocks in two multivariate variables x1 and x2.


    Parameters
    ----------
    var1_ids : list (nvars1)
        List of ids of the block variables in x1

    nvars_per_var1 : list (nvars1)
        Number of variables within each block in x1

    var2_ids : list (nvars2)
        List of ids of the block variables in x2

    nvars_per_var2 : list (nvars2)
        Number of variables within each block in x2

    Returns
    -------
    all_var_ids : list
       Unique ids of the block variables found in either x1 and x2

    nvars_per_all_vars : list
       Sizes of the block variables found in either x1 and x2
    """
    all_var_ids = copy.deepcopy(var1_ids)
    nvars_per_all_vars = copy.deepcopy(nvars_per_var1)
    var2_ids_set = set(var1_ids)
    for ii in range(len(var2_ids)):
        var_id = var2_ids[ii]
        if not var_id in var1_ids:
            all_var_ids = np.append(all_var_ids, var_id)
            nvars_per_all_vars = np.append(
                nvars_per_all_vars, nvars_per_var2[ii])

    return all_var_ids, nvars_per_all_vars


def expand_scope_of_gaussian(old_var_ids, new_var_ids, nvars_per_new_var, matrix,
                             vector):
    r"""
    Expand a compact representation of a multivariate Gaussian to include
    a new set of inactive variables. This function can be used for Gaussians
    in canonical or traditional form.

    The new entries of the new matrix and the vector will be set to zero.

    Parameters
    ----------
    old_var_ids : list (num_old_vars)
        List of ids of the variables in the compact representation

    new_var_ids : list (num_new_vars)
        List of ids of the variables in the expanded representation. new_var_ids
        must contain old_var_ids as a subset

    nvars_per_new_var : np.ndarray (num_new_vars)
        The number of variables in each mulitvariate variable in the compact
        representation

    matrix : np.ndarray (nvars, nvars)
        The precision matrix K or covariance matrix C of the compact
        representation

    vector : np.ndarray (nvars)
        The shift vector h or mean vector m of the compact representation

    Returns
    -------
    new_matrix : np.ndarray (nvars,nvars)
        The new precision matrix K or covariance Matrix C
        of the expanded representation

    new_vector : np.ndarray (nvars)
        The new shift h or mean m of the expanded representation

    """
    selected_rows_expanded, selected_rows = get_matrix_partition_indices(
        old_var_ids, new_var_ids, nvars_per_new_var)
    num_new_vars = sum(nvars_per_new_var)
    new_matrix = np.zeros((num_new_vars, num_new_vars))
    new_matrix[np.ix_(selected_rows_expanded, selected_rows_expanded)] = \
        matrix[np.ix_(selected_rows, selected_rows)]
    new_vector = np.zeros((num_new_vars))
    new_vector[selected_rows_expanded] = vector[selected_rows]
    return new_matrix, new_vector


def compute_gaussian_pdf_canonical_form_normalization(mean, shift, precision):
    r"""
    Compute the normalization factor

    g = -0.5 m^T h -0.5 n\log(2\pi) +0.5 \log |K|,

    of a Gaussian distribution in canonical form

    p(x|h,K) = exp(g+h^T x-0.5 x^T K x) x \in R^n

    where

    h = K m and K=inv(C)

    and m and C are the mean a covariance respectively.

    Parameters
    ----------
    precision_matrix : np.ndarray (nvars, nvars)
        The matrix K

    shift : np.ndarray (nvars)
        The vector h

    mean : np.ndarray (nvars)
        The mean m

    Returns
    -------
    normalization : float
        The normalization constant g
    """
    nvars = precision.shape[0]
    g = 0.5*(-mean.T.dot(shift)-nvars*np.log(2*np.pi) +
             np.linalg.slogdet(precision)[1])
    return g


def convert_gaussian_from_canonical_form(precision_matrix, shift):
    r"""
    Convert a Gaussian in canonical form

    p(x|h,K) = exp(g+h^T x-0.5 x^T K x) x \in R^n

    g = -0.5 m^T h -0.5 n\log(2\pi) +0.5 \log |K|,

    h = K m and K=inv(C)

    into a Gaussian distribution specified by a mean m and a covariance C.

    Parameters
    ----------
    precision_matrix : np.ndarray (nvars, nvars)
        The matrix K

    shift : np.ndarray (nvars)
        The vector h

    normalization : float
        The normalization constant g

    Returns
    ------
    mean : np.ndarray (nvars)
        The mean m

    covariance : np.ndarray (nvars,nvars)
        The covariance C
    """
    covar = np.linalg.inv(precision_matrix)
    mean = np.dot(covar, shift)
    return mean, covar


def convert_gaussian_to_canonical_form(mean, covariance):
    r"""
    Convert a Gaussian distribution specified by a mean m and a covariance C

    into canonical form

    p(x|h,K) = exp(g+h^T x-0.5 x^T K x) x \in R^n

    g = -0.5 m^T h -0.5 n\log(2\pi) +0.5 \log |K|,

    h = K m and K=inv(C)

    Parameters
    ----------
    mean : np.ndarray (nvars)
        The mean m

    covariance : np.ndarray (nvars,nvars)
        The covariance C

    Returns
    -------
    precision_matrix : np.ndarray (nvars,nvars)
        The matrix K

    shift : np.ndarray (nvars)
        The vector h

    normalization : float
        The normalization constant g
    """
    # TODO: compute cholesky factorization then compuate inverse
    # pass cholesky factor to compute normalization which can then
    # be cheaply used to compute determinant
    precision_matrix = np.linalg.inv(covariance)
    shift = precision_matrix.dot(mean)
    normalization = compute_gaussian_pdf_canonical_form_normalization(
        mean, shift, precision_matrix)
    return precision_matrix, shift, normalization


def condition_gaussian_in_canonical_form(fixed_indices, precision_matrix,
                                         shift, normalization, data,
                                         remain_indices=None):
    r"""
    Compute conditional density

    p(x1|x2)

    of joint density in canonical form

    p(x|h,K) = exp(g+h^T x-0.5 x^T K x) x \in R^n

    g = -0.5 m^T h -0.5 n\log(2\pi) +0.5 \log |K|,

    m is mean of distrbution and h = K m

    x=[x1]
      [x2]

    K = [K11 K12],  h = [h1]
        [K21 K22]       [h2]

    h_1|2 = h1-K12*x2
    K_1|2 = K11
    g_1|2 = g + h2*y - 0.5*y*K22*y

    y are the function values at the coordinates x2

    Canonical form of multivariate normal can be obtained using the following
    relationships

    mean       : m = K^{-1}h
    covariance : C = K^{-1}

    Computing the conditional Gaussian density p(x1|x2) is easier using
    the canonical form than the standard form. To condition using the
    standard form we have

    m_1|2 = m1+C12*C22^{-1}*(x2-m2)
    C_1|2 = C11-C12*C22^{-1}*C21


    Parameters
    ----------
    fixed_indices : np.ndarray (num_fixed_vars)
        Indices of variables x2 to condition on (in the order they enter
        precision matrix)

    data : np.ndarray (num_fixed_vars)
        Values of variables x2 to condition on (in the order they enter
        precision matrix)

    precision_matrix : np.ndarray (nvars, nvars)
        The matrix K

    shift : np.ndarray (nvars)
        The vector h

    normalization : float
        The normalization constant g


    Returns
    -------
    new_precision_matrix : np.ndarray (num_remain_vars, num_remain_vars)
        The matrix K of the conditional Gaussian

    new_shift : np.ndarray (num_remain_vars)
        The vector h of the conditional Gaussian

    new_normalization : float
        The normalization constant g of the conditional Gaussian

    """

    # if len(leave_rows) == 0:
    #     return precision_matrix, shift, normalization
    # keep_values = values[keep_rows]
    assert data.shape[0] == fixed_indices.shape[0]

    if remain_indices is None:
        indices = np.arange(precision_matrix.shape[0])
        remain_indices = np.setdiff1d(indices, fixed_indices)

    KXX = precision_matrix[np.ix_(remain_indices, remain_indices)]
    KXY = precision_matrix[np.ix_(remain_indices, fixed_indices)]
    KYY = precision_matrix[np.ix_(fixed_indices, fixed_indices)]
    new_precision_matrix = KXX

    hx = shift[np.ix_(remain_indices)]
    hy = shift[np.ix_(fixed_indices)]
    new_shift = hx - np.dot(KXY, data)

    new_normalization = normalization + \
        hy.dot(data)-0.5*data.T.dot(KYY.dot(data))

    return new_precision_matrix, new_shift, new_normalization


def marginalize_gaussian_in_canonical_form(marg_indices, precision_matrix,
                                           shift, normalization,
                                           remain_indices=None):
    r"""
    Compute marginal density

    p(x1)

    of joint density in canonical form

    p(x|h,K) = exp(g+h^T x-0.5 x^T K x) x \in R^n

    g = -0.5 m^T h -0.5 n\log(2\pi) +0.5 \log |K|,

    m is mean of distrbution and h = K m

    x=[x1]
      [x2]

    K = [K11 K12],  h = [h1]
        [K21 K22]       [h2]

    h_1^m = h1-K12*K22^{-1}*h2
    K_1^m = K11-K12*K22^{-1}*K21
    g_1^m = g + 0.5*(n2*log(2\pi)+\log|K22|+h2^T*K22*h2)

    y are the function values at the coordinates x2

    Canonical form of multivariate normal can be obtained using the following
    relationships

    mean       : m = K^{-1}h
    covariance : C = K^{-1}

    Note computing the marginal Gaussian density p(x1) is easier using
    the standard form than the canonical form. To marginalize using the
    standard form we have

    m_1^m = m1
    C_1^m = C11

    Parameters
    ----------
    marg_indices : np.ndarray (num_marg_vars)
        Indices of variables x2 to marginalize out (in the order they enter
        precision matrix)

    precision_matrix : np.ndarray (nvars, nvars)
        The matrix K

    shift : np.ndarray (nvars)
        The vector h

    normalization : float
        The normalization constant g


    Returns
    -------
    new_precision_matrix : np.ndarray (num_remain_vars, num_remain_vars)
        The matrix K of the marginalized Gaussian

    new_shift : np.ndarray (num_remain_vars)
        The vector h of the marginalized Gaussian

    new_normalization : float
        The normalization constant g of the marginalized Gaussian
    """
    if remain_indices is None:
        indices = np.arange(precision_matrix.shape[0])
        remain_indices = np.setdiff1d(indices, marg_indices)

    assert max(marg_indices) < precision_matrix.shape[0]

    KXX = precision_matrix[np.ix_(remain_indices, remain_indices)]
    KXY = precision_matrix[np.ix_(remain_indices, marg_indices)]
    KYX = precision_matrix[np.ix_(marg_indices, remain_indices)]
    KYY = precision_matrix[np.ix_(marg_indices, marg_indices)]

    solved = np.linalg.solve(KYY, KYX)
    new_precision_matrix = KXX - np.dot(KXY, solved)

    hx = shift[np.ix_(remain_indices)]
    hy = shift[np.ix_(marg_indices)]

    new_shift = hx - np.dot(solved.T, hy)
    hyKhy = np.dot(hy.T, np.dot(KYY, hy))

    logdet = np.linalg.slogdet(KYY)[1]
    new_normalization = normalization + 0.5 * (
        len(marg_indices) * np.log(2.0*np.pi) + logdet + hyKhy)

    return new_precision_matrix, new_shift, new_normalization


def joint_density_from_linear_conditional_relationship(mean1, cov1, cov2g1,
                                                       Amat, bvec):
    r"""
    Compute joint density :math:`P(x_1,x_2)`

    Given :math:`P(x_1)` normal with mean and covariance :math:`m_1, C_{11}`
    Given :math:`P(x_2\mid x_1)` normal with mean and covariance
    :math:` m_{2\mid 1}=A x_1+b, C_{2\mid 1}`

    Compute normal joint density  :math:`P(x_1,x_2)` with mean and covariance

    .. math::

       m = \begin{bmatrix} m_1 // A m_1+b]\end{bmatrix}\qquad
       C = \begin{bmatrix} C_{11}  &  C_{11} A^T\\
                           AC_{11} C_{2\mid 1}+A C_{11} A^T\end{bmatrix}

    Parameters
    ----------
    mean1 : np.ndarray (nvars1)
        The mean :math:`m_1` of the Gaussian distribution of :math:`x_1`

    cov1 : np.ndarray (nvars1,nvars1)
        The covariance :math:`C_{11}` of the Gaussian distribution of
        :math:`x_1`

    cov2g1 : np.ndarray (nvars2,nvars2)
        The covariance :math:`C_{2\mid 1}` of the Gaussian distribution of
        :math:`P(x_2\mid x_1)`

    Amat : np.ndarray (nvars2,nvars1)
        The matrix :math:`A` of the conditional distribution
        :math:`P(x_2\mid x_1)`

    bvec : np.ndarray (nvars2)
        The vector :math:`b` of the conditional distribution
        :math:`P(x_2\mid x_1)`

    Returns
    -------
    joint_mean : np.ndarray (nvars2+nvars1)
       The mean :math:`m` of the joint distribution :math:`P(x1,x2)`

    joint_cov : np.ndarray (nvars2+nvars1,nvars2+nvars1)
       The covariance :math:`C` of the joint distribution :math:`P(x1,x2)`

    Notes
    -----
    To derive the expressions for :math:`m` and :math:`C` we use the well
    known fact that the joint density of two Gaussian variables :math:`x_1,x_2`
    is Gaussian with mean and covariance

    .. math::

       m = \begin{bmatrix} m_1 // m_2]\end{bmatrix}\qquad
       C = \begin{bmatrix} C_{11}  &  C_12\\  C_{21} C_{22}\end{bmatrix}

    We know note that if :math:`x_2 = A x_1+b` then :math:`C_{12}=C_{11}A^T`.

    Now we use the well known expression for the covariance of
    :math:`P(x_2\mid x_1)` given by
    :math:`C_{2\mid 1}=C_{22}-C_{21}C_{11}^{-1}C_{12}` which implies that

    .. math:: C_{22}= C_{2\mid 1}+C_{21}C_{11}^{-1}C_{12}=C_{2\mid 1}+AC_{11}C_{11}^{-1}C_{11}A^T

    Plugging these two expressions into the joint density gives the result.
    """
    AC1 = np.dot(Amat, cov1)
    mean3 = bvec + Amat.dot(mean1)
    cov3 = cov2g1 + AC1.dot(Amat.T)

    joint_mean = np.concatenate([mean1, mean3])
    joint_cov = np.block([[cov1, AC1.T], [AC1, cov3]])

    return joint_mean, joint_cov


def marginal_density_from_linear_conditional_relationship(
        mean1, cov1, cov2g1, Amat, bvec):
    r"""
    Compute the marginal density of P(x2)

    Given p(x1) normal with mean and covariance
        m1, C1
    Given p(x2|x1) normal with mean and covariance
        m_2|1=A*x1+b, C_2|1

    P(x2) is normal with mean and covariance
       m2=A*m1+b, C2=C_2|1+A*C1*A.T

    Parameters
    ----------
    mean1 : np.ndarray (nvars1)
        The mean (m1) of the Gaussian distribution of x1

    cov1 : np.ndarray (nvars1,nvars1)
        The covariance (C1) of the Gaussian distribution of x1

    cov2g1 : np.ndarray (nvars2,nvars2)
        The covariance (C_2|1) of the Gaussian distribution of P(x2|x1)

    Amat : np.ndarray (nvars2,nvars1)
        The matrix (A) of the conditional distribution P(x2|x1)

    bvec : np.ndarray (nvars2)
        The vector (b) of the conditional distribution P(x2|x1)

    Returns
    -------
    mean2 : np.ndarray (nvars2)
       The mean (m2) of P(x2)

    cov2 : np.ndarray (nvars2,nvars2)
       The covariance (C_2) of P(x2)
    """
    AC1 = np.dot(Amat, cov1)
    mean2 = Amat.dot(mean1)+bvec
    cov2 = cov2g1+AC1.dot(Amat.T)
    return mean2, cov2


def conditional_density_from_linear_conditional_relationship(
        mean1, cov1, cov2g1, Amat, bvec, values):
    r"""
    Compute conditional density P(x1|x2)

    Given p(x1) normal with mean and covariance
        m1, C1
    Given p(x2|x1) normal with mean and covariance
        m_2|1=A*x1+b, C_2|1

    P(x1|x2) is normal with mean and covariance
        m_1|2=m_1+C_1*A.T*inv(C_2)*(x2-b-A*m_1)
        C_1|2=C1-C1*A.T*inv(C_2)*A*C1

    Parameters
    ----------
    mean1 : np.ndarray (nvars1)
        The mean (m1) of the Gaussian distribution of x1

    cov1 : np.ndarray (nvars1,nvars1)
        The covariance (C1) of the Gaussian distribution of x1

    cov2g1 : np.ndarray (nvars2,nvars2)
        The covariance (C_2|1) of the Gaussian distribution of P(x2|x1)

    Amat : np.ndarray (nvars2,nvars1)
        The matrix (A) of the conditional distribution P(x2|x1)

    bvec : np.ndarray (nvars2)
        The vector (b) of the conditional distribution P(x2|x1)

    values : np.ndarray (nvars2)
        The values of the conditioned variables x2

    Returns
    -------
    mean_1g2 : np.ndarray (nvars1)
       The mean (m_1|2) of P(x1|x2)

    cov_1g2 : np.ndarray (nvars1,nvars1)
       The covariance (C_1|2) of P(x1|x2)
    """
    AC1 = np.dot(Amat, cov1)

    # Compute marginal mean and covariance of P(x2)
    mean2 = Amat.dot(mean1)+bvec
    cov2 = cov2g1+AC1.dot(Amat.T)

    C2_inv_AC1 = np.linalg.solve(cov2, AC1)

    # Compute conditioanl mean and covariance of P(x1|x2)
    mean_1g2 = mean1+C2_inv_AC1.T.dot(values-mean2)
    cov_1g2 = cov1-AC1.T.dot(C2_inv_AC1)

    return mean_1g2, cov_1g2


def convert_conditional_probability_density_to_canonical_form(
        Amat, bvec, cov, var1_ids, nvars_per_var1, var2_ids, nvars_per_var2):
    r"""
    Convert a Gaussian conditional density (CPD) of the form
    :math:`P(x2|x1)`  with mean and covariance
    :math:`m_{2\mid 1}=A*x_1+b, C_{2\mid 1}`

    into canonical form

    .. math:: P(x|h,K) = \exp\left(g+h^T x-0.5 x^T K x\right)

    where

    .. math:: g = -0.5 m^T h -0.5 n\log(2\pi) +0.5 \log |K|,

    and :math:`m` is mean of distrbution and :math:`h = K m`

    Parameters
    ----------
    cov : np.ndarray (nvars1,nvars1)
        The covariance :math:`C_{2\mid 1}` of the Gaussian distribution
        :math:`P(x_2\mid x_1)`

    Amat : np.ndarray (nvars2,nvars1)
        The matrix :math:`A` of the conditional distribution
        :math:`P(x_2\mid x_1)`

    bvec : np.ndarray (nvars2)
        The vector :math:`b` of the conditional distribution
        :math:`P(x_2\mid x_1)`

    var1_ids : list (nvars1)
        List of ids of the variables in the compact representation of
        :math:`x_1`

    nvars_per_var1 : list (nvars1)
        Size of blocks in variable :math:`x_1`

    var2_ids : list (nvars2)
        List of ids of the variables in the compact_representation of
        :math:`x_2`

    nvars_per_var2 : list (nvars2)
        Size of blocks in variable :math:`x_2`

    Returns
    -------
    precision_matrix : np.ndarray (nvars2, nvars2)
        The matrix :math:`K` of the Gaussian :math:`P(x_2\mid x_1)`

    shift : np.ndarray (nvars2)
        The vector :math:`h` of the Gaussian :math:`P(x_2\mid x_1)`

    new_normalization : float
        The normalization constant :math:`g` of the Gaussian
        :math:`P(x_2\mid x_1)`

    all_var_ids : list (nvars)
        List of ids of the variable blocks in :math:`P(x_2\mid x_1)`

    nvars_per_all_vars : list (nvars)
        Sizes of the variable blocks in :math:`P(x_2\mid x_1)`

    """
    Cinv = np.linalg.inv(cov)
    CinvA = Cinv.dot(Amat)
    precision_matrix = np.block(
        [[Amat.T.dot(CinvA), -CinvA.T], [-CinvA, Cinv]])
    shift = np.concatenate([-CinvA.T.dot(bvec), Cinv.dot(bvec)])
    normalization = compute_gaussian_pdf_canonical_form_normalization(
        bvec, shift[Amat.shape[1]:], Cinv)
    all_var_ids, nvars_per_all_vars = get_unique_variable_blocks(
        var1_ids, nvars_per_var1, var2_ids, nvars_per_var2)

    return precision_matrix, shift, normalization, all_var_ids, nvars_per_all_vars


def multiply_gaussian_densities_in_canonical_form(
        precision_matrix1, shift1, normalization1,
        precision_matrix2, shift2, normalization2):
    r"""
    Multiply two multivariate Gaussian in canonical form.

    Parameters
    ----------
    precision_matrix1 : np.ndarray (nvars1,nvars1)
        The precision matrix K1 of the compact representation of x1

    shift1 : np.ndarray (nvars1)
        The vector h1 of the compact representation of x1

    normalization2 : float
        The normaliation g1 of the comact representation of x1

    precision_matrix2 : np.ndarray (nvars2,nvars2)
        The precision matrix K2 of the compact representation of x2

    shift2 : np.ndarray (nvars2)
        The vector h2 of the compact representation of x1

    normalization2 : float
        The normaliation g2 of the comact representation of x2

    Returns
    -------
    new_precision_matrix : np.ndarray (nvars,nvars)
        The new precision matrix K of the expanded representation

    new_shift : np.ndarray (nvars)
        The new vector h  of the expanded representation

    new_normalization : float
        The new normalization g
    """
    new_precision_matrix = precision_matrix1+precision_matrix2
    new_shift = shift1+shift2
    new_normalization = normalization1+normalization2

    return new_precision_matrix, new_shift, new_normalization


def multiply_gaussian_densities_in_compact_canonical_form(
        precision_matrix1, shift1, normalization1, var1_ids, nvars_per_var1,
        precision_matrix2, shift2, normalization2, var2_ids, nvars_per_var2):
    r"""
    Multiply the compact representations of two multivariate Gaussian in
    canonical form.

    The multivariate Gaussians may consist of different
    univariate variables each associated with a unique identifier.

    Parameters
    ----------
    precision_matrix1 : np.ndarray (nvars1,nvars1)
        The precision matrix K1 of the compact representation of x1

    shift1 : np.ndarray (nvars1)
        The vector h1 of the compact representation of x1

    normalization1 : float
        The normaliation g1 of the comact representation of x1

    var1_ids : list (nvars1)
        List of ids of the variables in the compact representation of x1

    nvars_per_var1 : list (nvars1)
        Size of blocks in variable x1

    precision_matrix2 : np.ndarray (nvars2,nvars2)
        The precision matrix K2 of the compact representation of x2

    shift2 : np.ndarray (nvars2)
        The vector h2 of the compact representation of x1

    normalization2 : float
        The normaliation g2 of the comact representation of x2

    var2_ids : list (nvars2)
        List of ids of the variables in the compact_representation of x2

    nvars_per_var2 : list (nvars2)
        Size of blocks in variable x2

    Returns
    -------
    new_precision_matrix : np.ndarray (nvars,nvars)
        The new precision matrix K of the expanded representation

    new_shift : np.ndarray (nvars)
        The new vector h  of the expanded representation

    new_normalization : float
        The new normalization g

    all_var_ids : list (nvars)
        List of ids of the variable blocks in the product P(x1)*P(x2)

    nvars_per_all_vars : list (nvars)
        Sizes of the variable blocks in the product P(x1)*P(x2)
    """
    all_var_ids, nvars_per_all_vars = get_unique_variable_blocks(
        var1_ids, nvars_per_var1, var2_ids, nvars_per_var2)

    expanded_precision_matrix1, expanded_shift1 = \
        expand_scope_of_gaussian(
            var1_ids, all_var_ids, nvars_per_all_vars, precision_matrix1, shift1)

    expanded_precision_matrix2, expanded_shift2 = \
        expand_scope_of_gaussian(
            var2_ids, all_var_ids, nvars_per_all_vars, precision_matrix2, shift2)

    new_precision_matrix, new_shift, new_normalization = \
        multiply_gaussian_densities_in_canonical_form(
            expanded_precision_matrix1, expanded_shift1, normalization1,
            expanded_precision_matrix2, expanded_shift2, normalization2)

    return new_precision_matrix, new_shift, new_normalization, all_var_ids, \
        nvars_per_all_vars


def compute_joint_density_from_canonical_conditional_probability_densities(
        cpds):
    r"""
    Compute the joint density from a factorization expressed as the
    products of conditional probability densities (CPDs).

    P(x1,x2,...,xn) = P(x1|x2)P(x2|x3)...P(xn)

    Each variable x1 is a multivariate normal with p_i variables. Total
    number of variables in the joint distribution is nvars = \sum_i p_i.

    Parameters
    ----------
    cpds : list (n)
         List of GaussianFactor

    Returns
    -------
    joint_density : GaussianFactor
        The joint density object
    """
    nfactors = len(cpds)
    joint_density = copy.deepcopy(cpds[0])
    for ii in range(1, nfactors):
        joint_density *= cpds[ii]

    return joint_density


class GaussianFactor(object):
    r"""
    A Gaussian random variable in compact canonical form

    p(x|h,K) = exp(g+h^T x-0.5 x^T K x) x \in R^n.

    where K is precision matrix and

    g = -0.5 m^T h -0.5 n\log(2\pi) +0.5 \log |K|,

    and

    h = K m

    where m is the mean of the Gaussian distribution
    """

    def __init__(self, precision_matrix, shift, normalization, var_ids,
                 nvars_per_var):
        r"""
        Initialize the PDF.

        Parameters
        ----------
        precision matrix : np.ndarray (nvars,nvars)
             The precision matrix (K), i.e. the inverse of the covariance matrix

        shift : np.ndarray (nvars)
             The shift (h) of the distribution

        normalization : float
             The normalization factor (g) of the canoical form.
        """
        self._initialize(
            precision_matrix, shift, normalization, var_ids, nvars_per_var)

    def _initialize(self, precision_matrix, shift, normalization, var_ids,
                    nvars_per_var):
        assert np.asarray(nvars_per_var).ndim == 1
        assert shift.shape[0] == precision_matrix.shape[0]
        assert np.sum(nvars_per_var) == precision_matrix.shape[0]
        assert len(nvars_per_var) == len(var_ids)

        self.precision_matrix = precision_matrix
        self.shift = shift
        self.normalization = normalization

        self.var_ids = np.asarray(var_ids)
        self.nvars_per_var = np.asarray(nvars_per_var)

        assert self.nvars_per_var.sum() == self.precision_matrix.shape[0]
        assert self.shift.shape[0] == self.precision_matrix.shape[0]
        assert self.var_ids.shape[0] == self.nvars_per_var.shape[0]

    def __str__(self):
        return 'Factor'+str([(self.var_ids[ii], self.nvars_per_var[ii])
                             for ii in range(len(self.var_ids))])

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.precision_matrix.shape[0]

    def __call__(self, samples):
        r"""
        Evaluate the PDF at a set of samples.
        """
        assert samples.ndim == 2
        assert samples.shape[0] == self.precision_matrix.shape[0]

        exponent = np.empty(samples.shape[1])
        for ii in range(samples.shape[1]):
            exponent[ii] = -0.5*np.dot(samples[:, ii].T, self.precision_matrix.dot(
                samples[:, ii]))+np.dot(self.shift.T, samples[:, ii]) +\
                self.normalization
        return np.exp(exponent)

    def _multiply(self, gauss1, gauss2):
        precision_matrix, shift, normalization, var_ids, nvars_per_var = \
            multiply_gaussian_densities_in_compact_canonical_form(
                gauss1.precision_matrix, gauss1.shift, gauss1.normalization,
                gauss1.var_ids, gauss1.nvars_per_var,
                gauss2.precision_matrix, gauss2.shift, gauss2.normalization,
                gauss2.var_ids, gauss2.nvars_per_var)
        gauss1._initialize(precision_matrix, shift, normalization, var_ids,
                           nvars_per_var)

    def __mul__(self, other):
        new = copy.deepcopy(self)
        self._multiply(new, other)
        return new

    def __imul__(self, other):
        self._multiply(self, other)
        return self

    def marginalize(self, marg_ids):
        marg_indices, __ = get_matrix_partition_indices(
            marg_ids, self.var_ids, self.nvars_per_var)
        mask = np.asarray(
            [(self.var_ids[ii] in marg_ids) for ii in range(len(self.var_ids))])
        precision_matrix, shift, normalization = \
            marginalize_gaussian_in_canonical_form(
                marg_indices, self.precision_matrix,
                self.shift, self.normalization, remain_indices=None)
        self._initialize(
            precision_matrix, shift, normalization, self.var_ids[~mask],
            self.nvars_per_var[~mask])

    def condition(self, data_ids, data):
        # Find array indices of any data_ids found in self.var_ids.
        relevant_data_ids, relevant_data_indices, relevant_var_indices = \
            np.intersect1d(data_ids, self.var_ids, return_indices=True)

        if relevant_var_indices.shape[0] == 0:
            # No data is relevant to this factor
            return

        # Select data associated with data_ids found in self.var_ids
        relevant_data = data[relevant_data_indices]
        # Get array indices of relevant_data_ids as they appear in self.var_ids
        remove_indices = np.sort(relevant_var_indices)
        # Get array indices of self.var_ids that will remain after conditioning
        remain_indices = np.setdiff1d(
            np.arange(len(self.var_ids)), remove_indices)
        # Get indices of rows in precision matrix that correspond to data blocks
        remove_matrix_indices = subselect_matrix_blocks(
            remove_indices, self.nvars_per_var)

        precision_matrix, shift, normalization = \
            condition_gaussian_in_canonical_form(
                remove_matrix_indices, self.precision_matrix, self.shift,
                self.normalization, relevant_data, remain_indices=None)

        self._initialize(
            precision_matrix, shift, normalization, self.var_ids[remain_indices],
            self.nvars_per_var[remain_indices])
