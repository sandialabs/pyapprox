import numpy as np
from scipy.linalg import solve_triangular
from sklearn.linear_model import (
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV
)

from pyapprox.surrogates.orthopoly.leja_sequences import christoffel_weights
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion,
    define_poly_options_from_variable_transformation
)
from pyapprox.surrogates.polychaos.induced_sampling import (
    increment_induced_samples_migliorati,
    generate_induced_samples_migliorati_tolerance,
    compute_preconditioned_canonical_basis_matrix_condition_number
)
from pyapprox.util.utilities import hash_array
from pyapprox.util.linalg import (
    add_columns_to_pivoted_lu_factorization,
    continue_pivoted_lu_factorization,
    get_final_pivots_from_sequential_pivots,
    split_lu_factorization_matrix,
    pivot_rows,
    truncated_pivoted_lu_factorization, unprecondition_LU_factor
)
from pyapprox.surrogates.interp.adaptive_sparse_grid import (
    SubSpaceRefinementManager
)
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)


def get_subspace_active_poly_array_indices(adaptive_pce, ii):
    idx1 = adaptive_pce.unique_poly_indices_idx[ii]
    if ii < adaptive_pce.unique_poly_indices_idx.shape[0]-1:
        idx2 = adaptive_pce.unique_poly_indices_idx[ii+1]
    else:
        idx2 = adaptive_pce.poly_indices.shape[1]
    return np.arange(idx1, idx2)


def get_active_poly_array_indices(adaptive_pce):
    indices = np.empty((0), dtype=int)
    for key, ii in adaptive_pce.active_subspace_indices_dict.items():
        subspace_array_indices = get_subspace_active_poly_array_indices(
            adaptive_pce, ii)
        indices = np.hstack([indices, subspace_array_indices])
    return indices


def variance_pce_refinement_indicator(
        subspace_index, num_new_subspace_samples, adaptive_pce,
        normalize=True, mean_only=False):
    """
    Set pce coefficients of new subspace poly indices to zero to compute
    previous mean then set them to be non-zero
    """
    key = hash_array(subspace_index)
    ii = adaptive_pce.active_subspace_indices_dict[key]
    II = get_subspace_active_poly_array_indices(adaptive_pce, ii)
    error = np.sum(adaptive_pce.pce.coefficients[II]**2, axis=0)
    indicator = error.copy()

    if normalize:
        msg = """Attempted normalization of variance with values at first grid point close to 0.
        Possible options are:
        - Use a different sample or sampling sequence
          (if random sampling, try a different seed value)
        - Set the `normalize` option to False (see docs for: `variance_pce_refinement_indicator()`)
        """
        assert np.all(np.absolute(adaptive_pce.values[0, :]) > 1e-6), msg
        indicator /= np.absolute(adaptive_pce.values[0, :])**2

    qoi_chosen = np.argmax(indicator)
    indicator = indicator.max()

    cost_per_sample = adaptive_pce.eval_cost_function(
        subspace_index[:, np.newaxis])
    cost = cost_per_sample*num_new_subspace_samples

    # compute marginal benefit
    indicator /= cost

    return -indicator, error[qoi_chosen]


def solve_preconditioned_least_squares(basis_matrix_func, samples, values,
                                       precond_func):
    basis_matrix = basis_matrix_func(samples)
    weights = precond_func(basis_matrix, samples)
    basis_matrix = basis_matrix*weights[:, np.newaxis]
    rhs = values*weights[:, np.newaxis]
    coef = np.linalg.lstsq(basis_matrix, rhs, rcond=None)[0]
    return coef


def solve_preconditioned_orthogonal_matching_pursuit(basis_matrix_func,
                                                     samples, values,
                                                     precond_func,
                                                     tol=1e-8):
    basis_matrix = basis_matrix_func(samples)
    weights = precond_func(basis_matrix, samples)
    basis_matrix = basis_matrix*weights[:, np.newaxis]
    rhs = values*weights[:, np.newaxis]
    if basis_matrix.shape[1] == 1 or tol > 0:
        omp = OrthogonalMatchingPursuit(tol=tol)
    else:
        omp = OrthogonalMatchingPursuitCV(cv=min(samples.shape[1], 10))
    res = omp.fit(basis_matrix, rhs)
    coef = omp.coef_
    coef[0] += res.intercept_
    return coef[:, np.newaxis]


def christoffel_preconditioning_function(basis_matrix, samples):
    weights = np.sqrt(basis_matrix.shape[1]*christoffel_weights(basis_matrix))
    return weights


def generate_probability_samples_tolerance(
        pce, nindices, cond_tol, samples=None,
        verbosity=0):
    r"""
    Add samples in integer increments of nindices.
    E.g. if try nsamples = nindices, 2*nindices, 3*nindices
    until condition number is less than tolerance.

    Parameters
    ----------
    samples : np.ndarray
        samples in the canonical domain of the polynomial

    Returns
    -------
    new_samples : np.ndarray(nvars, nnew_samples)
        New samples appended to samples. must be in canonical space
    """
    variable = pce.var_trans.variable
    if samples is None:
        new_samples = generate_independent_random_samples(
            variable, nindices)
        new_samples = pce.var_trans.map_to_canonical(new_samples)
    else:
        new_samples = samples.copy()
    cond = compute_preconditioned_canonical_basis_matrix_condition_number(
        pce, new_samples)
    if verbosity > 0:
        print('\tCond No.', cond, 'No. samples', new_samples.shape[1],
              'cond tol', cond_tol)
    cnt = 1
    max_nsamples = 1000*pce.indices.shape[1]
    while cond > cond_tol:
        tmp = generate_independent_random_samples(variable, cnt*nindices)
        tmp = pce.var_trans.map_to_canonical(tmp)
        new_samples = np.hstack((new_samples, tmp))
        cond = compute_preconditioned_canonical_basis_matrix_condition_number(
            pce, new_samples)
        if verbosity > 0:
            print('\tCond No.', cond, 'No. samples', new_samples.shape[1],
                  'cond tol', cond_tol)
        # double number of samples added so loop does not take to long
        cnt *= 2
        if new_samples.shape[1] > max_nsamples:
            msg = "Basis and sample combination is ill conditioned"
            raise RuntimeError(msg)
    return new_samples


def increment_probability_samples(pce, cond_tol, samples, indices,
                                  new_indices, verbosity=0):
    r"""
    Parameters
    ----------
    samples : np.ndarray
        samples in the canonical domain of the polynomial

    Returns
    -------
    new_samples : np.ndarray(nvars, nnew_samples)
        New samples appended to samples. must be in canonical space
    """
    # allocate at one sample for every new basis
    tmp = generate_independent_random_samples(
        pce.var_trans.variable, new_indices.shape[1])
    tmp = pce.var_trans.map_to_canonical(tmp)
    new_samples = np.hstack((samples, tmp))
    # keep sampling until condition number is below cond_tol
    new_samples = generate_probability_samples_tolerance(
        pce, new_indices.shape[1], cond_tol, new_samples, verbosity)
    if verbosity > 0:
        print('No. samples', new_samples.shape[1])
        print('No. initial samples', samples.shape[1])
        print('No. indices', indices.shape[1], pce.indices.shape[1])
        print('No. new indices', new_indices.shape[1])
        print('No. new samples',
              new_samples.shape[1]-samples.shape[1])
    return new_samples


class AdaptiveInducedPCE(SubSpaceRefinementManager):
    """
    An adaptive PCE built using induced sampling and generalized sparse grid
    like refinement.
    """
    def __init__(self, num_vars, cond_tol=1e2, induced_sampling=True,
                 fit_opts={'omp_tol': 0}):
        """
        num_vars : integer
            The number of random variables

        cond_tol : float
            The target condition number of the basis matrix used for regression

        induced_sampling : boolean
            True - use induced sampling
            False - use random sampling

        fit_opts : dict
            Options used to solve the regression problem at each step of the
            adaptive algorithm
        """

        super(AdaptiveInducedPCE, self).__init__(num_vars)
        self.cond_tol = cond_tol
        self.fit_opts = fit_opts
        self.set_preconditioning_function(christoffel_preconditioning_function)
        self.fit_function = self._fit
        self.induced_sampling = induced_sampling
        assert abs(cond_tol) > 1
        if not induced_sampling:
            self.set_preconditioning_function(
                precond_func=lambda m, x: np.ones(x.shape[1]))

        self.moments = None

    def set_function(self, function, var_trans=None, pce=None):
        super(AdaptiveInducedPCE, self).set_function(function, var_trans)
        self.set_polynomial_chaos_expansion(pce)

    def set_polynomial_chaos_expansion(self, pce=None):
        if pce is None:
            poly_opts = define_poly_options_from_variable_transformation(
                self.var_trans)
            self.pce = PolynomialChaosExpansion()
            self.pce.configure(poly_opts)
        else:
            self.pce = pce

    def increment_samples(self, current_poly_indices, unique_poly_indices):
        if self.induced_sampling:
            return increment_induced_samples_migliorati(
                self.pce, self.cond_tol, self.samples,
                current_poly_indices, unique_poly_indices)
        if self.cond_tol < 0:
            sample_ratio = -self.cond_tol
            samples = generate_independent_random_samples(
                self.pce.var_trans.variable,
                sample_ratio*unique_poly_indices.shape[1])
            samples = self.pce.var_trans.map_to_canonical(samples)
            samples = np.hstack([self.samples, samples])
            return samples

        return increment_probability_samples(
                self.pce, self.cond_tol, self.samples,
                current_poly_indices, unique_poly_indices)

    def allocate_initial_samples(self):
        if self.induced_sampling:
            return generate_induced_samples_migliorati_tolerance(
                self.pce, self.cond_tol)

        if self.cond_tol < 0:
            sample_ratio = -self.cond_tol
            return generate_independent_random_samples(
                self.pce.var_trans.variable,
                sample_ratio*self.pce.num_terms())

        return generate_probability_samples_tolerance(
                self.pce, self.pce.num_terms(),
                self.cond_tol)

    def create_new_subspaces_data(self, new_subspace_indices):
        num_current_subspaces = self.subspace_indices.shape[1]
        self.initialize_subspaces(new_subspace_indices)

        self.pce.set_indices(self.poly_indices)
        if self.samples.shape[1] == 0:
            unique_subspace_samples = self.allocate_initial_samples()
            return unique_subspace_samples, np.array(
                [unique_subspace_samples.shape[1]])

        num_vars, num_new_subspaces = new_subspace_indices.shape
        unique_poly_indices = np.zeros((num_vars, 0), dtype=int)
        for ii in range(num_new_subspaces):
            II = get_subspace_active_poly_array_indices(
                self, num_current_subspaces+ii)
            unique_poly_indices = np.hstack(
                [unique_poly_indices, self.poly_indices[:, II]])

        # Current_poly_indices will include active indices not added
        # during this call, i.e. in new_subspace_indices.
        # thus cannot use
        # II = get_active_poly_array_indices(self)
        # unique_poly_indices = self.poly_indices[:,II]
        # to replace above loop
        current_poly_indices = self.poly_indices[
            :, :self.unique_poly_indices_idx[num_current_subspaces]]
        num_samples = self.samples.shape[1]
        samples = self.increment_samples(
            current_poly_indices, unique_poly_indices)
        unique_subspace_samples = samples[:, num_samples:]

        # warning num_new_subspace_samples does not really make sense for
        # induced sampling as new samples are not directly tied to newly
        # added basis
        num_new_subspace_samples = unique_subspace_samples.shape[1]*np.ones(
            new_subspace_indices.shape[1])//new_subspace_indices.shape[1]
        return unique_subspace_samples, num_new_subspace_samples

    def _fit(self, pce, canonical_basis_matrix, samples, values,
             precond_func=None, omp_tol=0):
        # do to, just add columns to stored basis matrix
        # store qr factorization of basis_matrix and update the factorization
        # self.samples are in canonical domain
        if omp_tol == 0:
            coef = solve_preconditioned_least_squares(
                canonical_basis_matrix, samples, values, precond_func)
        else:
            coef = solve_preconditioned_orthogonal_matching_pursuit(
                canonical_basis_matrix, samples, values, precond_func, omp_tol)
        self.pce.set_coefficients(coef)

    def fit(self):
        return self.fit_function(
            self.pce, self.pce.canonical_basis_matrix, self.samples,
            self.values, **self.fit_opts)

    def add_new_subspaces(self, new_subspace_indices):
        num_new_subspace_samples = super(
            AdaptiveInducedPCE, self).add_new_subspaces(new_subspace_indices)

        self.fit()

        return num_new_subspace_samples

    def __call__(self, samples, return_grad=False):
        return self.pce(samples, return_grad)

    def get_active_unique_poly_indices(self):
        II = get_active_poly_array_indices(self)
        return self.poly_indices[:, II]

    def set_preconditioning_function(self, precond_func):
        """
        precond_func : callable
            Callable function with signature precond_func(basis_matrix,samples)
        """
        self.precond_func = precond_func
        self.fit_opts['precond_func'] = self.precond_func

    def num_training_samples(self):
        return self.samples.shape[1]

    def build(self, callback=None):
        """
        """
        while (not self.active_subspace_queue.empty() or
               self.subspace_indices.shape[1] == 0):
            self.refine()
            self.recompute_active_subspace_priorities()

            if callback is not None:
                callback(self)


class AdaptiveLejaPCE(AdaptiveInducedPCE):
    """
    An adaptive PCE built using multivariate Leja sequences and 
    generalized sparse grid like refinement.
    """
    def __init__(self, num_vars, candidate_samples, factorization_type='fast'):
        """
        num_vars : integer
            The number of random variables

        candidate_samples : np.ndarray (num_vars, ncandidate_samples)
            The candidate samples from which the leja sequence is selected

        factorization_type : string
            fast - update LU factorization at each step
            slow - recompute LU factorization at each step
        """

        super(AdaptiveLejaPCE, self).__init__(num_vars, 1e6)

        # Make sure correct preconditioning function is used.
        # AdaptiveInducedPCE has some internal logic that can overide default
        # we want
        self.set_preconditioning_function(christoffel_preconditioning_function)

        # Must be in canonical space
        # TODO: generate candidate samples at each iteration from induced
        # distribution using current self.poly_indices
        self.candidate_samples = candidate_samples
        self.factorization_type = factorization_type

    def precond_canonical_basis_matrix(self, samples):
        basis_matrix = self.pce.canonical_basis_matrix(samples)
        precond_weights = self.precond_func(basis_matrix, samples)
        precond_basis_matrix = basis_matrix*precond_weights[:, np.newaxis]

        return precond_basis_matrix, precond_weights

    def get_num_new_subspace_samples(self, new_subspace_indices,
                                     num_current_subspaces):
        num_current_subspaces = self.subspace_indices.shape[1]
        num_vars, num_new_subspaces = new_subspace_indices.shape

        num_new_subspace_samples = np.empty((num_new_subspaces), dtype=int)
        for ii in range(num_new_subspaces):
            II = get_subspace_active_poly_array_indices(
                self, num_current_subspaces+ii)
            num_new_subspace_samples[ii] = II.shape[0]
        return num_new_subspace_samples

    def condition_number(self):
        if self.factorization_type == 'slow':
            return np.linalg.cond(self.L_factor.dot(self.U_factor))
        else:
            L, U = split_lu_factorization_matrix(
                self.LU_factor, num_pivots=self.samples.shape[1])
            return np.linalg.cond(L.dot(U))

    def update_leja_sequence_slow(self, new_subspace_indices):
        num_samples = self.samples.shape[1]

        # There will be two copies of self.samples in candidate_samples
        # but pivoting will only choose these samples once when number of
        # desired samples is smaller than
        # self.candidate_samples.shape[0]-self.samples.shape[1]
        candidate_samples = np.hstack([self.samples, self.candidate_samples])

        self.pce.set_indices(self.poly_indices)
        precond_basis_matrix, precond_weights = \
            self.precond_canonical_basis_matrix(candidate_samples)

        # TODO: update LU factorization using new candidate points, This
        # requires writing a function that updates not just new columns of
        # L and U factor but also allows new rows to be added.
        max_iters = self.poly_indices.shape[1]
        num_initial_rows = num_samples
        self.L_factor, self.U_factor, pivots =\
            truncated_pivoted_lu_factorization(
                precond_basis_matrix, max_iters,
                num_initial_rows=num_initial_rows)
        self.pivots = np.arange(num_samples)[pivots[:num_initial_rows]]
        self.pivots = np.concatenate(
            [self.pivots, np.arange(num_initial_rows, pivots.shape[0])])
        self.precond_weights = precond_weights[pivots, np.newaxis]

        return candidate_samples[:, pivots[num_samples:]]

    def update_leja_sequence_fast(self, new_subspace_indices,
                                  num_current_subspaces):
        num_samples = self.samples.shape[1]
        if num_samples == 0:
            self.pce.set_indices(self.poly_indices)
            max_iters = self.poly_indices.shape[1]

            # Keep unconditioned
            self.basis_matrix = self.precond_canonical_basis_matrix(
                self.candidate_samples)[0]
            self.LU_factor, self.seq_pivots = \
                truncated_pivoted_lu_factorization(
                    self.basis_matrix, max_iters, truncate_L_factor=False)
            self.pivots = get_final_pivots_from_sequential_pivots(
                self.seq_pivots.copy())[:max_iters]
            self.precond_weights = self.precond_func(
                self.basis_matrix, self.candidate_samples)[:, np.newaxis]

            return self.candidate_samples[
                :, self.pivots[num_samples:self.poly_indices.shape[1]]]

        num_vars, num_new_subspaces = new_subspace_indices.shape
        unique_poly_indices = np.zeros((num_vars, 0), dtype=int)
        for ii in range(num_new_subspaces):
            II = get_subspace_active_poly_array_indices(
                self, num_current_subspaces+ii)
            unique_poly_indices = np.hstack(
                [unique_poly_indices, self.poly_indices[:, II]])
        self.pce.set_indices(unique_poly_indices)

        precond_weights_prev = self.precond_weights
        pivoted_precond_weights_prev = pivot_rows(
            self.seq_pivots, precond_weights_prev, False)

        new_cols = self.pce.canonical_basis_matrix(self.candidate_samples)
        self.basis_matrix = np.hstack([self.basis_matrix, np.array(new_cols)])
        new_cols *= precond_weights_prev
        self.LU_factor = add_columns_to_pivoted_lu_factorization(
            np.array(self.LU_factor), new_cols, self.seq_pivots[:num_samples])

        self.precond_weights = self.precond_func(
            self.basis_matrix, self.candidate_samples)[:, np.newaxis]
        pivoted_precond_weights = pivot_rows(
            self.seq_pivots, self.precond_weights, False)

        self.LU_factor = unprecondition_LU_factor(
            self.LU_factor,
            pivoted_precond_weights_prev/pivoted_precond_weights,
            num_samples)

        max_iters = self.poly_indices.shape[1]
        self.LU_factor, self.seq_pivots, _ = continue_pivoted_lu_factorization(
            self.LU_factor.copy(), self.seq_pivots, self.samples.shape[1],
            max_iters, num_initial_rows=0)
        self.pivots = get_final_pivots_from_sequential_pivots(
            self.seq_pivots.copy())[:max_iters]
        self.pce.set_indices(self.poly_indices)

        return self.candidate_samples[
            :, self.pivots[num_samples:self.poly_indices.shape[1]]]

    def create_new_subspaces_data(self, new_subspace_indices):
        num_current_subspaces = self.subspace_indices.shape[1]
        self.initialize_subspaces(new_subspace_indices)

        if self.factorization_type == 'fast':
            unique_subspace_samples = self.update_leja_sequence_fast(
                new_subspace_indices, num_current_subspaces)
        else:
            unique_subspace_samples = self.update_leja_sequence_slow(
                new_subspace_indices)

        num_new_subspace_samples = self.get_num_new_subspace_samples(
            new_subspace_indices, num_current_subspaces)

        return unique_subspace_samples, num_new_subspace_samples

    def add_new_subspaces(self, new_subspace_indices):
        num_new_subspace_samples = super(
            AdaptiveInducedPCE, self).add_new_subspaces(new_subspace_indices)

        if self.factorization_type == 'fast':
            it = self.samples.shape[1]
            temp = solve_triangular(
                self.LU_factor[:it, :it],
                self.values*self.precond_weights[self.pivots],
                lower=True, unit_diagonal=True)
            a_f = self.LU_factor[:it, :it]
        else:
            temp = solve_triangular(
                self.L_factor,
                self.values[self.pivots]*self.precond_weights,
                lower=True)
            a_f = self.U_factor

        coef = solve_triangular(a_f, temp, lower=False)
        self.pce.set_coefficients(coef)

        return num_new_subspace_samples

    def __call__(self, samples, return_grad=False):
        return self.pce(samples, return_grad)

    def get_active_unique_poly_indices(self):
        II = get_active_poly_array_indices(self)
        return self.poly_indices[:, II]
