import numpy as np


def swap_cols(A, col1, col2):
    if (col1 == col2):
        return A

    tmp = A[:, col1].copy()
    A[:, col1] = A[:, col2]
    A[:, col2] = tmp
    return A


def swap_rows(A, row1, row2):
    if (row1 == row2):
        return A

    tmp = A[row1, :].copy()
    A[row1, :] = A[row2, :]
    A[row2, :] = tmp
    return A


def swap_entries(vec, ii, jj):
    if ii == jj:
        return vec
    tmp = vec[ii]
    vec[ii] = vec[jj]
    vec[jj] = tmp
    return vec


def get_block_diagonal_matrix_num_rows(matrix_blocks):
    num_rows = 0
    for ii in range(len(matrix_blocks)):
        num_rows += matrix_blocks[ii].shape[0]
    return num_rows


def get_block_diagonal_matrix_num_cols(matrix_blocks):
    num_cols = 0
    for ii in range(len(matrix_blocks)):
        num_cols += matrix_blocks[ii].shape[1]
    return num_cols


def pre_multiply_block_diagonal_matrix(matrix, matrix_blocks, block_trans):
    num_blocks = len(matrix_blocks)
    if (block_trans == True):
        block_num_cols = get_block_diagonal_matrix_num_rows(matrix_blocks)
        result_num_rows = get_block_diagonal_matrix_num_cols(matrix_blocks)
    else:
        block_num_cols = get_block_diagonal_matrix_num_cols(matrix_blocks)
        result_num_rows = get_block_diagonal_matrix_num_rows(matrix_blocks)

    if (block_num_cols != matrix.shape[0]):
        msg = "pre_multiply_block_diagonal_matrix() Matrices sizes are "
        msg += "inconsistent"
        raise Exception(msg)

    result = np.empty((result_num_rows, matrix.shape[1]), dtype=float)
    sub_matrix_start_row = 0
    sub_result_start_row = 0
    for ii in range(num_blocks):
        if (block_trans == True):
            matrix_block_view = matrix_blocks[ii].T
        else:
            matrix_block_view = matrix_blocks[ii]
        num_block_rows = matrix_block_view.shape[0]
        num_block_cols = matrix_block_view.shape[1]
        num_submatrix_rows = num_block_cols
        sub_matrix = matrix[
            sub_matrix_start_row:sub_matrix_start_row+num_submatrix_rows, :]
        result[sub_result_start_row:sub_result_start_row+num_block_rows, :] =\
            np.dot(matrix_block_view, sub_matrix)
        sub_matrix_start_row += num_submatrix_rows
        sub_result_start_row += num_block_rows
    return result


def get_dense_block_diagonal_matrix(matrix_blocks):
    num_rows = get_block_diagonal_matrix_num_rows(matrix_blocks)
    num_cols = get_block_diagonal_matrix_num_cols(matrix_blocks)
    result = np.zeros((num_rows, num_cols), dtype=float)
    row_cnt = 0
    col_cnt = 0
    num_blocks = len(matrix_blocks)
    for ii in range(num_blocks):
        num_block_rows, num_block_cols = matrix_blocks[ii].shape
        result[row_cnt:row_cnt+num_block_rows, col_cnt:col_cnt+num_block_cols] =\
            matrix_blocks[ii]
        row_cnt += num_block_rows
        col_cnt += num_block_cols
    return result


def row_reduce_degree_vandermonde_matrix(degree_vandermonde, L_factor, lu_row,
                                         update_degree_specific_data_flag):
    """
    If we are working with a new degree we must orthogonalise against
    all points (min_q=0). Otherwise we can only have to update
    the orthogonalisation for the new row (min_q=lu_row-1).
    """

    num_rows, num_cols = degree_vandermonde.shape

    min_q = 0
    if (not update_degree_specific_data_flag):
        min_q = lu_row-1
    for qq in range(min_q, lu_row):
        if ((qq < lu_row-1) or (update_degree_specific_data_flag)):
            degree_vandermonde[qq, :] /= L_factor[qq, qq]
        degree_vandermonde[qq+1:, :] -= np.dot(
            L_factor[qq+1:, qq:qq+1], degree_vandermonde[qq:qq+1, :])

    return degree_vandermonde


def get_degree_basis_indices(num_vars, current_degree, degree_block_num,
                             update_degree_list, basis_degrees,
                             generate_degree_basis_indices, verbosity):

    prev_degree = current_degree
    current_degree, new_indices = generate_degree_basis_indices(
        num_vars, degree_block_num)

    if (update_degree_list):
        basis_degrees.append(prev_degree)
    else:
        new_indices = np.empty((num_vars, 0), dtype=int)

    return current_degree, new_indices, basis_degrees


def update_degree_specific_data(pts, permutations, selected_basis_indices,
                                new_indices, basis_cardinality, pce, lu_row,
                                H_factor_blocks, current_index_counter,
                                precond_weights):
    # Make sure new indices have correct array_indices.
    # This is necessary because I do not assume the basis indices
    # are ordered in anyway. I must ensure this is only done once
    # per degree
    selected_basis_indices = np.hstack((selected_basis_indices, new_indices))
    basis_cardinality.append(selected_basis_indices.shape[1])
    current_index_counter = 0

    # Build new degree vandermonde
    # The vandermonde only needs to be built once per degree.
    # It needs to be permuted everytime but this can be done at
    # the end of each iteration.
    pce.set_indices(new_indices)
    degree_vandermonde = pce.basis_matrix(pts)
    if (precond_weights is not None):
        degree_vandermonde = precondition_matrix(
            pts, precond_weights, degree_vandermonde)

    # TODO: Eventually make initial row size smaller then increment
    # memory when needed.
    current_block_num_initial_rows = min(
        degree_vandermonde.shape[1], pts.shape[1]-lu_row)
    current_H_factor_block = np.empty(
        (current_block_num_initial_rows, degree_vandermonde.shape[1]),
        dtype=float)
    H_factor_blocks.append(current_H_factor_block)
    current_block_num_rows = 0
    return degree_vandermonde, H_factor_blocks, current_H_factor_block, \
        current_block_num_rows, current_index_counter, selected_basis_indices


def compute_pivot_norms(degree_vandermonde, lu_row):
    # We only what the submatrix that contains the rows (points)
    # that have not already been chosen.
    sub_vand_trans = degree_vandermonde[lu_row:, :].T

    # Find the norms of each column
    norms = np.linalg.norm(sub_vand_trans, axis=0)
    assert norms.shape[0] == sub_vand_trans.shape[1]
    return norms, sub_vand_trans


def find_next_best_index(norms, num_initial_pts_selected, num_initial_pts,
                         enforce_ordering_of_initial_points):
    # Find the column with the largest norm. Note evec is
    # defined 0,...,numTotalPts_-lu_row-1 for points
    # lu_row+1,...,numTotalPts_

    if (num_initial_pts_selected < num_initial_pts):
        if (enforce_ordering_of_initial_points):
            # Enforce the ordering of the initial points
            next_index = 0
        else:
            # Chose column with largest norm that corresponds to
            # a point in the initial point set.
            next_index = np.argmax(
                norms[:num_initial_pts-num_initial_pts_selected])
        num_initial_pts_selected += 1
    else:
        next_index = np.argmax(norms)
    return next_index, num_initial_pts_selected


def compute_inner_products(sub_vand_trans, norms, next_index):
    # Compute inner products of each column with the chosen column
    magic_row = sub_vand_trans[:, next_index] / norms[next_index]
    inner_products = np.dot(magic_row, sub_vand_trans)
    return inner_products, magic_row


def determine_if_low_rank(norms, next_index, degree_max_norm,
                          current_index_counter):
    if (current_index_counter == 0):
        degree_max_norm = norms[next_index]

    low_rank = False
    if ((current_index_counter != 0) and
            (norms[next_index] < 0.001*degree_max_norm)):
        low_rank = True

    return low_rank, degree_max_norm


def update_factorization(next_index, inner_products, norms, magic_row,
                         num_current_indices, permutations, lu_row,
                         permuted_pts,
                         L_factor, U_factor, degree_vandermonde,
                         H_factor_blocks, current_block_num_rows,
                         current_index_counter, points_to_degree_map,
                         basis_degrees, current_H_factor_block, limited_memory,
                         precond_weights, verbosity):

    # Update the LU permutations based
    permutations = swap_entries(permutations, lu_row, lu_row+next_index)

    # Update the premuted pts
    permuted_pts = swap_cols(permuted_pts, lu_row, lu_row+next_index)

    if (precond_weights is not None):
        # Update the precondition weights
        # Todo if I make preconditioing degree dependent then I do not need
        # to swap precondWeights as they are only applied once at the begining
        # of each degree
        precond_weights = swap_entries(
            precond_weights, lu_row, lu_row+next_index)

    # Update the L factor of the LU factorization to be consistent
    # with the new permutations
    l_sub = L_factor[lu_row:, :lu_row]
    if ((l_sub.shape[0] > 0) and (l_sub.shape[1] > 0)):
        L_factor[lu_row:, :lu_row] = swap_rows(l_sub, 0, next_index)

    # Update L_factor with inner products
    inner_products = swap_entries(inner_products, 0, next_index)
    inner_products[0] = norms[next_index]
    # the following line accounts for 50% of runtime for large
    # number of candidate samples
    L_factor[lu_row:, lu_row] = inner_products

    # Update U. That is enforce orthogonality to all
    # rows with indices < lu_row
    # To do this we must find the inner products of all the other
    # rows above the current row in degreeVandermonde_
    if (lu_row > 0):
        U_factor[:lu_row, lu_row] = np.dot(
            degree_vandermonde[:lu_row, :], magic_row.T)

    # Update the non-zero entries of the H matrix. Essentially these
    # entries are the directions needed to orthogonalise the entries
    # (basis blocks) in the LU factorization
    current_H_factor_block[current_block_num_rows, :] = magic_row
    H_factor_blocks[-1] = \
        current_H_factor_block[:current_block_num_rows+1, :].copy()
    current_block_num_rows += 1
    current_index_counter += 1

    if (current_index_counter >= num_current_indices):
        update_degree_specific_data_flag = True
    else:
        sub_vand = degree_vandermonde[lu_row:lu_row +
                                      inner_products.shape[0]+1, :]
        degree_vandermonde[lu_row:lu_row+inner_products.shape[0]+1, :] = \
            swap_rows(sub_vand, 0, next_index)
        update_degree_specific_data_flag = False

    if (verbosity > 2):
        print(("Iteration: ", lu_row+1))
        print("\t Adding point:")
        print((permuted_pts[:, lu_row]))

    points_to_degree_map.append(basis_degrees[-1])
    lu_row += 1

    return num_current_indices, permutations, lu_row, permuted_pts,\
        L_factor, U_factor, degree_vandermonde,\
        H_factor_blocks, current_block_num_rows, current_index_counter,\
        points_to_degree_map, basis_degrees, current_H_factor_block,\
        update_degree_specific_data_flag, precond_weights


def least_factorization_sequential_update(
        permuted_pts, lu_row, current_degree, current_degree_basis_indices,
        H_factor_blocks, update_degree_specific_data_flag, basis_degrees,
        generate_degree_basis_indices, selected_basis_indices, permutations,
        basis_cardinality, pce, L_factor, U_factor, num_initial_pts_selected,
        num_initial_pts, current_index_counter,
        points_to_degree_map, limited_memory, row_reduced_vandermonde_blocks,
        assume_non_degeneracy, precond_weights, degree_vandermonde,
        degree_max_norm, current_block_num_rows,
        current_H_factor_block, enforce_all_initial_points_used,
        enforce_ordering_of_initial_points, initial_pts_degenerate,
        points_to_num_indices_map, verbosity):

    num_vars = permuted_pts.shape[0]

    if (lu_row >= permuted_pts.shape[1]):
        msg = "least_factorization_sequential_update() "
        msg += "Cannot proceed: all points have been added to the interpolant"
        raise Exception(msg)

    # Get the number of basis terms with degree equal to current_degree
    if update_degree_specific_data_flag:
        current_degree, current_degree_basis_indices, basis_degrees = \
            get_degree_basis_indices(
                num_vars, current_degree, len(
                    H_factor_blocks), True, basis_degrees,
                generate_degree_basis_indices, verbosity)

    if (update_degree_specific_data_flag and verbosity > 1):
        print(("Incrementing degree to ",  current_degree))
        print(("\tCurrent number of points ",  lu_row+1))
        print(("\tCurrent number of terms ", selected_basis_indices.shape[1]))
        print(("\tNew number of terms ", selected_basis_indices.shape[1] +
               current_degree_basis_indices.shape[1]))

    # Determine the number of indices of degree current_degree
    num_current_indices = current_degree_basis_indices.shape[1]

    # If there exists any indices in the pce basis with degree equal to
    # degree counter then attempt to use these indices to interpolate some
    # of the data
    if (num_current_indices > 0):

        # Update all the objects and other data structures that must
        # be changes wwhen the degree of the interpolant is increased.
        if (update_degree_specific_data_flag):
            degree_vandermonde, H_factor_blocks, current_H_factor_block, \
                current_block_num_rows, current_index_counter, \
                selected_basis_indices = \
                update_degree_specific_data(
                    permuted_pts, permutations, selected_basis_indices,
                    current_degree_basis_indices, basis_cardinality, pce,
                    lu_row,
                    H_factor_blocks, current_index_counter, precond_weights)

        # Row-reduce degreeVandermonde_ according to previous
        # elimination steps
        degree_vandermonde = row_reduce_degree_vandermonde_matrix(
            degree_vandermonde, L_factor, lu_row,
            update_degree_specific_data_flag)

        # Compute the pivots needed to update the LU factorization
        norms, sub_vand_trans = compute_pivot_norms(degree_vandermonde, lu_row)

        # Find the column of the degree_vandermonde with the largest norm.
        next_index, num_initial_pts_selected = \
            find_next_best_index(
                norms, num_initial_pts_selected, num_initial_pts,
                enforce_ordering_of_initial_points)

        low_rank, degree_max_norm = determine_if_low_rank(
            norms, next_index, degree_max_norm, current_index_counter)
        if ((low_rank) and ((num_initial_pts_selected < num_initial_pts))):
            initial_pts_degenerate = True
            if enforce_ordering_of_initial_points:
                msg = 'enforce_ordering_of_initial_points was set to True, '
                msg += 'initial points are degenerate'
                raise Exception(msg)

        if (not low_rank):
            # Compute the inner products necessary to update the LU
            # factorization
            inner_products, magic_row = compute_inner_products(
                sub_vand_trans, norms, next_index)

            # normalize pivot row in degreeVandermonde. The new row
            # has already been computed and stored in magic_row
            degree_vandermonde[lu_row+next_index, :] = magic_row

            num_current_indices, permutations, lu_row, permuted_pts,\
                L_factor, U_factor, degree_vandermonde,\
                H_factor_blocks, current_block_num_rows, \
                current_index_counter,\
                points_to_degree_map, basis_degrees, current_H_factor_block,\
                update_degree_specific_data_flag, precond_weights = \
                update_factorization(
                    next_index, inner_products, norms, magic_row,
                    num_current_indices, permutations, lu_row, permuted_pts,
                    L_factor, U_factor, degree_vandermonde,
                    H_factor_blocks, current_block_num_rows,
                    current_index_counter,
                    points_to_degree_map, basis_degrees,
                    current_H_factor_block,
                    limited_memory, precond_weights, verbosity)

            points_to_num_indices_map.append(selected_basis_indices.shape[1])

        else:
            update_degree_specific_data_flag = True

            # num_initial_pts_selected was incremented in find_next_best_index
            # but no point was actually added because point was low rank
            # so decrement counter here
            if ((num_initial_pts_selected <= num_initial_pts) and
                    (num_initial_pts_selected > 0)):
                num_initial_pts_selected -= 1

            if (assume_non_degeneracy):
                msg = "least_factorization_sequential_update() Factorization "
                msg += "of new points was requested but new points were "
                msg += "degenerate"
                raise Exception(msg)
            if (verbosity > 1):
                print(("Low rank at lu_row ", lu_row,))
                print(" incrementing degree counter")

        if ((low_rank) or (current_index_counter >= num_current_indices)):
            # If the next step will be on a higher degree (because low rank or
            # the degree block has been filled) then deep copy
            # current_H_factor_block to H_factor_blocks.
            # current_H_factor_block is overwritten when the degree
            # is increased. Copy (deep) the current degree block to H_factor
            H_factor_blocks[-1] = \
                current_H_factor_block[:current_block_num_rows, :].copy()
        else:
            update_degree_specific_data_flag = False

    if ((update_degree_specific_data_flag) and (not limited_memory)):
        # Store previous row_reduced vandermonde matrix
        row_reduced_vandermonde_blocks.append(degree_vandermonde[:lu_row, :])

    return permutations, lu_row, permuted_pts, L_factor, U_factor, \
        H_factor_blocks,\
        current_index_counter, points_to_degree_map, basis_degrees,\
        update_degree_specific_data_flag, selected_basis_indices, \
        current_degree_basis_indices, degree_vandermonde, \
        current_block_num_rows,\
        current_H_factor_block, degree_max_norm, num_initial_pts_selected,\
        initial_pts_degenerate, current_degree, points_to_num_indices_map


def least_factorization_sequential(
        pce, candidate_pts, generate_degree_basis_indices,
        initial_pts=None, num_pts=None, verbosity=3,
        preconditioning_function=False, assume_non_degeneracy=False,
        enforce_all_initial_points_used=False,
        enforce_ordering_of_initial_points=False):

    if num_pts is None:
        num_selected_pts = candidate_pts.shape[1]
    else:
        num_selected_pts = num_pts
    # --------------------------------------------------------------------- #
    #                          Initialization                               #
    # --------------------------------------------------------------------- #

    # Extract the basis indices of the pce. If non-zero these will be used
    # to interpolate the data
    # set_pce( pce );

    # must clear selected basis indices in case it was set previously
    selected_basis_indices = np.empty((candidate_pts.shape[0], 0), dtype=int)

    # must clear the H_factor
    H_factor_blocks = []

    update_degree_specific_data_flag = True

    if initial_pts is not None:
        assert initial_pts.shape[0] == candidate_pts.shape[0]
        assert num_selected_pts >= initial_pts.shape[1]
        permuted_pts = np.hstack((initial_pts, candidate_pts))
        num_initial_pts = initial_pts.shape[1]
    else:
        permuted_pts = candidate_pts
        num_initial_pts = 0
    assert num_selected_pts <= permuted_pts.shape[1]

    if (verbosity > 0):
        print("Least factorization: Choosing ", num_selected_pts)
        print(" points using ",
              permuted_pts.shape[1]-candidate_pts.shape[1])
        print(" initial points and ", candidate_pts.shape[1])
        print(" additional points\n")

    # Initialise the permutation for the LU factorization. Taking into
    # account any previous permutations
    permutations = np.arange(0, permuted_pts.shape[1], 1)

    # Initialize memory for the L and U factors
    # assumes LU factorization has already been computed for the initial pts
    # and is stored in L U factors
    lu_row = 0
    current_degree = 0
    current_degree_basis_indices = None
    num_initial_pts_selected = 0
    current_index_counter = 0
    limited_memory = False
    degree_vandermonde = None
    basis_degrees = []
    degree_max_norm = None
    current_block_num_rows = 0
    current_H_factor_block = None
    row_reduced_vandermonde_blocks = []

    L_factor = np.zeros((permuted_pts.shape[1], num_selected_pts), dtype=float)
    U_factor = np.zeros((permuted_pts.shape[1], num_selected_pts), dtype=float)
    for jj in range(L_factor.shape[1]):
        U_factor[jj, jj] = 1.

    # Initialize the factorization counters_
    # necessary when using repeated calls to least_factorization_sequential,
    # e.g. when using with cross validation
    basis_cardinality = []
    points_to_degree_map = []
    points_to_num_indices_map = []
    initial_pts_degenerate = False

    # Initialize preconditioning weights
    if (preconditioning_function is not None):
        precond_weights = preconditioning_function(permuted_pts)
    else:
        precond_weights = None

    # Current degree is current_degree, and we iterate on this
    while (lu_row < num_selected_pts):
        permutations, lu_row, permuted_pts, L_factor, U_factor, H_factor_blocks,\
            current_index_counter, points_to_degree_map, basis_degrees,\
            update_degree_specific_data_flag, selected_basis_indices,\
            current_degree_basis_indices, degree_vandermonde,\
            current_block_num_rows, current_H_factor_block, degree_max_norm,\
            num_initial_pts_selected, initial_pts_degenerate, current_degree,\
            points_to_num_indices_map =\
            least_factorization_sequential_update(
                permuted_pts, lu_row, current_degree,
                current_degree_basis_indices,
                H_factor_blocks, update_degree_specific_data_flag,
                basis_degrees,
                generate_degree_basis_indices, selected_basis_indices,
                permutations,
                basis_cardinality, pce, L_factor, U_factor,
                num_initial_pts_selected,
                num_initial_pts, current_index_counter,
                points_to_degree_map, limited_memory,
                row_reduced_vandermonde_blocks, assume_non_degeneracy,
                precond_weights, degree_vandermonde, degree_max_norm,
                current_block_num_rows, current_H_factor_block,
                enforce_all_initial_points_used,
                enforce_ordering_of_initial_points, initial_pts_degenerate,
                points_to_num_indices_map, verbosity)
    return permuted_pts, L_factor, U_factor, lu_row, H_factor_blocks, \
        selected_basis_indices, precond_weights


class LeastInterpolationSolver(object):
    def set_pce(self, pce):
        self.pce = pce

        self.preconditioning_function = None
        self.basis_generator = None

    def configure(self, opts):
        self.verbosity = opts.get('verbosity', 0)
        self.limited_memory = opts.get('use_limited_memory', 0)
        self.assume_non_degeneracy = opts.get('assume_non_degeneracy', True)
        self.enforce_all_initial_points_used = opts.get(
            'enforce_all_initial_points_used', False)
        self.enforce_ordering_of_initial_points = opts.get(
            'enforce_ordering_of_initial_points', False)
        if self.enforce_ordering_of_initial_points:
            self.enforce_all_initial_points_used = True

    def set_preconditioning_function(self, preconditioning_function):
        self.preconditioning_function = preconditioning_function

    def set_basis_generator(self, basis_generator):
        self.basis_generator = basis_generator

    def get_basis_generator(self):
        return self.basis_generator

    def get_permuted_points(self):
        return self.permuted_pts

    def get_last_point_added(self):
        return self.permuted_pts[:, self.lu_row-1]

    def get_current_LUH_factors(self):
        return self.L_factor[:self.lu_row, :self.lu_row], \
            self.U_factor[:self.lu_row, :self.lu_row], \
            get_dense_block_diagonal_matrix(self.H_factor_blocks)

    def get_initial_points_degenerate(self):
        return self.initial_pts_degenerate

    def num_points_added(self):
        return self.lu_row

    def initialize(self):
        self.selected_basis_indices = np.empty(
            (self.candidate_pts.shape[0], 0), dtype=int)

        self.H_factor_blocks = []

        self.update_degree_specific_data_flag = True

        # Initialise the permutation for the LU factorization. Taking into
        # account any previous permutations
        self.permutations = np.arange(0, self.permuted_pts.shape[1], 1)

        # Initialize memory for the L and U factors
        self.lu_row = 0
        self.current_degree = 0
        self.current_degree_basis_indices = None
        self.num_initial_pts_selected = 0
        self.current_index_counter = 0
        self.degree_vandermonde = None
        self.basis_degrees = []
        self.degree_max_norm = None
        self.current_block_num_rows = 0
        self.current_H_factor_block = None
        self.row_reduced_vandermonde_blocks = []

        # Do not set to num_selected_pts
        self.num_selected_pts = 0

        self.L_factor = np.zeros(
            (self.permuted_pts.shape[1], self.num_selected_pts), dtype=float)
        self.U_factor = np.zeros(
            (self.permuted_pts.shape[1], self.num_selected_pts), dtype=float)
        for jj in range(self.L_factor.shape[1]):
            self.U_factor[jj, jj] = 1.

        self.basis_cardinality = []
        self.points_to_degree_map = []
        self.points_to_num_indices_map = []
        self.initial_pts_degenerate = False

    def factorize(self, candidate_pts, initial_pts, num_selected_pts):

        self.candidate_pts = candidate_pts
        self.initial_pts = initial_pts
        if initial_pts is not None:
            assert initial_pts.shape[0] == candidate_pts.shape[0]
            assert num_selected_pts >= initial_pts.shape[1]
            self.permuted_pts = np.hstack((initial_pts, candidate_pts))
            self.num_initial_pts = initial_pts.shape[1]
        else:
            self.permuted_pts = candidate_pts
            self.num_initial_pts = 0
        assert num_selected_pts <= self.permuted_pts.shape[1]

        if (self.verbosity > 0):
            print("Least factorization: Choosing ", num_selected_pts)
            print(" points using ",
                  self.permuted_pts.shape[1]-self.candidate_pts.shape[1])
            print(" initial points and ", self.candidate_pts.shape[1])
            print(" additional points\n")

        self.initialize()

        # Initialize preconditioning weights
        if (self.preconditioning_function is not None):
            self.precond_weights = self.preconditioning_function(
                self.permuted_pts)
        else:
            self.precond_weights = None

        if self.basis_generator is None:
            raise Exception('call set_basis_generator()')

        # Current degree is current_degree, and we iterate on this
        self.update_factorization(num_selected_pts)

    def update_factorization(self, num_new_pts):
        self.num_selected_pts += num_new_pts
        if (self.L_factor.shape[1] <= self.lu_row):
            # Resize memory to allow for increment in num_selected_pts
            L_factor_new = np.zeros(
                (self.permuted_pts.shape[1], self.num_selected_pts))
            L_factor_new[:self.L_factor.shape[0], :self.L_factor.shape[1]] = \
                self.L_factor
            U_factor_new = np.zeros(
                (self.permuted_pts.shape[1], self.num_selected_pts))
            U_factor_new[:self.U_factor.shape[0], :self.U_factor.shape[1]] = \
                self.U_factor
            jj_min = min(L_factor_new.shape[0], L_factor_new.shape[1])
            for jj in range(self.lu_row, jj_min):
                U_factor_new[jj, jj] = 1.
            self.L_factor = L_factor_new
            self.U_factor = U_factor_new

        while (self.lu_row < self.num_selected_pts):
            self.update_factorization_step()

    def update_factorization_step(self):
        self.permutations, self.lu_row, self.permuted_pts, self.L_factor, \
            self.U_factor, self.H_factor_blocks, self.current_index_counter, \
            self.points_to_degree_map, self.basis_degrees, \
            self.update_degree_specific_data_flag, \
            self.selected_basis_indices, \
            self.current_degree_basis_indices, self.degree_vandermonde, \
            self.current_block_num_rows, self.current_H_factor_block, \
            self.degree_max_norm, self.num_initial_pts_selected, \
            self.initial_pts_degenerate, self.current_degree, \
            self.points_to_num_indices_map =\
            least_factorization_sequential_update(
                self.permuted_pts,
                self.lu_row,
                self.current_degree,
                self.current_degree_basis_indices,
                self.H_factor_blocks,
                self.update_degree_specific_data_flag,
                self.basis_degrees,
                self.basis_generator,
                self.selected_basis_indices,
                self.permutations,
                self.basis_cardinality,
                self.pce,
                self.L_factor,
                self.U_factor,
                self.num_initial_pts_selected,
                self.num_initial_pts,
                self.current_index_counter,
                self.points_to_degree_map,
                self.limited_memory,
                self.row_reduced_vandermonde_blocks,
                self.assume_non_degeneracy,
                self.precond_weights,
                self.degree_vandermonde,
                self.degree_max_norm,
                self.current_block_num_rows,
                self.current_H_factor_block,
                self.enforce_all_initial_points_used,
                self.enforce_ordering_of_initial_points,
                self.initial_pts_degenerate,
                self.points_to_num_indices_map,
                self.verbosity)

    def get_current_interpolant(self, permuted_samples, permuted_vals):
        assert permuted_vals.ndim == 2
        assert permuted_vals.shape[0] == permuted_samples.shape[1]

        return get_current_least_interpolant(
            permuted_samples, permuted_vals, self.pce, self.L_factor,
            self.U_factor, self.lu_row, self.H_factor_blocks,
            self.selected_basis_indices, self.precond_weights)

    def get_current_permutation(self):
        return self.permutations

    def get_points_to_degree_map(self):
        return self.points_to_degree_map

    def get_points_to_num_indices_map(self):
        return self.points_to_num_indices_map

    def get_selected_basis_indices(self):
        return self.selected_basis_indices

    def get_current_degree(self):
        return self.basis_degrees[-1]

    def get_basis_cardinality(self):
        return self.basis_cardinality

    def get_current_points(self):
        return self.permuted_pts[:, :self.lu_row]


def precondition_matrix(pts, precond_weights, matrix):
    if (precond_weights.shape[0] != matrix.shape[0]):
        raise Exception("This should not happen")
    matrix = np.dot(np.diag(np.sqrt(precond_weights)), matrix)
    return matrix


def precondition_function_values(vals, pts, precond_weights):
    # precondition but do not modify pts in place
    # only use precond weights associated with pts
    return precondition_matrix(pts, precond_weights[:vals.shape[0]], vals)


def transform_least_interpolant(pts, vals, pce, L_factor, U_factor,
                                H_factor_blocks, precond_weights):
    num_pts, num_qoi = vals.shape
    if (num_pts != L_factor.shape[0]):
        print((num_pts, L_factor.shape))
        msg = "transform_least_interpolant() "
        msg += "values supplied are inconsistent with the LU factorization"
        raise Exception(msg)

    if precond_weights is not None:
        precond_vals = precondition_function_values(vals, pts, precond_weights)
    else:
        precond_vals = vals

    # H = get_dense_block_diagonal_matrix(H_factor_blocks)

    LU_inv = np.linalg.inv(np.dot(L_factor, U_factor))

    V_inv = pre_multiply_block_diagonal_matrix(LU_inv, H_factor_blocks, True)

    coefficients = np.dot(V_inv, precond_vals)

    pce.set_coefficients(coefficients)

    return pce


def get_current_least_interpolant(pts, vals, pce, L_factor, U_factor, lu_row,
                                  H_factor_blocks, selected_basis_indices,
                                  precond_weights):
    # Chop off parts of unnecessarily allocated vector HFactorEntries_
    L_factor = L_factor[:lu_row, :lu_row]
    U_factor = U_factor[:lu_row, :lu_row]

    pce.set_indices(selected_basis_indices)

    pce = transform_least_interpolant(
        pts, vals, pce, L_factor, U_factor, H_factor_blocks, precond_weights)

    return pce
