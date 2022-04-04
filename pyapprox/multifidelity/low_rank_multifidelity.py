import numpy as np
import hashlib

from pyapprox.util.linalg import (
    cholesky_solve_linear_system, pivoted_cholesky_decomposition
)


def select_nodes(V, N, weights=None, order=None):
    r"""
    Algorithm 1 . Cholesky decomposition method for selection of
    interpolation nodes from a finite-cardinality candidate set.
    The output Cholesky factor L is not necessary for this selection,
    but it is useful later in the synthesis of Algorithm 2.
    -----------------------------------------------------------------

    compute rank num_hf_runs pivoted cholesky factorization of Grammian
    of lf solutions

    V (matrix): columns of V are snapshots of low-fidelity model
        V is not positive symmetric definite only V.T*V. This algorithm
        takes advantage of the fact that we want to compute cholesky of V.T*V
    N (int)   : the number of interpolation nodes/ high-fidelity runs
    M (int)   : the number of snapshots/ low-fidelity runs

    order = columns of V that must be added first. These columns correspond
    to previously used points when adding new points to a bifidelity
    approximation
    """
    # We will edit V so make a copy
    V = V.copy()

    M = V.shape[1]
    print("QOI: %s, snapshots: %s, interpolation nodes: %s" % (V.shape[0],
                                                               M, N))
    assert N <= M

    # Initialize the ensemble for each parameter z[m]
    # INNER PRODUCT addition
    if weights == None:
        w = np.array([np.dot(V[:, m], V[:, m]) for m in range(M)])
        # assert np.allclose(np.diag(dot(V.T, V)), w)
    else:
        w = np.array([dot(V[:, m], np.dot(weights, V[:, m]))
                         for m in range(M)])

    # Initialize the nodal selection vector and Cholesky factor
    # EDIT/ERROR causes repeat indices
    list_p = []
    P = np.arange(M, dtype=int)
    O = np.arange(M, dtype=int)
    # EDIT/ERROR ((M, N)) not ((N, M))
    L = np.zeros((M, N))
    r = np.empty((M))

    for n in range(N):
        # Find the next interpolation point (the next pivot)
        # EDIT/ERROR as above different p
        if order is not None and n < len(order):
            p = O[order[n]]
            # swap order[n] with column n of V
            # column n contains the original column P[n]
            O[P[n]] = p
            O[P[p]] = n
        else:
            p = np.argmax(w[n:M]) + n

        # Avoid ill-conditioning
        if w[p] < 2*np.finfo(float).eps:
            print('Grammian is numerically singular...',)
            print('The grammian has rank %s and size %s' % (n, M))
            n -= 1
            break

        # Update P and exchange column n and p in V
        # EDIT/ERROR as above different p
        P[[n, p]] = P[[p, n]]
        list_p.append(p)
        V[:, [n, p]] = V[:, [p, n]]

        # EDIT/ERROR these switches are omitted
        L[[n, p], :] = L[[p, n], :]
        w[[n, p]] = w[[p, n]]

        # Update L
        for t in range(n+1, M):
            # EDIT/ERROR range(n) not range(N)
            # INNER PRODUCT addition
            if weights == None:
                r[t] = np.dot(V[:, t], V[:, n]) - \
                    sum([L[t, j]*L[n, j] for j in range(n)])
            else:
                r[t] = np.dot(V[:, t], dot(weights, V[:, n])) - \
                    sum([L[t, j]*L[n, j] for j in range(n)])

        # TODO move L[n,n] out and combine loops?
        L[n, n] = np.sqrt(w[n])
        for t in range(n+1, M):
            L[t, n] = r[t]/L[n, n]
            w[t] = w[t] - L[t, n]**2

    # Truncate the Cholesky factor
    L = L[:n+1, :]
    # EDIT/ERROR as above different p
    P = P[:n+1]

    return P, L


def select_nodes_cholesky(V, npivots, weights=None, order=None):
    print(V.shape)
    L, P, error, chol_flag = pivoted_cholesky_decomposition(
        V.T.dot(V), npivots, init_pivots=order)
    # pivoted_cholesky_decomposition returns unpivoted L so change to pivoted
    # L
    L = L[P, :]
    return P, L


def synthesis_operator(lf_selected_values, hf_selected_values,
                       chol_factor, lf_test_values, weights=None):
    r"""
    Algorithm 2. Algorithmic evaluation of the synthesis operation
    -----------------------------------------------------------------

    G_inv is $G^{-1} = u^L(\gamma)^{-1} u^L(\gamma)^{-T}$

    lf_selected_values: matrix (num_selected__samples x num_qoi)
    hf_selected_values: matrix (num_selected_samples x num_qoi)
    lf_test_values:     matrix (num_test_samples x num_qoi)
    """
    assert lf_selected_values.shape == hf_selected_values.shape
    assert lf_selected_values.shape[1] == lf_test_values.shape[1]

    # INNER PRODUCT addition
    if weights == None:
        g = np.dot(lf_selected_values, lf_test_values.T)
    else:
        g = np.dot(lf_selected_values, dot(weights, lf_test_values.T))

    # TODO
    # should not compute G_l_inv but rather use back aand forward subsitution
    # to compute L^{-T}L^{-1}g (see akils first paper), L is cholesky factor
    G_l_inv = np.linalg.inv(np.dot(chol_factor, chol_factor.T))
    c = np.dot(G_l_inv, g)
    #c = cholesky_solve_linear_system( L, rhs[pivots] ).squeeze()
    mf_test_values = np.dot(hf_selected_values.T, c).squeeze()
    # dot product returns num-qoi x num-samples
    # but my models return num-samples x num-qoi
    return mf_test_values.T, c


class BiFidelityModel(object):
    def __init__(self, lf_model, hf_model):
        self.lf_model = lf_model
        self.hf_model = hf_model
        self.candidate_samples = None
        self.chol_factor = None
        self.id = None

    def build(self, num_hf_runs, generate_samples, num_lf_candidates=1e3):
        # 1. Evaluate the low-fidelity model u_L on a candidate set Gamma.
        # ----------------------------------------------------------------
        self.candidate_samples = generate_samples(num_lf_candidates)
        lf_candidate_values = self.lf_model(self.candidate_samples)
        self.build_from_samples(
            num_hf_runs, self.candidate_samples, lf_candidate_values)

    def build_from_samples(self, num_hf_runs, candidate_samples,
                           lf_candidate_values):
        # 2. Choose an ordered subset of N nodes gamma using Algorithm 1.
        # ----------------------------------------------------------------
        # select_nodes assumes num-qoi x num-samples
        # but my models return num-samples x num-qoi
        pivots, self.chol_factor = select_nodes(
            V=lf_candidate_values.T, N=num_hf_runs)

        self.lf_selected_samples = candidate_samples[:, pivots]
        self.lf_selected_values = lf_candidate_values[pivots, :]

        # 3. Evaluate the high-fidelity u_H model on gamma.
        # ----------------------------------------------------------------
        self.hf_selected_values = self.hf_model(self.lf_selected_samples)

    def evaluate_set(self, samples):
        # 4. Use u_H(gamma) to construct the interpolation operator
        #    I_L_H (gamma,.) and evaluate at any z using Algorithm 2 with
        #    input data v = u_L(z).
        # ----------------------------------------------------------------
        print('ID', self.id)
        lf_values = self.lf_model(samples)
        mf_values = synthesis_operator(
            self.lf_selected_values, self.hf_selected_values,
            self.chol_factor, lf_values)[0]
        return mf_values

    def get_condition_number_data(self):
        lf_condition_number = np.linalg.cond(
            np.dot(self.lf_selected_values, self.lf_selected_values.T))
        hf_condition_number = np.linalg.cond(
            np.dot(self.hf_selected_values, self.hf_selected_values.T))

        print('LF condition number', '%e' % lf_condition_number)
        print('HF condition number', '%e' % hf_condition_number)
        print("1/machine eps", 1./np.finfo(float).eps)

        return lf_condition_number, hf_condition_number

    def __call__(self, samples):
        return self.evaluate_set(samples)


def compute_mean_l2_error(solutions, true_solutions):
    error = np.linalg.norm(true_solutions-solutions, ord=2, axis=1)
    abs_error = np.mean(error)
    rel_error = np.mean(error/np.linalg.norm(true_solutions, axis=1))
    return abs_error, rel_error
