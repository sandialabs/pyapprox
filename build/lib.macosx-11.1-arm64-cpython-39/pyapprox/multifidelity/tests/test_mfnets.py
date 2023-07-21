import unittest
import copy

from pyapprox.util.utilities import check_gradients, approx_jacobian
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices

from pyapprox.multifidelity.mfnets import (
    MFNets, nx, np, partial, monomial_1d, least_squares_objective,
    multiplicative_additive_discrepancy_fun, monomial_nd,
    populate_functions_multiplicative_additive_graph,
    mfnets_node_objective
)


def setup_peer_mfnets_graph(nnodes):
    r"""Make a two layer graph with one leaf node and multiple root nodes

    Parameters
    ----------
    nnodes: integer
        The number of nodes in the graph

    Returns
    -------
    graph : nx.DiGraph
        Empty directed acyclic graph

    Notes
    -----
    graph.nodes returns nodes in order they are added here. This function
    adds nodes [1, 2, ..., nnodes] so that nnodes is the high-fidelity source
    if we instead used graph.add_node(nnodes) then
    [graph.add_node(n) for n in range(1, nnodes-1)] the high fidelity source
    would be in the first entry, then lowest and then increasing fidelity
    """
    # initialize empty graph
    graph = nx.DiGraph()

    # add nodes ane edges
    for ii in range(1, nnodes+1):
        graph.add_node(ii)
    for ii in range(1, nnodes):
        graph.add_edge(ii, nnodes)
    return graph


def setup_4_model_graph():
    r"""
    1 -> 2 -> \
               4
         3 -> /
    """
    nnodes = 4
    # initialize empty graph
    graph = nx.DiGraph()
    # add nodes ane edges
    for ii in range(1, nnodes+1):
        graph.add_node(ii)
    graph.add_edge(1, 2)
    graph.add_edge(2, 4)
    graph.add_edge(3, 4)
    return graph


class TestMFNets(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_multiplicative_additive_discrepancy_fun_two_model(self):
        ninputs, ninput_samples = 1, 2
        discrepancy_fun, scaling_fun = monomial_1d, monomial_1d
        nparents, ndiscrepancy_params, nscaling_params = 1, 3, 1

        input_samples = np.random.uniform(0, 1, (nparents+1, ninput_samples))

        fun_and_jac = partial(
            multiplicative_additive_discrepancy_fun, discrepancy_fun,
            [scaling_fun]*nparents, ndiscrepancy_params,
            [nscaling_params]*nparents, ninputs, input_samples=input_samples)

        # compute derivative with respect to parameters
        nparams = ndiscrepancy_params + nscaling_params*nparents
        params0 = np.random.uniform(-1, 1, (nparams, 1))
        jac_fd = approx_jacobian(
            partial(fun_and_jac, return_jac=False), params0)

        jac = fun_and_jac(params0, return_jac="params")[1]
        assert np.allclose(jac, jac_fd)

        def monomial_basis_1d(x, degree):
            return x.T**np.arange(degree+1)[None, :]

        discrepancy_basis = monomial_basis_1d(
            input_samples[:1], ndiscrepancy_params-1)
        scaling_bases = [
            monomial_basis_1d(input_samples[:1], nscaling_params-1)]*nparents
        scaling_jac = np.hstack(
            [np.diag(input_samples[1, :]).dot(b) for b in scaling_bases])
        jac_true = np.hstack((discrepancy_basis, scaling_jac))
        # print(jac)
        # print(jac_true)
        # print(jac_fd)
        assert np.allclose(jac, jac_true)

        # compute derivative with respect to parent inputs
        fun_and_jac = partial(
            multiplicative_additive_discrepancy_fun, discrepancy_fun,
            [scaling_fun]*nparents, ndiscrepancy_params,
            [nscaling_params]*nparents, ninputs, params0)

        def fun_and_jac_wrapper(x, return_jac):
            return fun_and_jac(np.vstack((input_samples[:1, :], x.T)),
                               return_jac)
        parent_inputs0 = np.random.uniform(0, 1, (nparents, ninput_samples))
        jac_fd = approx_jacobian(
            lambda x: fun_and_jac_wrapper(x, return_jac=False),
            parent_inputs0.T)
        jac = fun_and_jac_wrapper(parent_inputs0.T, return_jac="inputs")[1]
        # print(jac)
        # print(jac_fd)
        assert np.allclose(jac, jac_fd)

    def test_multiplicative_additive_discrepancy_fun_three_model(self):
        ninputs, ninput_samples = 1, 4
        discrepancy_fun, scaling_fun = monomial_1d, monomial_1d
        nparents, ndiscrepancy_params, nscaling_params = 2, 3, 1

        input_samples = np.random.uniform(0, 1, (nparents+1, ninput_samples))

        fun_and_jac = partial(
            multiplicative_additive_discrepancy_fun, discrepancy_fun,
            [scaling_fun]*nparents, ndiscrepancy_params,
            [nscaling_params]*nparents, ninputs, input_samples=input_samples)

        # compute derivative with respect to parameters
        nparams = ndiscrepancy_params + nscaling_params*nparents
        params0 = np.random.uniform(-1, 1, (nparams, 1))
        jac_fd = approx_jacobian(
            partial(fun_and_jac, return_jac=False), params0)

        jac = fun_and_jac(params0, return_jac="params")[1]
        assert np.allclose(jac, jac_fd)

        def monomial_basis_1d(x, degree):
            return x.T**np.arange(degree+1)[None, :]

        discrepancy_basis = monomial_basis_1d(
            input_samples[:1], ndiscrepancy_params-1)
        scaling_bases = [
            monomial_basis_1d(input_samples[:1], nscaling_params-1)]*nparents
        scaling_jac = np.hstack(
            [np.diag(input_samples[ii+1, :]).dot(scaling_bases[ii])
             for ii in range(len(scaling_bases))])
        jac_true = np.hstack((discrepancy_basis, scaling_jac))
        # print(jac)
        # print(jac_true)
        # print(jac_fd)
        assert np.allclose(jac, jac_true)

        # compute derivative with respect to parent inputs
        fun_and_jac = partial(
            multiplicative_additive_discrepancy_fun, discrepancy_fun,
            [scaling_fun]*nparents, ndiscrepancy_params,
            [nscaling_params]*nparents, ninputs, params0)

        def fun_and_jac_wrapper(x, return_jac):
            x = x.reshape(nparents, x.shape[0]//nparents)
            return fun_and_jac(np.vstack((input_samples[:1, :], x)),
                               return_jac)
        parent_inputs0 = np.random.uniform(0, 1, (nparents, ninput_samples))
        jac_fd = approx_jacobian(
            lambda x: fun_and_jac_wrapper(x, return_jac=False),
            parent_inputs0.flatten())
        jac = fun_and_jac_wrapper(parent_inputs0.flatten(),
                                  return_jac="inputs")[1]
        # print(jac)
        # print(jac_fd)
        assert np.allclose(np.hstack(jac), jac_fd)

    def test_evaluation_and_gradient_two_models(self):
        """
        y = f_2(x, f_1(x))
          = d_2(x) + 3f_1(x)
          = d_2(x) + 3(1+x+x^2)
          = 2+2x+2x^2 + 3(1+x+x^2)
        """
        ninputs = 1
        nnodes = 2
        ndiscrepancy_params, nscaling_params = 3, 1
        graph = setup_peer_mfnets_graph(nnodes)
        populate_functions_multiplicative_additive_graph(
            graph, monomial_1d, monomial_1d, ndiscrepancy_params,
            nscaling_params, ninputs)

        input_samples = np.linspace(0, 2, 11)[None, :]
        mfnets = MFNets(graph)
        params = np.hstack((
            np.ones(ndiscrepancy_params),
            2*np.ones(ndiscrepancy_params),  [3]))[:, None]
        mfnets.set_parameters(params)
        node_id = mfnets.get_nnodes()
        values = mfnets.forward_pass(input_samples, node_id)

        # print([(n, mfnets.graph.nodes[n]) for n in mfnets.graph.nodes])
        true_values = 5*(1+input_samples+input_samples**2).T
        assert np.allclose(values, true_values)

        def fun(samples):
            return 5*(1+samples+samples**2).T

        noise_std = 1.0
        obj_fun = least_squares_objective
        train_samples = np.linspace(0, 2, 5)[None, :]
        train_values = fun(train_samples)
        params0 = np.arange(params.shape[0])[:, None]
        mfnets.clear_data()
        mfnets.set_parameters(params0)
        vals = mfnets.forward_pass(train_samples, node_id)
        assert np.allclose(vals[:, 0], [3, 12.25, 30, 56.25, 91])

        def node_objective(pp, return_grad):
            mfnets.clear_data()
            return mfnets_node_objective(
                mfnets, node_id, train_samples, train_values, obj_fun,
                noise_std, pp, return_grad)

        node_obj_val, node_grad = node_objective(params0, True)

        node_grad_fd = approx_jacobian(
            partial(node_objective, return_grad=False), params0)
        true_node_grad = np.array(
             [[630, 1065, 1878, 105, 177.5, 313, 803.5]]).T
        # print(true_node_grad, node_grad_fd.T, node_grad)
        assert np.allclose(true_node_grad, node_grad_fd.T)
        assert np.allclose(node_grad, true_node_grad)

        def node_objective_wrapper(pp):
            result = node_objective(pp, return_grad=True)
            return result[0], result[1].T
        diffs = check_gradients(node_objective_wrapper, True, params0)
        assert diffs.min() < 2e-7 and diffs.max() > 1e-2

    def test_evaluation_and_gradient_three_models_peer(self):
        """
        ninputs = 1
        y = f_3(x, f_1(x), f_2(x))
          = d_3(x) + 3f_1(x)+4f_2(x)
          = d_3(x) + 3(1+x+x^2) + 4(1+x+x^2)
          = 2+2x+2x^2 + 3(1+x+x^2) + 4(1+x+x^2)

        ninputs = 2
        y = f_3(x, f_1(x), f_2(x))
          = d_3(x) + 3f_1(x)+4f_2(x)
          = d_3(x) + 3(1+x+y) + 4(1+x+y)
          = 2+2x+2y + 3(1+x+y) + 4(1+x+y)
        """
        ninputs = 2
        nnodes = 3
        discrep_order, scaling_order = 1, 0
        graph = setup_peer_mfnets_graph(nnodes)
        discrep_indices = compute_hyperbolic_indices(ninputs, discrep_order)
        scaling_indices = compute_hyperbolic_indices(ninputs, scaling_order)
        ndiscrepancy_params, nscaling_params = (
            discrep_indices.shape[1], scaling_indices.shape[1])
        discrep_fun = partial(monomial_nd, discrep_indices)
        scaling_fun = partial(monomial_nd, scaling_indices)
        populate_functions_multiplicative_additive_graph(
            graph, discrep_fun, scaling_fun, ndiscrepancy_params,
            nscaling_params, ninputs)

        # input_samples = np.linspace(0, 2, 11)[None, :]
        input_samples = np.random.uniform(0, 2, (ninputs, 11))
        mfnets = MFNets(graph)
        params = np.hstack((
            np.ones(2*ndiscrepancy_params),
            2*np.ones(ndiscrepancy_params),  [3, 4]))[:, None]
        mfnets.set_parameters(params)
        node_id = mfnets.get_nnodes()
        values = mfnets.forward_pass(input_samples, node_id)

        # ninputs = 1
        # true_values = 9*(1+input_samples+input_samples**2).T
        # ninputs = 2
        true_values = 9*(1+input_samples.sum(axis=0))[:, None]
        # print(values)
        # print(true_values)
        assert np.allclose(values, true_values)

        def fun(samples):
            # ninput = 1
            # return 9*(1+samples+samples**2).T
            # ninput = 2
            return 9*(1+samples.sum(axis=0))[:, None]

        noise_std = 1.0
        obj_fun = least_squares_objective
        # train_samples = np.linspace(0, 2, 5)[None, :]
        train_samples = np.random.uniform(0, 2, (ninputs, 5))
        train_values = fun(train_samples)
        params0 = np.arange(params.shape[0])[:, None]

        def node_objective(pp, return_grad):
            mfnets.clear_data()
            return mfnets_node_objective(
                mfnets, node_id, train_samples, train_values, obj_fun,
                noise_std, pp, return_grad)

        node_grad = node_objective(params0, True)[1]

        node_grad_fd = approx_jacobian(
            partial(node_objective, return_grad=False), params0)
        # print(node_grad_fd)
        # print(node_grad)
        assert np.allclose(node_grad, node_grad_fd.T)

        def node_objective_wrapper(pp):
            result = node_objective(pp, return_grad=True)
            return result[0], result[1].T
        diffs = check_gradients(node_objective_wrapper, True, params0)
        assert diffs.min() < 1e-7 and diffs.max() > 1e-2

    def test_evaluation_and_gradient_four_models(self):
        """
        y = f_4(x, f_2(x), f_3(x))
          = d_4(x) + 6f_2(x, f_1(x)) +  7f_3(x))
          = d_4(x) + 6(d_2(x)+5f_1(x)) +  7f_3(x))
          = d_4(x) + 6((2+2x+2x^2)+5(1+x+x^2)) + 7(3+3x+3x^2)
          = 4+4x+4x^2 + 42(1+x+x^2) + 21(1+x+x^2)
          = 67(1+x+x^2)
        """
        ninputs = 1
        # nnodes = 4
        ndiscrepancy_params, nscaling_params = 3, 1
        graph = setup_4_model_graph()
        populate_functions_multiplicative_additive_graph(
            graph, monomial_1d, monomial_1d, ndiscrepancy_params,
            nscaling_params, ninputs)

        input_samples = np.linspace(0, 2, 11)[None, :]
        mfnets = MFNets(graph)
        params = np.hstack((
            np.ones(ndiscrepancy_params), 2*np.ones(ndiscrepancy_params),
            [5], 3*np.ones(ndiscrepancy_params),
            4*np.ones(ndiscrepancy_params), [6, 7]))[:, None]
        mfnets.set_parameters(params)

        node_id = mfnets.get_nnodes()
        values = mfnets.forward_pass(input_samples, node_id)
        true_values = 67*(1+input_samples+input_samples**2).T
        assert np.allclose(values, true_values)

        node_id = 3
        values = mfnets.forward_pass(input_samples, node_id)
        true_values = 3*(1+input_samples+input_samples**2).T
        assert np.allclose(values, true_values)

        node_id = 2
        values = mfnets.forward_pass(input_samples, node_id)
        true_values = 7*(1+input_samples+input_samples**2).T
        assert np.allclose(values, true_values)

        node_id = 1
        values = mfnets.forward_pass(input_samples, node_id)
        true_values = (1+input_samples+input_samples**2).T
        assert np.allclose(values, true_values)

        def fun(samples):
            return 67*(1+samples+samples**2).T

        noise_std = 1.0
        obj_fun = least_squares_objective
        train_samples = np.linspace(0, 2, 5)[None, :]
        train_values = fun(train_samples)
        params0 = np.arange(params.shape[0])[:, None]

        def node_objective(pp, return_grad):
            mfnets.clear_data()
            return mfnets_node_objective(
                mfnets, node_id, train_samples, train_values, obj_fun,
                noise_std, pp, return_grad)

        node_obj_val, node_grad = node_objective(params0, True)

        node_grad_fd = approx_jacobian(
            partial(node_objective, return_grad=False), params0)
        # print(node_grad.T)
        # print(node_grad_fd, params0.shape)
        assert np.allclose(node_grad, node_grad_fd.T)

        def node_objective_wrapper(pp):
            result = node_objective(pp, return_grad=True)
            return result[0], result[1].T
        diffs = check_gradients(node_objective_wrapper, True, params0)
        assert diffs.min() < 1e-7 and diffs.max() > 1e-2

    def _check_least_squares_optimization(self, ninputs, nmodels,
                                          discrep_order, scaling_order):
        if nmodels == 4:
            true_graph = setup_4_model_graph()
        elif nmodels == 3:
            true_graph = setup_peer_mfnets_graph(nmodels)
        else:
            raise ValueError()
        discrep_indices = compute_hyperbolic_indices(ninputs, discrep_order)
        scaling_indices = compute_hyperbolic_indices(ninputs, scaling_order)
        ndiscrepancy_params, nscaling_params = (
            discrep_indices.shape[1], scaling_indices.shape[1])
        discrep_fun = partial(monomial_nd, discrep_indices)
        scaling_fun = partial(monomial_nd, scaling_indices)
        populate_functions_multiplicative_additive_graph(
            true_graph, discrep_fun, scaling_fun, ndiscrepancy_params,
            nscaling_params, ninputs)

        true_mfnets = MFNets(true_graph)
        true_params = np.random.normal(0, 1, (true_mfnets.get_nparams(), 1))
        true_mfnets.set_parameters(true_params)
        nnodes = true_mfnets.get_nnodes()
        # print(true_params, 'pp')

        ntrain_samples_list = [ndiscrepancy_params+nscaling_params+1]*nnodes
        train_samples_list = [
            np.random.uniform(0, 2, (ninputs, n)) for n in ntrain_samples_list]
        train_values_list = [
            true_mfnets(s, ii+1) for ii, s in enumerate(train_samples_list)]
        noise_std_list = [1]*nnodes
        init_params = np.random.normal(0, 1, true_params.shape)
        mfnets = MFNets(copy.deepcopy(true_graph))
        # print(train_samples_list, train_values_list)

        node_id_list = np.arange(1, mfnets.get_nnodes()+1)

        mfnets.node_id_list = node_id_list
        mfnets.train_samples_list = train_samples_list
        mfnets.train_values_list = train_values_list
        mfnets.noise_std_list = noise_std_list
        mfnets.obj_fun = least_squares_objective

        def objective_wrapper(pp):
            result = mfnets.fit_objective(pp)
            return result[0], result[1].T

        # print((np.zeros(true_mfnets.get_nparams()),
        #        objective_wrapper(true_params)))
        assert np.allclose(np.zeros(true_mfnets.get_nparams()),
                           objective_wrapper(true_params)[1])

        diffs = check_gradients(objective_wrapper, True, init_params)
        assert diffs.min()/diffs.max() < 3e-6

        tol = 1e-10
        # opts = {'disp': True, "iprint": 3, "gtol": tol, "ftol": tol,
        #         "maxiter": 1e3, "method": 'L-BFGS-B'}
        opts = {'disp': True, "iprint": 3, "gtol": tol, "ftol": tol,
                "maxiter": 1e3, "method": 'BFGS'}
        mfnets.fit(train_samples_list, train_values_list, noise_std_list,
                   node_id_list, init_params, opts=opts)

        # solution that interpolates data is not unique so just check
        # solution interpolates solution everywhere, i.e. at independent
        # validation data
        # params = get_graph_params(mfnets.graph)
        # print(params-true_params)
        # assert np.allclose(params, true_params)

        nvalidation_samples = 10
        validation_samples = np.random.uniform(
            0, 2, (ninputs, nvalidation_samples))
        assert np.allclose(
            true_mfnets(validation_samples), mfnets(validation_samples))

    def test_least_squares_optimization(self):
        test_cases = [[1, 4, 1, 0], [2, 3, 2, 0]]
        for test_case in test_cases:
            self._check_least_squares_optimization(*test_case)


if __name__ == "__main__":
    mfnets_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMFNets)
    unittest.TextTestRunner(verbosity=2).run(mfnets_test_suite)
