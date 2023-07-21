import numpy as np
import unittest
from scipy import stats
from scipy import sparse as scipy_sparse

from pyapprox.surrogates.coupled_systems import (
    SystemNetwork, build_chain_graph, build_peer_graph, plot_adjacency_matrix,
    gauss_jacobi_fixed_point_iteration, GaussJacobiSystemNetwork,
    get_extraction_indices, get_extraction_matrices
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.sampling import \
    generate_independent_random_samples
from pyapprox.util.utilities import flatten_2D_list


def get_3_peer_polynomial_components():
    r"""
    Each component has only one output (qoi) used as a coupling variable for
    the downstream model. Each component has only one random variable. Third
    model has two coupling variables 1 scalar from each upstream component.

    f0(z0)             f1(z0, z1)
         \             /
          \           /
           f2(z2,f1,f0)

    f0(z0)       = a00*z0**2
    f1(z0,z1)    = a10*z0**2 + a11*z1**2
    f2(z2,f1,f0) = a20*z2**2 + a21*f1(z0,z1)**2 + a22*f0(z0)**2

    A component consists of both random variables and coupling variables.

    The argument position of the random variables entering f2 are
    local_random_var_indices[2] = [0]

    The argument position of the coupling variables entering f2 are
    local_coupling_var_indices[2] = [1, 2]

    The index into variables list of function representing the entire system
    in this case that function is g(z0,z1,z2)=f2(z2,f1(z0,z1),f0(z0))
    global_random_var_indices[2] = [2]

    The global component id of the coupling variables entering f2 and the qoi
    id of the upstream component
    global_coupling_component_indices[2] = [1, 0,     0, 0].
    This means that the first coupling variable of f2 is
    is qoi 0 of component 1 and the second is qoi 0 of component 0.
    """
    local_random_var_indices = [[0], [0, 1], [0]]
    local_coupling_var_indices = [[], [], [1, 2]]
    global_random_var_indices = [[0], [0, 1], [2]]
    global_coupling_component_indices = [[], [], [1, 0, 0, 0]]
    ncomponents = len(local_random_var_indices)

    nlocal_vars = [
        len(local_random_var_indices[ii])+len(local_coupling_var_indices[ii])
        for ii in range(ncomponents)]
    aa = [(ii+2)*np.arange(1, 1+nlocal_vars[ii])[np.newaxis, :]
          for ii in range(ncomponents)]

    def f1(x): return np.sum(aa[0].dot(x**2), axis=0)[:, np.newaxis]
    def f2(x): return np.sum(aa[1].dot(x**2), axis=0)[:, np.newaxis]
    def f3(x): return np.sum(aa[2].dot(x**2), axis=0)[:, np.newaxis]
    funs = [f1, f2, f3]

    labels = [r"$M_%d$" % ii for ii in range(ncomponents)]
    graph_data = {
        'label': labels, 'functions': funs,
        'global_random_var_indices': global_random_var_indices,
        'local_random_var_indices': local_random_var_indices,
        'local_coupling_var_indices_in': local_coupling_var_indices,
        'global_coupling_component_indices': global_coupling_component_indices}
    graph = build_peer_graph(ncomponents, graph_data)

    nvars = np.unique(np.concatenate(global_random_var_indices)).sum()
    univariate_variables = [stats.uniform(0, 1)]*nvars
    variables = IndependentMarginalsVariable(univariate_variables)

    def system_fun(samples):
        funs = graph_data['functions']
        global_random_var_indices = graph_data['global_random_var_indices']
        values0 = funs[0](samples[global_random_var_indices[0], :])
        values1 = funs[1](samples[global_random_var_indices[1], :])
        values2 = funs[2](
            np.vstack([samples[global_random_var_indices[2], :],
                       values1.T, values0.T]))
        return [values0, values1, values2]

    return graph, variables, graph_data, system_fun


def get_3_peer_polynomial_components_multiple_qoi():
    r"""
    f0 and f1 have two outputs. One of output from f0 and both outputs of f1
    are used as input to f2

    f0(z0)                   f1(z0, z1)
         \                   /
          \                 /
           f2(z2,f10,f11,f01)

    f0(z0)       = [a00*z0**2, a00*z0]
    f1(z0,z1)    = [a10*z0**2 + a11*z1**2, a10*z0+a11*z1]
    f2(z2,f10,f01) = a20*z2**2+a21*f10(z0,z1)**2+a22*f11(z0)**2+a23*f01(z0)**2

    A component consists of both random variables and coupling variables.

    The argument position of the random variables entering f2 are
    local_random_var_indices[2] = [0]

    The argument position of the coupling variables entering f2 are
    local_coupling_var_indices[2] = [1, 2, 3]

    The index into variables list of function representing the entire system
    in this case that function is
    g(z0,z1,z2)=f2(z2,f10(z0,z1),f11(z0,z1),f01(z0))
    global_random_var_indices[2] = [2]

    The global component id of the coupling variables entering f2 and the qoi
    id of the upstream component
    global_coupling_component_indices[2] = [1, 0,  1, 1   0, 1].
    This means that the first coupling variable of f2 is
    is qoi 0 of component 1 and the second is qoi 1 of component 1
    the third is qoi 1 of component 0
    """
    local_random_var_indices = [[0], [0, 1], [0]]
    local_coupling_var_indices_in = [[], [], [1, 2, 3]]
    global_random_var_indices = [[0], [0, 1], [2]]
    global_coupling_component_indices = [[], [], [1, 0, 1, 1, 0, 1]]
    ncomponents = len(local_random_var_indices)

    nlocal_vars = [
        len(local_random_var_indices[ii]) +
        len(local_coupling_var_indices_in[ii])
        for ii in range(ncomponents)]
    aa = [(ii+2)*np.arange(1, 1+nlocal_vars[ii])[np.newaxis, :]
          for ii in range(ncomponents)]

    def f1(x): return np.hstack(
        [np.sum(aa[0].dot(x**2), axis=0)[:, np.newaxis],
         np.sum(aa[0].dot(x), axis=0)[:, np.newaxis]])

    def f2(x): return np.hstack(
        [np.sum(aa[1].dot(x**2), axis=0)[:, np.newaxis],
         np.sum(aa[1].dot(x), axis=0)[:, np.newaxis]])

    def f3(x): return np.sum(aa[2].dot(x**2), axis=0)[:, np.newaxis]
    funs = [f1, f2, f3]

    labels = [r"$M_%d$" % ii for ii in range(ncomponents)]
    graph_data = {
        'label': labels, 'functions': funs,
        'global_random_var_indices': global_random_var_indices,
        'local_random_var_indices': local_random_var_indices,
        'local_coupling_var_indices_in': local_coupling_var_indices_in,
        'global_coupling_component_indices': global_coupling_component_indices}
    graph = build_peer_graph(ncomponents, graph_data)

    nvars = np.unique(np.concatenate(global_random_var_indices)).sum()
    univariate_variables = [stats.uniform(0, 1)]*nvars
    variables = IndependentMarginalsVariable(univariate_variables)
    return graph, variables, graph_data


def get_3_recursive_polynomial_components():
    """
    Each component has only one output (qoi) used as a coupling variable for
    the downstream component. Each component has only one random variable

    f0(z0) -- f1(f0, z2) --  f2(z1,f1)

    f0(z0)    = a00*np.sqrt(z0)
    f1(f0,z1) = a10*f0(z0) + a11*z2**3
    f2(z2,f1) = a20*z1**2 + a21*f1(f0,z2)**2

    A component consists of both random variables and coupling variables.

    The argument position of the random variables entering f1
    local_random_var_indices[1] = [1]
    This means that the random variable is not the first argument to f1 but
    actually the second

    The argument position of the coupling variables entering f1 are
    local_coupling_var_indices[2] = [0]

    The index into variables list of function repsresenting the entire system
    in this case that function is g(z0,z1,z2)=f2(z2,f1(f0(z0),z1)).
    The random variable of f1 is z2, thus
    global_random_var_indices[1] = [2]

    The global component id of the coupling variables entering f1 and
    the qoi id of the upstream component
    global_coupling_component_indices[1] = [0, 0].
    This means that the first coupling variable of f1 is
    is qoi 0 of component 0.
    """
    local_random_var_indices = [[0], [1], [0]]
    local_coupling_var_indices_in = [[], [0], [1]]
    global_random_var_indices = [[0], [2], [1]]
    global_coupling_component_indices = [[], [0, 0], [1, 0]]
    ncomponents = len(local_random_var_indices)

    nlocal_vars = [
        len(local_random_var_indices[ii])+len(
            local_coupling_var_indices_in[ii])
        for ii in range(ncomponents)]
    aa = [(ii+2)*np.arange(1, 1+nlocal_vars[ii])[np.newaxis, :]
          for ii in range(ncomponents)]

    def f1(x): return np.sum(aa[0].dot(x**0.5), axis=0)[:, np.newaxis]
    def f2(x): return np.sum(
        aa[1].dot(np.vstack([x[0:1, :], x[1:2, :]**3])), axis=0)[:, np.newaxis]

    def f3(x): return np.sum(aa[2].dot(x**2), axis=0)[:, np.newaxis]
    funs = [f1, f2, f3]

    labels = [r"M_%d" % ii for ii in range(ncomponents)]
    graph_data = {
        'label': labels, 'functions': funs,
        'global_random_var_indices': global_random_var_indices,
        'local_random_var_indices': local_random_var_indices,
        'local_coupling_var_indices_in': local_coupling_var_indices_in,
        'global_coupling_component_indices': global_coupling_component_indices}
    graph = build_chain_graph(ncomponents, graph_data)

    nvars = np.unique(np.concatenate(global_random_var_indices)).sum()
    univariate_variables = [stats.uniform(0, 1)]*nvars
    variables = IndependentMarginalsVariable(univariate_variables)
    return graph, variables, graph_data


def get_3_recursive_polynomial_components_multiple_qoi():
    """
    First model has multiple qoi which are used as coupling variables for
    the second model. Third model has multiple random variables

    f0(z0) -- f1(f00, f01, z3) --  f2(z1, z2, f1)

    f0(z0)    = [a00*z0**2, a00*z0]
    f1(f0,z1) = a10*f00(z0)**2+a11*f01(z0)**2) + a12*z3**2
    f2(z2,f1) = a20*(z1**2+z2) + a21*f1(f0,z2)**2


    The global component id of the coupling variables entering f1
    and the qoi id of the upstream component
    global_coupling_component_indices[1] = [0, 0, 0, 1].
    This means that the first coupling variable of f1 is
    is qoi 0 of component 0.
    """
    local_random_var_indices = [[0], [2], [0, 1]]
    local_coupling_var_indices_in = [[], [0, 1], [2]]
    global_random_var_indices = [[0], [3], [1, 2]]
    global_coupling_component_indices = [[], [0, 0, 0, 1], [1, 0]]
    ncomponents = len(local_random_var_indices)

    nlocal_vars = [
        len(local_random_var_indices[ii]) +
        len(local_coupling_var_indices_in[ii])
        for ii in range(ncomponents)]
    aa = [(ii+2)*np.arange(1, 1+nlocal_vars[ii])[np.newaxis, :]
          for ii in range(ncomponents)]

    def f1(x): return np.hstack(
        [np.sum(aa[0].dot(x**2), axis=0)[:, np.newaxis],
         np.sum(aa[0].dot(x), axis=0)[:, np.newaxis]])

    def f2(x): return np.sum(aa[1].dot(x**2), axis=0)[:, np.newaxis]
    def f3(x): return np.sum(aa[2].dot(x**2), axis=0)[:, np.newaxis]
    funs = [f1, f2, f3]

    labels = [r"$M_%d$" % ii for ii in range(ncomponents)]
    graph_data = {
        'label': labels, 'functions': funs,
        'global_random_var_indices': global_random_var_indices,
        'local_random_var_indices': local_random_var_indices,
        'local_coupling_var_indices_in': local_coupling_var_indices_in,
        'global_coupling_component_indices': global_coupling_component_indices}
    graph = build_chain_graph(ncomponents, graph_data)

    nvars = np.unique(np.concatenate(global_random_var_indices)).sum()
    univariate_variables = [stats.uniform(0, 1)]*nvars
    variables = IndependentMarginalsVariable(univariate_variables)
    return graph, variables, graph_data


def get_chaudhuri_3_component_system():
    def fn1(samples):
        x = samples[:3]
        C = samples[3:]
        A1 = np.array([[9.7236, 0.2486]])
        C1 = 0.01*(x[0]**2+2*x[1]-x[2]) + A1.dot(C)[0, :]/np.linalg.norm(
            C, axis=0)
        y1 = 0.1*x[0] + x[1] - 0.5*x[2] + 10*C1
        vals = np.vstack([C1, y1]).T
        return vals

    def fn2(samples):
        x = samples[:3]
        C = samples[3:]
        A2 = np.array([[0.2486, 9.7764]])
        C2 = 0.01*(x[0]*x[1]+x[1]**2+x[2]) + A2.dot(C)[0, :]/np.linalg.norm(
            C, axis=0)
        y2 = 5*x[1] - x[2] - 5*C2
        return np.vstack([C2, y2]).T

    def fn3(samples):
        # samples y1, y2
        return (samples[0:1, :] + samples[1:2, :]).T

    nexog_vars = 5
    variable = IndependentMarginalsVariable(
        [stats.norm(1, 0.1)]*nexog_vars)

    # 0.02+(9.7236*x+0.2486*y)/sqrt[x^2+y^2]-x=0
    # 0.03+(9.7764*y+0.2486*x)/sqrt[x^2+y^2]-y=0

    return [fn1, fn2, fn3], variable


class TestCoupledSystem(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_get_extraction_matrices(self):
        component_output_labels = [
            ['v', 'dt_{orbit}', 'dt_{eclipse}', 'theta_{slew}'],
            ['tau_{tot}', 'PACS'],
            ['P_{tot}', 'A_{sa}', 'I_{max}', 'I_{min}']
        ]
        component_coupling_labels = [
            [],
            ['v', 'dt_{orbit}', 'theta_{slew}', 'I_{max}', 'I_{min}'],
            ['PACS', 'dt_{orbit}', 'dt_{eclipse}']]
        coup_ext_matrices = get_extraction_matrices(
            flatten_2D_list(component_output_labels),
            component_coupling_labels)
        coup_ext_indices = get_extraction_indices(
            flatten_2D_list(component_output_labels),
            component_coupling_labels)
        true_inds = [[], [0, 1, 3, 8, 9], [5, 1, 2]]
        for inds, mat, tinds in zip(
                coup_ext_indices, coup_ext_matrices, true_inds):
            assert np.allclose(inds, tinds)
            assert np.allclose(inds, np.where(mat == 1)[1])

    def test_gauss_jacobi_fpi_peer(self):
        ncomponent_outputs = [1, 1, 1]
        ncomponent_coupling_vars = [0, 0, 2]
        noutputs = np.sum(ncomponent_outputs)
        ncoupling_vars = np.sum(ncomponent_coupling_vars)
        adjacency_matrix = np.zeros((ncoupling_vars, noutputs))

        adjacency_matrix[0, 0] = 1
        adjacency_matrix[1, 1] = 1
        adjacency_matrix = scipy_sparse.csr_matrix(adjacency_matrix)
        # plot_adjacency_matrix(adjacency_matrix, component_shapes)
        # from matplotlib import pyplot as plt
        # plt.show()

        # output_extraction_indices = [[0], [1], [2]]
        exog_extraction_indices = [[0], [0, 1], [2]]
        coup_extraction_indices = [[], [], [1, 0]]
        qoi_ext_indices = [2]

        graph, variables, graph_data, system_fun = \
            get_3_peer_polynomial_components()
        component_funs = graph_data["functions"]

        nsamples = 10
        exog_samples = generate_independent_random_samples(
            variables, nsamples)
        init_coup_samples = np.ones((adjacency_matrix.shape[0], nsamples))

        outputs = gauss_jacobi_fixed_point_iteration(
            adjacency_matrix, exog_extraction_indices,
            coup_extraction_indices, component_funs,
            init_coup_samples, exog_samples,
            tol=1e-15, max_iters=100, verbose=0)[0]

        true_outputs = system_fun(exog_samples)
        assert np.allclose(outputs, np.hstack(true_outputs))

        # test when component_ids are specified
        network = GaussJacobiSystemNetwork(graph)
        network.set_adjacency_matrix(adjacency_matrix)
        network.set_extraction_indices(
            exog_extraction_indices, coup_extraction_indices, qoi_ext_indices,
            ncomponent_outputs)
        component_ids = [0, 1, 2]
        outputs = network(exog_samples, component_ids,
                          init_coup_samples=init_coup_samples)
        assert np.allclose(np.hstack(outputs), np.hstack(true_outputs))

        # test when component_ids are not specified and so qoi indices are
        # needed
        qoi_ext_indices = [2]  # return qoi of last model
        network = GaussJacobiSystemNetwork(graph)
        network.set_adjacency_matrix(adjacency_matrix)
        network.set_extraction_indices(
            exog_extraction_indices, coup_extraction_indices,
            qoi_ext_indices,
            ncomponent_outputs)
        component_ids = None
        outputs = network(exog_samples, component_ids,
                          init_coup_samples=init_coup_samples)
        # print(outputs, true_outputs[-1])
        assert np.allclose(outputs, true_outputs[-1])

    def test_gauss_jacobi_fpi_feedback(self):
        ncomponent_outputs = [2, 2, 1]
        ncomponent_coupling_vars = [2, 2, 2]
        noutputs = np.sum(ncomponent_outputs)
        ncoupling_vars = np.sum(ncomponent_coupling_vars)
        adjacency_matrix = np.zeros((ncoupling_vars, noutputs))

        adjacency_matrix[0, 0] = 1  # xi_00 = C1
        adjacency_matrix[1, 2] = 1  # xi_01 = C2
        adjacency_matrix[2, 0] = 1  # xi_10 = C1
        adjacency_matrix[3, 2] = 1  # xi_11 = C2
        adjacency_matrix[4, 1] = 1  # xi_20 = y1
        adjacency_matrix[5, 3] = 1  # xi_21 = y2
        adjacency_matrix = scipy_sparse.csr_matrix(adjacency_matrix)
        # plot_adjacency_matrix(
        #     adjacency_matrix, (ncomponent_coupling_vars, ncomponent_outputs))
        # from matplotlib import pyplot as plt
        # plt.show()

        component_funs, variables = get_chaudhuri_3_component_system()

        # output_extraction_indices = [[0, 1], [1, 2], [3]]
        exog_extraction_indices = [[0, 1, 2], [0, 3, 4], []]
        coup_extraction_indices = [[0, 1], [2, 3], [4, 5]]

        nsamples = 10
        exog_samples = generate_independent_random_samples(
            variables, nsamples-1)
        exog_samples = np.hstack(
            (exog_samples, variables.get_statistics("mean")))
        init_coup_samples = 5*np.ones((adjacency_matrix.shape[0], nsamples))

        outputs = gauss_jacobi_fixed_point_iteration(
            adjacency_matrix, exog_extraction_indices,
            coup_extraction_indices, component_funs,
            init_coup_samples, exog_samples,
            tol=1e-12, max_iters=20, verbose=0,
            anderson_memory=1)[0]

        # Mathematica Solution
        # Solve[(0.02 + (9.7236*x + 0.2486*y)/Sqrt[x^2 + y^2] - x == 0) &&
        # 0.03 + (9.7764*y + 0.2486*x)/Sqrt[x^2 + y^2] - y == 0, {x, y}]
        # print(outputs[-1, [0, 2]], [6.63852, 7.52628])
        assert np.allclose(outputs[-1, [0, 2]], [6.63852, 7.52628])

    def test_peer_feed_forward_system_of_polynomials(self):
        graph, variables, graph_data, system_fun = \
            get_3_peer_polynomial_components()
        network = SystemNetwork(graph)

        nsamples = 10
        samples = generate_independent_random_samples(variables, nsamples)
        values = network(samples, [0, 1, 2])

        component_nvars = network.component_nvars()
        assert component_nvars == [1, 2, 3]

        true_values = system_fun(samples)
        assert np.allclose(values, true_values)

    def test_peer_feed_forward_system_of_polynomials_multiple_qoi(self):
        graph, variables, graph_data = \
            get_3_peer_polynomial_components_multiple_qoi()
        network = SystemNetwork(graph)

        nsamples = 10
        samples = generate_independent_random_samples(variables, nsamples)
        values = network(samples)

        component_nvars = network.component_nvars()
        assert component_nvars == [1, 2, 4]

        funs = graph_data['functions']
        global_random_var_indices = graph_data['global_random_var_indices']
        values0 = funs[0](samples[global_random_var_indices[0], :])
        values1 = funs[1](samples[global_random_var_indices[1], :])
        values2 = funs[2](
            np.vstack([samples[global_random_var_indices[2], :],
                       values1.T, values0[:, 1:2].T]))

        assert np.allclose(values, values2)

        network_values = network(samples, [0, 1, 2])
        assert np.allclose(network_values[0], values0)
        assert np.allclose(network_values[1], values1)
        assert np.allclose(network_values[2], values2)

        # test when component_ids are specified
        ncomponent_outputs = [2, 2, 1]
        ncomponent_coupling_vars = [0, 0, 3]
        noutputs = np.sum(ncomponent_outputs)
        ncoupling_vars = np.sum(ncomponent_coupling_vars)
        adjacency_matrix = np.zeros((ncoupling_vars, noutputs))
        adjacency_matrix[0, 2] = 1
        adjacency_matrix[1, 3] = 1
        adjacency_matrix[2, 1] = 1
        adjacency_matrix = scipy_sparse.csr_matrix(adjacency_matrix)

        exog_extraction_indices = [[0], [0, 1], [2]]
        coup_extraction_indices = [[], [], [0, 1, 2]]
        qoi_extraction_indices = [0, 2, 3, 4]

        network = GaussJacobiSystemNetwork(graph)
        network.set_adjacency_matrix(adjacency_matrix)
        network.set_extraction_indices(
            exog_extraction_indices, coup_extraction_indices,
            qoi_extraction_indices, ncomponent_outputs)
        network.set_initial_coupling_sample(np.ones((ncoupling_vars, 1)))
        component_ids = [0, 1, 2]
        outputs = network(samples, component_ids, init_coup_samples=None)
        true_outputs = [values0, values1, values2]
        # print(outputs[0], true_outputs[0])
        # print(outputs[1], true_outputs[1])
        # print(outputs[2], true_outputs[2])
        assert np.allclose(np.hstack(outputs), np.hstack(true_outputs))

        outputs = network(samples, component_ids=None, init_coup_samples=None)
        assert np.allclose(outputs, np.hstack(true_outputs)[:, [0, 2, 3, 4]])

    def test_recursive_feed_forward_system_of_polynomials(self):
        graph, variables, graph_data = get_3_recursive_polynomial_components()
        network = SystemNetwork(graph)

        nsamples = 10
        samples = generate_independent_random_samples(variables, nsamples)
        values = network(samples)

        component_nvars = network.component_nvars()
        assert component_nvars == [1, 2, 2]

        funs = graph_data['functions']
        global_random_var_indices = graph_data['global_random_var_indices']
        values0 = funs[0](samples[global_random_var_indices[0], :])
        values1 = funs[1](
            np.vstack([values0.T, samples[global_random_var_indices[1], :]]))
        true_values = funs[2](
            np.vstack([samples[global_random_var_indices[2], :], values1.T]))

        assert np.allclose(values, true_values)

    def test_recursive_feed_forward_system_of_polynomials_multiple_qoi(self):
        graph, variables, graph_data = \
            get_3_recursive_polynomial_components_multiple_qoi()
        network = SystemNetwork(graph)

        nsamples = 10
        samples = generate_independent_random_samples(variables, nsamples)
        values = network(samples)

        component_nvars = network.component_nvars()
        assert component_nvars == [1, 3, 3]

        funs = graph_data['functions']
        global_random_var_indices = graph_data['global_random_var_indices']
        values0 = funs[0](samples[global_random_var_indices[0], :])
        values1 = funs[1](
            np.vstack([values0.T, samples[global_random_var_indices[1], :]]))
        true_values = funs[2](
            np.vstack([samples[global_random_var_indices[2], :], values1.T]))

        assert np.allclose(values, true_values)


if __name__ == "__main__":
    coupled_system_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestCoupledSystem)
    unittest.TextTestRunner(verbosity=2).run(coupled_system_test_suite)
    
