import numpy as np
import unittest
from scipy import stats

from pyapprox.coupled_systems import (
    SystemNetwork, build_chain_graph, build_peer_graph
)
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.probability_measure_sampling import \
    generate_independent_random_samples


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
    variables = IndependentMultivariateRandomVariable(univariate_variables)
    return graph, variables, graph_data


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
    is qoi 0 of component 1 and the second is qoi 1 of component 0.
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
    variables = IndependentMultivariateRandomVariable(univariate_variables)
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
    variables = IndependentMultivariateRandomVariable(univariate_variables)
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
    variables = IndependentMultivariateRandomVariable(univariate_variables)
    return graph, variables, graph_data


class TestCoupledSystem(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_peer_feed_forward_system_of_polynomials(self):
        graph, variables, graph_data = get_3_peer_polynomial_components()
        network = SystemNetwork(graph)

        nsamples = 10
        samples = generate_independent_random_samples(variables, nsamples)
        values = network(samples)

        component_nvars = network.component_nvars()
        assert component_nvars == [1, 2, 3]

        funs = graph_data['functions']
        global_random_var_indices = graph_data['global_random_var_indices']
        values0 = funs[0](samples[global_random_var_indices[0], :])
        values1 = funs[1](samples[global_random_var_indices[1], :])
        true_values = funs[2](
            np.vstack([samples[global_random_var_indices[2], :],
                       values1.T, values0.T]))

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
