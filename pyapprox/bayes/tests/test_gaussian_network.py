import scipy
import unittest
import numpy as np
import networkx as nx

from pyapprox.bayes.laplace import (
    laplace_posterior_approximation_for_linear_models
)
from pyapprox.variables.gaussian import (
    joint_density_from_linear_conditional_relationship,
    convert_gaussian_to_canonical_form,
    convert_conditional_probability_density_to_canonical_form,
    multiply_gaussian_densities_in_compact_canonical_form,
    convert_gaussian_from_canonical_form
)
from pyapprox.bayes.gaussian_network import (
    get_var_ids_to_eliminate_from_node_query, get_var_ids_to_eliminate,
    GaussianNetwork, cond_prob_variable_elimination
)


class TestGaussianNetwork(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_get_var_ids_to_eliminate_from_node_query(self):
        nmodels, nvars_per_model = 4, 2
        network_labels = ['M%d' % ii for ii in range(nmodels)]
        # network_ids = np.arange(len(network_labels))
        network_node_var_ids = [
            list(range(nvars_per_model*ii, nvars_per_model*(ii+1)))
            for ii in range(nmodels)]
        query_labels_idx = [1]

        query_labels = [network_labels[idx] for idx in query_labels_idx]
        var_ids_to_eliminate = get_var_ids_to_eliminate_from_node_query(
            network_node_var_ids, network_labels, query_labels,
            evidence_node_ids=None)
        true_var_ids_to_eliminate = np.concatenate(
            [network_node_var_ids[ii] for ii in range(nmodels)
             if ii not in query_labels_idx])
        assert np.allclose(var_ids_to_eliminate, true_var_ids_to_eliminate)

    def test_get_var_ids_to_eliminate(self):
        nmodels = 4
        network_labels = ['M%d' % ii for ii in range(nmodels)]
        network_ids = np.arange(len(network_labels))
        query_labels_idx = [1]

        query_labels = [network_labels[idx] for idx in query_labels_idx]
        var_ids_to_eliminate = get_var_ids_to_eliminate(
            network_ids, network_labels, query_labels, evidence_ids=None)
        mask = np.zeros(len(network_labels), dtype=bool)
        mask[query_labels_idx] = True
        true_var_ids_to_eliminate = network_ids[~mask]
        assert np.allclose(var_ids_to_eliminate, true_var_ids_to_eliminate)

        nmodels = 4
        network_labels = ['M%d' % ii for ii in range(nmodels)]
        network_ids = np.arange(len(network_labels))
        query_labels_idx = [1, 3]

        query_labels = [network_labels[idx] for idx in query_labels_idx]
        var_ids_to_eliminate = get_var_ids_to_eliminate(
            network_ids, network_labels, query_labels, evidence_ids=None)
        mask = np.zeros(len(network_labels), dtype=bool)
        mask[query_labels_idx] = True
        true_var_ids_to_eliminate = network_ids[~mask]
        assert np.allclose(var_ids_to_eliminate, true_var_ids_to_eliminate)

        nmodels, ndata_per_model = 4, 2
        network_labels = ['M%d' % ii for ii in range(nmodels)]
        query_labels_idx = [1]
        evidence_labels = []
        for jj in range(ndata_per_model):
            evidence_labels += ['M%d_%d' % (ii, jj) for ii in range(nmodels)]
        network_labels += evidence_labels
        network_ids = np.arange(len(network_labels))
        evidence_ids = network_ids[nmodels:]

        query_labels = [network_labels[idx] for idx in query_labels_idx]
        var_ids_to_eliminate = get_var_ids_to_eliminate(
            network_ids, network_labels, query_labels,
            evidence_ids=evidence_ids)
        # variable labels must not contain any labels associated with evidence
        # thus the slicing in the following loop
        mask = np.zeros(len(network_labels), dtype=bool)
        mask[query_labels_idx] = True
        mask[nmodels:] = True
        true_var_ids_to_eliminate = network_ids[~mask]
        assert np.allclose(var_ids_to_eliminate, true_var_ids_to_eliminate)

    def test_conditional_probability_density_factorization(self):
        nvars1, nvars2 = [3, 2]
        Amat = np.random.normal(0., 1., (nvars2, nvars1))
        bvec = np.random.normal(0, 1, (nvars2))
        mean1 = np.random.normal(0, 1, nvars1)
        temp1 = np.random.normal(0., 1., (nvars1, nvars1))
        cov1 = temp1.T.dot(temp1)
        temp2g1 = np.random.normal(0., 1., (nvars2, nvars2))
        cov2g1 = temp2g1.T.dot(temp2g1)

        joint_mean, joint_covar = \
            joint_density_from_linear_conditional_relationship(
                mean1, cov1, cov2g1, Amat, bvec)

        precision_matrix1, shift1, normalization1 = \
            convert_gaussian_to_canonical_form(mean1, cov1)

        var1_ids, nvars_per_var1 = [0], [nvars1]
        var2_ids, nvars_per_var2 = [1], [nvars2]

        precision_matrix2g1, shift2g1, normalization2g1, var2g1_ids, \
            nvars_per_var2g1 = \
            convert_conditional_probability_density_to_canonical_form(
                Amat, bvec, cov2g1, var1_ids, nvars_per_var1,
                var2_ids, nvars_per_var2)

        (precision_matrix, shift, normalization, all_var_ids,
         nvars_per_all_vars) = \
             multiply_gaussian_densities_in_compact_canonical_form(
                 precision_matrix1, shift1, normalization1, var1_ids,
                 nvars_per_var1, precision_matrix2g1, shift2g1,
                 normalization2g1, var2g1_ids, nvars_per_var2g1)
        mean, covariance = convert_gaussian_from_canonical_form(
            precision_matrix, shift)

        assert np.allclose(joint_mean, mean)
        assert np.allclose(joint_covar, covariance)

    def test_hierarchical_graph_prior_same_nparams(self):
        nnodes = 3
        prior_covs = [1, 2, 3]
        prior_means = [-1, -2, -3]
        cpd_scales = [0.5, 0.4]
        node_labels = [f'Node_{ii}' for ii in range(nnodes)]
        nparams = np.array([2]*nnodes)
        cpd_mats = [None, cpd_scales[0]*np.eye(nparams[1], nparams[0]),
                    cpd_scales[1]*np.eye(nparams[2], nparams[1])]
        tmp21 = np.random.normal(0, 1, (nparams[1], nparams[0]))
        tmp21 /= tmp21.max()
        tmp32 = np.random.normal(0, 1, (nparams[2], nparams[1]))
        tmp32 /= tmp32.max()
        cpd_mats = [None, cpd_scales[0]*tmp21, cpd_scales[1]*tmp32]

        graph = nx.DiGraph()
        ii = 0
        graph.add_node(
            ii, label=node_labels[ii], cpd_cov=prior_covs[ii] *
            np.eye(nparams[ii]),
            nparams=nparams[ii], cpd_mat=cpd_mats[ii],
            cpd_mean=prior_means[ii]*np.ones((nparams[ii], 1)))
        for ii in range(1, nnodes):
            cpd_mean = np.ones(
                (nparams[ii], 1))*prior_means[ii]-cpd_mats[ii].dot(
                    np.ones((nparams[ii-1], 1))*prior_means[ii-1])
            cpd_cov = prior_covs[ii]*np.eye(nparams[ii])-cpd_mats[ii].dot(
                prior_covs[ii-1]*np.eye(nparams[ii-1])).dot(cpd_mats[ii].T)
            graph.add_node(ii, label=node_labels[ii], cpd_cov=cpd_cov,
                           nparams=nparams[ii], cpd_mat=cpd_mats[ii],
                           cpd_mean=cpd_mean)

        graph.add_edges_from([(ii, ii+1) for ii in range(nnodes-1)])

        network = GaussianNetwork(graph)
        network.convert_to_compact_factors()
        labels = [ll[1] for ll in network.graph.nodes.data('label')]
        factor_prior = cond_prob_variable_elimination(network, labels, None)
        prior_mean, prior_cov = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix, factor_prior.shift)
        print('Prior Covariance\n', prior_cov)
        print('Prior Mean\n', prior_mean)

        true_prior_mean = np.hstack(
            [[prior_means[ii]]*nparams[ii] for ii in range(nnodes)])
        print(true_prior_mean)
        assert np.allclose(true_prior_mean, prior_mean)
        true_prior_var = np.hstack(
            [[prior_covs[ii]]*nparams[ii] for ii in range(nnodes)])
        assert np.allclose(true_prior_var, np.diag(prior_cov))

        assert np.all(np.diff(nparams) == 0)
        v1, v2, v3 = prior_covs
        A21, A32 = cpd_mats[1:]
        I1, I2, I3 = [np.eye(nparams[0])]*nnodes
        S11, S22, S33 = v1*I1, v2*I2, v3*I3
        true_prior_cov = np.zeros((nparams.sum(), nparams.sum()))
        true_prior_cov[:nparams.sum(), :nparams.sum()] = np.vstack(
            [np.hstack([S11, S11.dot(A21.T), S11.dot(A21.T.dot(A32.T))]),
             np.hstack([A21.dot(S11), S22, S22.dot(A32.T)]),
             np.hstack([A32.dot(A21).dot(S11), A32.dot(S22),
                        S33])])
        assert np.allclose(true_prior_cov, prior_cov)

    def test_hierarchical_graph_prior_varying_nparams(self):
        nnodes = 3
        prior_covs = [1, 2, 3]
        prior_means = [-1, -2, -3]
        cpd_scales = [0.5, 0.4]
        node_labels = [f'Node_{ii}' for ii in range(nnodes)]
        nparams = np.array([2]*(nnodes-1)+[3])
        cpd_mats = [None, cpd_scales[0]*np.eye(nparams[1], nparams[0]),
                    cpd_scales[1]*np.eye(nparams[2], nparams[1])]
        tmp21 = np.random.normal(0, 1, (nparams[1], nparams[0]))
        tmp21 /= tmp21.max()
        tmp32 = np.random.normal(0, 1, (nparams[2], nparams[1]))
        tmp32 /= tmp32.max()
        cpd_mats = [None, cpd_scales[0]*tmp21, cpd_scales[1]*tmp32]

        graph = nx.DiGraph()
        ii = 0
        graph.add_node(
            ii, label=node_labels[ii], cpd_cov=prior_covs[ii] *
            np.eye(nparams[ii]),
            nparams=nparams[ii], cpd_mat=cpd_mats[ii],
            cpd_mean=prior_means[ii]*np.ones((nparams[ii], 1)))
        for ii in range(1, nnodes):
            cpd_mean = np.ones(
                (nparams[ii], 1))*prior_means[ii]-cpd_mats[ii].dot(
                np.ones((nparams[ii-1], 1))*prior_means[ii-1])
            cpd_cov = prior_covs[ii]*np.eye(nparams[ii])-cpd_mats[ii].dot(
                prior_covs[ii-1]*np.eye(nparams[ii-1])).dot(cpd_mats[ii].T)
            graph.add_node(ii, label=node_labels[ii], cpd_cov=cpd_cov,
                           nparams=nparams[ii], cpd_mat=cpd_mats[ii],
                           cpd_mean=cpd_mean)

        graph.add_edges_from([(ii, ii+1) for ii in range(nnodes-1)])

        network = GaussianNetwork(graph)
        network.convert_to_compact_factors()
        labels = [ll[1] for ll in network.graph.nodes.data('label')]
        factor_prior = cond_prob_variable_elimination(network, labels, None)
        prior_mean, prior_cov = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix, factor_prior.shift)

        true_prior_mean = np.hstack(
            [[prior_means[ii]]*nparams[ii] for ii in range(nnodes)])
        assert np.allclose(true_prior_mean, prior_mean)
        true_prior_var = np.hstack(
            [[prior_covs[ii]]*nparams[ii] for ii in range(nnodes)])
        assert np.allclose(true_prior_var, np.diag(prior_cov))

        assert np.all(np.diff(nparams[:nnodes-1]) == 0)
        v1, v2, v3 = prior_covs
        A21, A32 = cpd_mats[1:]
        I1, I2, I3 = [np.eye(nparams[ii]) for ii in range(nnodes)]
        S11, S22, S33 = v1*I1, v2*I2, v3*I3
        true_prior_cov = np.vstack(
            [np.hstack([S11, S11.dot(A21.T), S11.dot(A21.T.dot(A32.T))]),
             np.hstack([A21.dot(S11), S22, S22.dot(A32.T)]),
             np.hstack([A32.dot(A21).dot(S11), A32.dot(S22), S33])])
        assert np.allclose(true_prior_cov, prior_cov)

        print(network.node_labels)
        factor_prior = cond_prob_variable_elimination(network, ['Node_2'])
        prior = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix, factor_prior.shift)
        assert np.allclose(prior[0], true_prior_mean[nparams[:-1].sum():])
        assert np.allclose(
            prior[1], true_prior_cov[nparams[:-1].sum():, nparams[:-1].sum():])

    def test_peer_graph_prior(self):
        nnodes = 3
        graph = nx.DiGraph()
        prior_covs = [1, 2, 3]
        prior_means = [-1, -2, -3]
        cpd_scales = [0.5, 0.4]
        node_labels = [f'Node_{ii}' for ii in range(nnodes)]
        nparams = np.array([2]*nnodes)
        cpd_mats = [None, None, np.hstack(
            [cpd_scales[0]*np.eye(nparams[2], nparams[0]),
             cpd_scales[1]*np.eye(nparams[1], nparams[2])])]

        graph = nx.DiGraph()
        for ii in range(nnodes-1):
            graph.add_node(
                ii, label=node_labels[ii],
                cpd_cov=prior_covs[ii]*np.eye(nparams[ii]),
                nparams=nparams[ii], cpd_mat=cpd_mats[ii],
                cpd_mean=prior_means[ii]*np.ones((nparams[ii], 1)))

        ii = nnodes-1
        cov = np.eye(nparams[ii])*max(1e-8, prior_covs[ii]-np.dot(
            np.asarray(cpd_scales)**2, prior_covs[:ii]))
        graph.add_node(
            ii, label=node_labels[ii], cpd_cov=cov, nparams=nparams[ii],
            cpd_mat=cpd_mats[ii],
            cpd_mean=(prior_means[ii]-np.dot(cpd_scales[:ii],
                                             prior_means[:ii])) *
            np.ones((nparams[ii], 1)))

        graph.add_edges_from(
            [(ii, nnodes-1, {'cpd_cov': np.eye(nparams[ii])*cpd_scales[ii]})
             for ii in range(nnodes-1)])

        network = GaussianNetwork(graph)
        network.convert_to_compact_factors()
        labels = [ll[1] for ll in network.graph.nodes.data('label')]
        factor_prior = cond_prob_variable_elimination(network, labels, None)
        prior_mean, prior_cov = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix, factor_prior.shift)

        a31, a32 = cpd_scales
        v1, v2, v3 = prior_covs
        assert np.all(np.diff(nparams) == 0)
        I1, I2, I3 = [np.eye(nparams[0])]*nnodes
        rows = [np.hstack([v1*I1, 0*I1, a31*v1*I1]),
                np.hstack([0*I2, v2*I2, a32*v2*I2]),
                np.hstack([a31*v1*I3, a32*v2*I3, v3*I3])]
        true_prior_cov = np.vstack(rows)
        assert np.allclose(true_prior_cov, prior_cov)

        true_prior_mean = np.hstack(
            [[prior_means[ii]]*nparams[ii] for ii in range(nnodes)])
        # print(true_prior_mean)
        assert np.allclose(true_prior_mean, prior_mean)

    def test_one_node_inference(self):
        nnodes = 1
        prior_covs = [1]
        prior_means = [-1]
        node_labels = [f'Node_{ii}' for ii in range(nnodes)]
        nparams = np.array([2]*3)
        cpd_mats = [None]

        graph = nx.DiGraph()
        ii = 0
        graph.add_node(
            ii, label=node_labels[ii], cpd_cov=prior_covs[ii] *
            np.eye(nparams[ii]),
            nparams=nparams[ii], cpd_mat=cpd_mats[ii],
            cpd_mean=prior_means[ii]*np.ones((nparams[ii], 1)))

        nsamples = [3]
        noise_std = [0.01]*nnodes
        data_cpd_mats = [np.random.normal(0, 1, (nsamples[ii], nparams[ii]))
                         for ii in range(nnodes)]
        data_cpd_vecs = [np.ones((nsamples[ii], 1)) for ii in range(nnodes)]
        true_coefs = [
            np.random.normal(0, np.sqrt(prior_covs[ii]), (nparams[ii], 1))
            for ii in range(nnodes)]
        noise_covs = [np.eye(nsamples[ii])*noise_std[ii]**2
                      for ii in range(nnodes)]

        network = GaussianNetwork(graph)
        network.add_data_to_network(data_cpd_mats, data_cpd_vecs, noise_covs)
        network.convert_to_compact_factors()

        noise = [noise_std[ii]*np.random.normal(
            0, noise_std[ii], (nsamples[ii], 1))]
        values_train = [
            b.dot(c)+s+n for b, c, s, n in zip(
                data_cpd_mats, true_coefs, data_cpd_vecs, noise)]

        evidence, evidence_ids = network.assemble_evidence(values_train)
        factor_post = cond_prob_variable_elimination(
            network, node_labels, evidence_ids=evidence_ids, evidence=evidence)
        gauss_post = convert_gaussian_from_canonical_form(
            factor_post.precision_matrix, factor_post.shift)

        # solve using classical inversion formula
        true_post = laplace_posterior_approximation_for_linear_models(
            data_cpd_mats[0], prior_means[ii]*np.ones((nparams[0], 1)),
            np.linalg.inv(prior_covs[ii]*np.eye(nparams[0])),
            np.linalg.inv(noise_covs[0]), values_train[0], data_cpd_vecs[0])

        assert np.allclose(gauss_post[1], true_post[1])
        assert np.allclose(gauss_post[0], true_post[0].squeeze())

    def test_hierarchical_graph_inference(self):
        nnodes = 3
        graph = nx.DiGraph()
        prior_covs = [1, 2, 3]
        prior_means = [-1, -2, -3]
        cpd_scales = [0.5, 0.4]
        node_labels = [f'Node_{ii}' for ii in range(nnodes)]
        nparams = np.array([2]*3)
        cpd_mats = [None, cpd_scales[0]*np.eye(nparams[1], nparams[0]),
                    cpd_scales[1]*np.eye(nparams[2], nparams[1])]

        ii = 0
        graph.add_node(
            ii, label=node_labels[ii], cpd_cov=prior_covs[ii] *
            np.eye(nparams[ii]),
            nparams=nparams[ii], cpd_mat=cpd_mats[ii],
            cpd_mean=prior_means[ii]*np.ones((nparams[ii], 1)))
        for ii in range(1, nnodes):
            cpd_mean = np.ones((nparams[ii], 1))*(
                prior_means[ii]-cpd_scales[ii-1]*prior_means[ii-1])
            cpd_cov = np.eye(nparams[ii])*max(
                1e-8, prior_covs[ii]-cpd_scales[ii-1]**2*prior_covs[ii-1])
            graph.add_node(ii, label=node_labels[ii], cpd_cov=cpd_cov,
                           nparams=nparams[ii], cpd_mat=cpd_mats[ii],
                           cpd_mean=cpd_mean)

        graph.add_edges_from([(ii, ii+1) for ii in range(nnodes-1)])

        nsamples = [3]*nnodes
        noise_std = [0.01]*nnodes
        data_cpd_mats = [np.random.normal(0, 1, (nsamples[ii], nparams[ii]))
                         for ii in range(nnodes)]
        data_cpd_vecs = [np.ones((nsamples[ii], 1)) for ii in range(nnodes)]
        true_coefs = [
            np.random.normal(0, np.sqrt(prior_covs[ii]), (nparams[ii], 1))
            for ii in range(nnodes)]
        noise_covs = [np.eye(nsamples[ii])*noise_std[ii]**2
                      for ii in range(nnodes)]

        network = GaussianNetwork(graph)
        network.add_data_to_network(data_cpd_mats, data_cpd_vecs, noise_covs)

        noise = [noise_std[ii]*np.random.normal(
            0, noise_std[ii], (nsamples[ii], 1)) for ii in range(nnodes)]
        values_train = [
            b.dot(c)+s+n for b, c, s, n in zip(
                data_cpd_mats, true_coefs, data_cpd_vecs, noise)]

        evidence, evidence_ids = network.assemble_evidence(values_train)

        network.convert_to_compact_factors()

        query_labels = node_labels[:nnodes]
        factor_post = cond_prob_variable_elimination(
            network, query_labels, evidence_ids=evidence_ids,
            evidence=evidence)
        gauss_post = convert_gaussian_from_canonical_form(
            factor_post.precision_matrix, factor_post.shift)

        assert np.all(np.diff(nparams) == 0)
        prior_mean = np.hstack(
            [[prior_means[ii]]*nparams[ii] for ii in range(nnodes)])

        v1, v2, v3 = prior_covs
        A21, A32 = cpd_mats[1:]
        I1, I2, I3 = [np.eye(nparams[0])]*3
        S11, S22, S33 = v1*I1, v2*I2, v3*I3
        prior_cov = np.vstack(
            [np.hstack([S11, S11.dot(A21.T), S11.dot(A21.T.dot(A32.T))]),
             np.hstack([A21.dot(S11), S22, S22.dot(A32.T)]),
             np.hstack([A32.dot(A21).dot(S11), A32.dot(S22),
                        S33])])

        # dataless_network = GaussianNetwork(graph)
        # dataless_network.convert_to_compact_factors()
        # labels = [l[1] for l in dataless_network.graph.nodes.data('label')]
        # factor_prior = cond_prob_variable_elimination(
        #     dataless_network, labels, None)
        # prior_mean1,prior_cov1 = convert_gaussian_from_canonical_form(
        #     factor_prior.precision_matrix,factor_prior.shift)
        # print('Prior Covariance\n',prior_cov1)
        # print('Prior Mean\n',prior_mean1)

        basis_mat = scipy.linalg.block_diag(*data_cpd_mats)
        noise_cov_inv = np.linalg.inv(scipy.linalg.block_diag(*noise_covs))
        values = np.vstack(values_train)
        # print('values\n',values)
        bvec = np.vstack(data_cpd_vecs)
        prior_mean = prior_mean[:nparams[:nnodes].sum()]
        prior_cov = prior_cov[:nparams[:nnodes].sum(), :nparams[:nnodes].sum()]
        true_post = laplace_posterior_approximation_for_linear_models(
            basis_mat, prior_mean, np.linalg.inv(
                prior_cov), noise_cov_inv, values,
            bvec)

        # print(gauss_post[0],'\n',true_post[0].squeeze())
        # print(true_post[1])
        assert np.allclose(gauss_post[1], true_post[1])
        assert np.allclose(gauss_post[0], true_post[0].squeeze())

        # check ability to marginalize prior after data has
        # been added.
        factor_prior = cond_prob_variable_elimination(network, ['Node_2'])
        prior = convert_gaussian_from_canonical_form(
            factor_prior.precision_matrix, factor_prior.shift)
        assert np.allclose(prior[0], prior_mean[nparams[:-1].sum():])
        assert np.allclose(
            prior[1], prior_cov[nparams[:-1].sum():, nparams[:-1].sum():])


if __name__ == '__main__':
    gaussian_network_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGaussianNetwork)
    unittest.TextTestRunner(verbosity=2).run(gaussian_network_test_suite)
