import unittest

from scipy import stats
import numpy as np

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.univariate.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.nonlinear.classifiers import LogisticClassifier
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.optimization.minimize import (
    MultiStartOptimizer,
    RandomUniformOptimzerIterateGenerator,
)


class TestClassifiers(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return TorchMixin

    def _setup_tensor_product_legendre_pce(self, nvars: int, nterms_1d: int):
        bkd = self.get_backend()
        marginals = [stats.uniform(0, 1)] * nvars
        variable = IndependentMarginalsVariable(marginals, backend=bkd)
        bases_1d = [
            setup_univariate_orthogonal_polynomial_from_marginal(
                marginal, backend=bkd
            )
            for marginal in marginals
        ]
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_tensor_product_indices([nterms_1d] * variable.nvars())
        nqoi = 1
        bexp = PolynomialChaosExpansion(basis, None, nqoi=nqoi)
        return bexp

    def _setup_optimizer(self, clf):
        bkd = clf._bkd
        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(
            gtol=1e-8,
            maxiter=100,
            method="trust-constr",
        )
        optimizer.set_verbosity(0)
        ms_optimizer = MultiStartOptimizer(optimizer, ncandidates=2)
        iterate_gen = RandomUniformOptimzerIterateGenerator(
            clf.hyp_list().nactive_vars(), backend=bkd
        )
        # need to set bounds to be small because initial guess effects
        # optimization
        iterate_gen.set_bounds([-2, 2])
        ms_optimizer.set_initial_iterate_generator(iterate_gen)
        ms_optimizer.set_verbosity(0)
        return ms_optimizer

    def test_logistic_classifier(self):
        bkd = self.get_backend()
        bexp = self._setup_tensor_product_legendre_pce(1, 10)
        clf = LogisticClassifier(bexp)
        # train_samples = bkd.linspace(0.0, 1.0, 101)[None, :]
        train_samples = bkd.array(np.random.uniform(0.0, 1.0, 101))[None, :]

        def label_fun(xx):
            # return bkd.array(xx > 0.5, dtype=bkd.double_type()).T
            return bkd.array(
                bkd.abs(xx - 0.5) > 0.25, dtype=bkd.double_type()
            ).T

        train_values = label_fun(train_samples)

        clf._set_training_data(train_samples, train_values)
        penalty_weight = 1e-2
        clf.set_optimizer(self._setup_optimizer(clf), penalty_weight)
        iterate = bkd.asarray(
            np.random.normal(0, 1, (clf.hyp_list().nactive_vars(), 1))
        )
        errors = clf._optimizer._objective.check_apply_jacobian(
            iterate, disp=False
        )
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.1
        errors = clf._optimizer._objective.check_apply_hessian(
            iterate, disp=False
        )
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.1
        clf.fit(train_samples, train_values)

        test_samples = bkd.array(np.random.uniform(0.0, 1.0, 101))[None, :]
        test_labels = label_fun(test_samples)
        clf_labels = clf.labels(test_samples)
        nwrong = np.where(bkd.abs(clf_labels - test_labels) > 1e-14)[0].shape[
            0
        ]
        label_acuracy = 1 - nwrong / test_labels.shape[0]
        print("NWRONG", nwrong)
        print("Label accuracy", label_acuracy)
        assert nwrong == 1

        # import matplotlib.pyplot as plt
        # ax = plt.figure().gca()
        # # ax.plot(train_samples[0], train_values, 'X')
        # ax.plot(test_samples[0], test_labels, 'o')
        # ax.plot(test_samples[0], clf_labels, 'X')
        # clf.plot_surface(ax, [0, 1])
        # plt.show()


if __name__ == "__main__":
    unittest.main(verbosity=2)
