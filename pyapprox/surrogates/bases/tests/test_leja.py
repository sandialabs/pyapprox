import unittest

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.bases.leja import (
    PDFLejaObjective, BetaMarginal, UniformMarginal, GaussianMarginal,
    LejaSequence
)
from pyapprox.surrogates.bases.orthopoly import (
    setup_univariate_orthogonal_polynomial_from_marginal
)
from pyapprox.interface.model import ModelFromCallable
from pyapprox.optimization.pya_minimize import ScipyConstrainedOptimizer


class TestLeja:
    def setUp(self):
        np.random.seed(1)

    def test_marginals(self):
        bkd = self.get_backend()
        # use model infasttucture to test gradients
        marginal = BetaMarginal(stats.beta(2, 2), backend=bkd)
        model = ModelFromCallable(
            1, marginal.pdf, jacobian=marginal.pdf_jacobian, backend=bkd
        )
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])
        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = model.check_apply_jacobian(iterate, fd_eps=fd_eps)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.1

        marginal = GaussianMarginal(stats.norm(0, 1), backend=bkd)
        model = ModelFromCallable(
            1, marginal.pdf, jacobian=marginal.pdf_jacobian, backend=bkd
        )
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])
        errors = model.check_apply_jacobian(iterate)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.1

        marginal = UniformMarginal(stats.uniform(-1, 2), backend=bkd)
        model = ModelFromCallable(
            1, marginal.pdf, jacobian=marginal.pdf_jacobian, backend=bkd
        )
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])
        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = model.check_apply_jacobian(
            iterate, relative=False, fd_eps=fd_eps
        )
        # gradient is zero so check its small for large finite difference step
        assert errors[0] < 1e-14

    def test_pdf_leja_objective(self):
        bkd = self.get_backend()
        scipy_marginal = stats.beta(2, 2, -1, 2)
        poly = setup_univariate_orthogonal_polynomial_from_marginal(
            scipy_marginal, backend=bkd)
        marginal = BetaMarginal(scipy_marginal, backend=bkd)
        obj = PDFLejaObjective(marginal, poly)
        obj.set_sequence(bkd.array([[-.95, 0]]))
        iterate = bkd.array([[marginal._marginal.interval(0.75)[1]]])
        # make sure not to step outside bounds
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = obj.check_apply_jacobian(iterate, fd_eps=fd_eps)
        assert errors.min()/errors.max() < 1e-6 and errors.max() > 0.08

    def test_pdf_leja_sequence(self):
        bkd = self.get_backend()
        scipy_marginal = stats.beta(1, 1, -1, 2)
        poly = setup_univariate_orthogonal_polynomial_from_marginal(
            scipy_marginal, backend=bkd)
        marginal = BetaMarginal(scipy_marginal, backend=bkd)
        obj = PDFLejaObjective(marginal, poly)
        obj.set_sequence(bkd.array([[scipy_marginal.mean()]]))

        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(
            gtol=1e-8, maxiter=1000, method="trust-constr"
        )
        leja = LejaSequence(obj, optimizer)

        # test plotting of objective before a single new point is added
        ax = plt.figure().gca()
        leja.plot(ax)
        leja.step(nsamples=2)
        ax.axvline(x=leja.sequence()[0, -1])

        # test adding multiple samples
        leja.step(nsamples=10)

        # test quadrature weights
        quad_weights = leja.quadrature_weights(leja.sequence())
        print(quad_weights)
        integral = (leja.sequence()**4) @ quad_weights
        exact_integral = 1./5.
        print(integral, exact_integral)
        assert np.allclose(integral, exact_integral)


class TestNumpyLeja(TestLeja, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
class TestTorchLeja(TestLeja, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
