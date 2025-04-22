import unittest
import numpy as np
from scipy import stats
from scipy.special import jv as bessel_function

from pyapprox.surrogates.affine.low_rank_multifidelity import (
    expected_l2_error,
    BiFidelityModel,
)
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.univariate.orthopoly import LegendrePolynomial1D
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.variables.transforms import AffineTransform
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.interface.model import Model
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


class OscillatoryPolyModel(Model):
    def __init__(self, mesh_ndof=100, nterms=35, eps=1e-3, backend=NumpyMixin):
        super().__init__(backend)
        self.set_eps(eps)
        self._mesh = self._bkd.linspace(-1.0, 1.0, mesh_ndof)
        self._nterms = nterms

        # define random variable
        self._variable = IndependentMarginalsVariable(
            [stats.uniform(0, 10 * np.pi)], backend=self._bkd
        )

        # polynomial defined over spatial domain [-1, 1]
        trans = AffineTransform(
            IndependentMarginalsVariable(
                [stats.uniform(-1, 2)], backend=self._bkd
            ),
            enforce_bounds=True,
        )
        polys_1d = [
            LegendrePolynomial1D(trans=trans, backend=self._bkd)
            for ii in range(trans.variable().nvars())
        ]
        basis = OrthonormalPolynomialBasis(polys_1d)
        basis.set_indices(self._bkd.arange(self._nterms)[None, :])
        self.poly = PolynomialChaosExpansion(basis)

    def set_eps(self, eps):
        self._eps = eps

    def nqoi(self):
        return self._mesh.shape[0]

    def nvars(self):
        return 1

    def _abs_z(self, z):
        return self._bkd.abs(z + self._eps * z**2)

    def _basis_matrix(self):
        return self.poly.basis()(self._mesh.reshape(1, self._mesh.shape[0]))

    def _values(self, samples):
        z = samples[0, :]
        basis_matrix = self._basis_matrix()
        coeffs = self._bkd.zeros((self._nterms, samples.shape[1]))
        abs_z = self._abs_z(z)
        for k in range(self._nterms):
            ck = self._bkd.exp(self._bkd.sign(z) * 1j) * 1j**k
            ck = ck.real
            gk = (
                ck
                * self._bkd.sqrt(np.pi * (2.0 * k + 1.0) / abs_z)
                # autograd will not work because we are calling scipy function
                # here because it does not exist in torch
                * self._bkd.asarray(bessel_function(k + 0.5, abs_z))
            )
            # gk not defined for z=0
            coeffs[k, :] = gk
            # must divide by sqrt(2), due to using orthonormal basis with
            # respect to w=1/2, but needing orthonormal basis with respect
            # to w=1
            coeffs[k, :] /= np.sqrt(2)

        result = basis_matrix @ coeffs
        return result.T

    def variable(self):
        return self._variable

    def mesh(self):
        return self._mesh


class OscillatorySinLowFidelityModel(OscillatoryPolyModel):
    def __init__(self, mesh_dof=100, nterms=35, backend=NumpyMixin):
        super().__init__(mesh_dof, nterms, 0, backend)

    def _basis_matrix(self):
        kk = self._bkd.arange(self._nterms)[None, :]
        return self._bkd.sin(np.pi * (kk + 1) * self._mesh[:, None])


class TestLowRankMultiFidelity:
    def setUp(self):
        np.random.seed(1)

    def test_oscillatory_model(self):
        bkd = self.get_backend()
        eps = 1.0e-3
        mesh_dof = 100
        K = 35
        hf_model = OscillatoryPolyModel(mesh_dof, 100, eps, backend=bkd)
        lf_model = OscillatorySinLowFidelityModel(mesh_dof, K, backend=bkd)

        nlf_candidates = int(1e4)
        nhf_runs = 20
        ntest_samples = int(1e3)
        test_samples = hf_model.variable().rvs(ntest_samples)
        hf_test_values = hf_model(test_samples)

        mf_model = BiFidelityModel(backend=bkd)
        # 1. Evaluate the low-fidelity model u_L on a candidate set Gamma.
        lf_samples = hf_model.variable().rvs(nlf_candidates)
        mf_model.build(lf_model, hf_model, lf_samples, nhf_runs)
        mf_test_values = mf_model(test_samples)

        error_mf = expected_l2_error(hf_test_values, mf_test_values)[1]
        print(error_mf)
        assert error_mf < 1e-4  # 3.04e-05)


class TestNumpyLowRankMultiFidelity(
    TestLowRankMultiFidelity, unittest.TestCase
):
    def get_backend(self):
        return NumpyMixin


class TestTorchLowRankMultiFidelity(
    TestLowRankMultiFidelity, unittest.TestCase
):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
