import numpy as np

from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.util.linalg import PivotedCholeskyFactorizer
from pyapprox.interface.model import Model
from pyapprox.util.backends.numpy import NumpyMixin


class BiFidelityModel(Model):
    def nqoi(self) -> int:
        return self._selected_hf_values.shape[1]

    def nvars(self) -> int:
        return 1

    def select_samples(
        self, nhf_samples: int, lf_samples: Array, lf_values: Array
    ) -> Array:
        # Choose an ordered subset of N nodes gamma using Algorithm 1.
        V = lf_values.T
        self._factorizer = PivotedCholeskyFactorizer(V.T @ V, bkd=self._bkd)
        self._factorizer.factorize(nhf_samples)
        self._selected_lf_values = lf_values[self._factorizer.pivots()]
        return lf_samples[:, self._factorizer.pivots()]

    def synthesis(self, lf_values: Array) -> Array:
        # Algorithm 2. Algorithmic evaluation of the synthesis operation
        # G_inv is $G^{-1} = u^L(\gamma)^{-1} u^L(\gamma)^{-T}$
        rhs = self._selected_lf_values @ lf_values.T
        coefs = self._factorizer.solve_linear_system(rhs)
        return (self._selected_hf_values.T @ coefs).T

    def set_high_fidelity_values(self, selected_hf_values: Array):
        self._selected_hf_values = selected_hf_values

    def set_low_fidelity_model(self, lf_model: Model):
        self._lf_model = lf_model

    def set_high_fidelity_model(self, hf_model: Model):
        self._hf_model = hf_model

    def _values(self, samples: Array) -> Array:
        if not hasattr(self, "_lf_model"):
            raise ValueError("must call set_low_fidelity_model")
        lf_values = self._lf_model(samples)
        return self.synthesis(lf_values)

    def build(
        self,
        lf_model: Model,
        hf_model: Model,
        lf_samples: Array,
        nhf_samples: int,
    ):
        self.set_low_fidelity_model(lf_model)
        self.set_high_fidelity_model(hf_model)
        # 1. Evaluate the low-fidelity model u_L on a candidate set Gamma.
        lf_values = lf_model(lf_samples)
        # 2. Choose an ordered subset of N nodes gamma using Algorithm 1.
        selected_samples = self.select_samples(
            nhf_samples, lf_samples, lf_values
        )
        # 3. Evaluate the high-fidelity u_H model on gamma.
        hf_values = hf_model(selected_samples)
        self.set_high_fidelity_values(hf_values)


def expected_l2_error(
    solutions: Array, true_solutions: Array, bkd: BackendMixin = NumpyMixin
):
    error = bkd.norm(true_solutions - solutions, axis=1)
    abs_error = bkd.mean(error)
    rel_error = bkd.mean(error / bkd.norm(true_solutions, axis=1))
    return abs_error, rel_error
