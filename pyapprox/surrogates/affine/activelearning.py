from pyapprox.util.backends.template import Array
from pyapprox.surrogates.sampler import CandidateSampler
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from scipy.linalg import qr


class FeketeSampler(CandidateSampler):
    def set_surrogate(self, surrogate: PolynomialChaosExpansion):
        if not isinstance(surrogate, PolynomialChaosExpansion):
            raise ValueError(
                "surrogate must be an instance of PolynomialChaosExpansion"
            )
        if surrogate.basis().nterms() == 0:
            raise ValueError("must call surrogate.basis().set_indices()")
        self._surrogate = surrogate

    def _christoffel_function(self, basis_mat: Array) -> Array:
        return (
            self._bkd.sum(basis_mat**2, axis=1)[:, None] / self._ntrain_samples
        )

    def _generate_new_samples(self, nsamples: int) -> Array:
        if self._ntrain_samples > 0:
            raise RuntimeError(
                "{0} does not generate nested sequences".format(self)
                + " but samples have already been computed"
            )
        if nsamples != self._surrogate.basis().nterms():
            print(
                "nsamples is determined by surrogate.basis().nterms()"
                "overiding to nsamples={0}".format(
                    self._surrogate.basis().nterms()
                )
            )
            nsamples = self._surrogate.basis().nterms()
        self._ntrain_samples = nsamples
        basis_mat = self._surrogate.basis()(self._candidate_samples)
        self._candidate_weights = self._bkd.sqrt(
            self._christoffel_function(basis_mat)
        )
        # use scipy because it allows pivoting
        mat = self._bkd.to_numpy(self._candidate_weights * basis_mat)
        Q, R, pivots = qr(mat.T, pivoting=True)
        self._Q = self._bkd.asarray(Q)
        self._R = self._bkd.asarray(R[:, : basis_mat.shape[1]])
        self._pivots = self._bkd.asarray(
            pivots[: basis_mat.shape[1]], dtype=int
        )
        self._weights = self._candidate_weights[self._pivots]
        return self._candidate_samples_subset(self._pivots)

    def interpolatory_coefficients(self, values: Array) -> Array:
        temp = self._bkd.solve_triangular(
            self._R.T, (self._weights * values), lower=True
        )
        print(temp.shape, self._Q.shape)
        coef = self._Q @ temp
        return coef
