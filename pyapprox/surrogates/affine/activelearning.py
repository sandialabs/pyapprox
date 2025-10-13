from scipy.linalg import qr

from pyapprox.util.backends.template import Array
from pyapprox.surrogates.sampler import CandidateSampler
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.util.linalg import PivotedLUFactorizer
from pyapprox.surrogates.affine.basis import Basis
from pyapprox.variables.joint import JointVariable
from pyapprox.surrogates.regressor import AdaptiveRegressorMixin

# todo move schedules to regressor
from pyapprox.surrogates.gaussianprocess.activelearning import (
    SamplingSchedule,
    ConstantSamplingSchedule,
)
from pyapprox.surrogates.affine.multiindex import IterativeIndexGenerator


class ChristoffelMixin:
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


class FeketeSampler(ChristoffelMixin, CandidateSampler):
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
        coef = self._Q @ temp
        return coef


class LejaSampler(ChristoffelMixin, CandidateSampler):

    def _generate_new_samples(self, nsamples: int) -> Array:
        new_pivots = self._update_pivots(nsamples)
        new_samples = self._candidate_samples_subset(new_pivots)
        self._train_samples = self._candidate_samples_subset(self.pivots())
        self._update_ntrain_samples()
        self._weights = self._candidate_weights[self.pivots()]
        return new_samples

    def _update_ntrain_samples(self):
        self._ntrain_samples = self._train_samples.shape[1]

    def set_initial_pivots(self, init_pivots: Array):
        self._init_pivots = init_pivots

    def _get_init_pivots(self) -> Array:
        if hasattr(self, "_init_pivots"):
            if self._init_pivots.shape[0] > self._surrogate.basis().nterms():
                # this conditioned can be removed if code is changed
                # to allow init pivots to be passed to update pivots
                # but I do not think this case needs to be supported
                raise ValueError(
                    "too many init_pivots given. "
                    "Increase nterms of initial basis"
                )
            return self._init_pivots
        return None

    def _init_factorization(self, nprev_train_samples: int, nsamples: int):
        basis_mat = self._surrogate.basis()(self._candidate_samples)
        # set ntrain samples for christoffel_function
        self._ntrain_samples = nsamples
        self._candidate_weights = self._bkd.sqrt(
            self._christoffel_function(basis_mat)
        )
        mat = self._candidate_weights * basis_mat
        self._LUFactorizer = PivotedLUFactorizer(mat, bkd=self._bkd)

        self._L, self._U = self._LUFactorizer.factorize(
            nsamples, init_pivots=self._get_init_pivots()
        )
        return self._LUFactorizer.pivots()[nprev_train_samples:nsamples]

    def _update_pivots(self, nsamples: int) -> Array:
        nprev_train_samples = self.ntrain_samples()
        if nsamples > self._surrogate.basis().nterms():
            print(
                "nsamples will produce and underdetermined system"
                "overiding to nsamples={0}".format(
                    self._surrogate.basis().nterms()
                )
            )
            nsamples = self._surrogate.basis().nterms()

        if nprev_train_samples == 0:
            return self._init_factorization(nprev_train_samples, nsamples)

        # todo only compute new basis columns
        nnew_basis = nsamples - nprev_train_samples
        prev_candidate_weights = self._bkd.copy(self._candidate_weights)
        basis_mat = self._surrogate.basis()(self._candidate_samples)
        # must update weights when new rows are added as they
        # effect value of christoffel_function
        # set ntrain samples for christoffel_function
        self._ntrain_samples = nsamples
        self._candidate_weights = self._bkd.sqrt(
            self._christoffel_function(basis_mat)
        )
        # we can take away old weights while using new ones by feeding in ratio
        # to prev_candidate_weights
        self._LUFactorizer.update_preconditioning(
            prev_candidate_weights,
            self._candidate_weights,
            self._LUFactorizer._ncompleted_pivots,
            update_internal_state=True,
        )
        mat = self._candidate_weights * basis_mat
        self._LUFactorizer.add_columns(mat[:, -nnew_basis:])
        self._LUFactorizer.update(nsamples)
        self._L, self._U = self._LUFactorizer._split_lu(
            self._LUFactorizer._LU_factor,
            self._LUFactorizer._ncompleted_pivots,
        )
        return self._LUFactorizer.pivots()[nprev_train_samples:nsamples]

    def pivots(self) -> Array:
        return self._LUFactorizer.pivots()

    def interpolatory_coefficients(self, values: Array) -> Array:
        temp = self._bkd.solve_triangular(
            self._L, (self._weights * values), lower=True
        )
        coef = self._bkd.solve_triangular(self._U, temp, lower=False)
        return coef


class AdaptivePolynomialChaosExpansion(
    PolynomialChaosExpansion, AdaptiveRegressorMixin
):
    def __init__(
        self,
        basis: Basis,
        variable: JointVariable,
        index_generator: IterativeIndexGenerator,
        max_nsamples: int,
        nqoi=1,
    ):

        super().__init__(basis, None, nqoi, None, True)
        self._sampler = LejaSampler(variable)
        self._sampler.set_surrogate(self)
        self._max_nsamples = max_nsamples
        self._gen = index_generator

    def _init_sample_step(self):
        indices = self._gen.get_indices()
        if indices.shape[1] == 0:
            # assume generator starts with a nonzero number of indices
            raise ValueError("indices must not be empty")
        nnew_samples = indices.shape[1]
        self._basis.set_indices(indices)
        new_samples = self._sampler(
            self._sampler.ntrain_samples() + nnew_samples
        )
        self._train_samples_history = new_samples
        return new_samples

    def step_samples(self) -> Array:
        if not hasattr(self, "_train_samples_history"):
            return self._init_sample_step()

        nprev_indices = self._gen.get_indices().shape[1]
        if nprev_indices >= self._max_nsamples:
            return None

        self._gen.step()
        indices = self._gen.get_indices()
        nnew_samples = indices.shape[1] - nprev_indices
        self._basis.set_indices(indices)
        new_samples = self._sampler(
            self._sampler.ntrain_samples() + nnew_samples
        )
        # TODO make sure there is not an inconsitency
        # using ctrain samples and how leja samples are generated
        self._train_samples_history = self._bkd.hstack(
            (self._train_samples_history, new_samples)
        )
        return new_samples

    def step_values(self, new_values: Array):
        if not hasattr(self, "_train_values_history"):
            self._train_values_history = new_values
        else:
            self._train_values_history = self._bkd.vstack(
                (self._train_values_history, new_values)
            )
        self.fit(self._train_samples_history, self._train_values_history)

    def _fit(self, iterate: Array):
        self.set_coefficients(
            self._sampler.interpolatory_coefficients(self._ctrain_values)
        )

    def sampler(self) -> LejaSampler:
        return self._sampler
