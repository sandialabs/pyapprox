from abc import ABC, abstractmethod

from pyapprox.util.backends.template import Array
from pyapprox.expdesign.sequences import HaltonSequence
from pyapprox.surrogates.regressor import Surrogate
from pyapprox.variables.joint import IndependentMarginalsVariable


class CandidateSampler(ABC):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
    ):
        if not isinstance(variable, IndependentMarginalsVariable):
            raise ValueError(
                "variable must be an instance of IndependentMarginalsVariable"
            )

        self._bkd = variable._bkd
        self._variable = variable
        self._train_samples = self._bkd.zeros((self._variable.nvars(), 0))
        self._ntrain_samples = 0
        self._initialized = False

    @abstractmethod
    def set_surrogate(self, surrogate: Surrogate):
        raise NotImplementedError

    def default_candidate_samples(self, ncandidates: int) -> Array:
        nhalton_candidates = ncandidates // 2
        nrandom_candidates = ncandidates - nhalton_candidates
        seq = HaltonSequence(self._variable.nvars(), variable=self._variable)
        halton_samples = seq.rvs(nhalton_candidates)
        random_samples = self._variable.rvs(nrandom_candidates)
        return self._bkd.hstack((halton_samples, random_samples))

    def set_candidate_samples(self, candidate_samples: Array):
        if not hasattr(self, "_surrogate"):
            raise RuntimeError("Must call set_surrogate first")
        self._candidate_samples = candidate_samples
        self._canonical_candidate_samples = (
            self._surrogate._in_trans.map_to_canonical(self._candidate_samples)
        )
        self._candidate_samples_changed = True

    def ncandidates(self) -> int:
        return self._candidate_samples.shape[1]

    def ntrain_samples(self) -> int:
        return self._ntrain_samples

    def _setup_before_first_step(self):
        if not hasattr(self, "_surrogate"):
            raise RuntimeError("Must call set_surrogate")
        if not hasattr(self, "_candidate_samples"):
            self.set_candidate_samples(self.default_candidate_samples(2000))
        self._initialized = True

    def __repr__(self) -> int:
        if hasattr(self, "_surrogate"):
            return "{0}(ncandidates={1}, surrogate={2})".format(
                self.__class__.__name__, self.ncandidates(), self._surrogate
            )
        if not hasattr(self, "_ncandidates"):
            return "{0}".format(self.__class__.__name__)
        return "{0}(ncandidates={1})".format(
            self.__class__.__name__, self.ncandidates()
        )

    def _candidate_samples_subset(self, indices: Array) -> Array:
        return self._candidate_samples[:, indices]

    @abstractmethod
    def _generate_new_samples(self) -> Array:
        raise NotImplementedError

    def __call__(self, nsamples: int) -> Array:
        if not self._initialized:
            self._setup_before_first_step()
        if nsamples <= self.ntrain_samples():
            msg = f"Requesting number of samples {nsamples} which is less "
            "than number of train samples already generated "
            f"{self.ntrain_samples()}"
            raise ValueError(msg)
        return self._generate_new_samples(nsamples)
