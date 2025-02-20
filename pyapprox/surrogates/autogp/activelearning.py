from abc import ABC, abstractmethod
from typing import Union, List

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.linearalgebra.linalg import PivotedCholeskyFactorizer
from pyapprox.expdesign.sequences import HaltonSequence
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.regressor import AdaptiveRegressorMixin
from pyapprox.surrogates.bases.basisexp import BasisExpansion
from pyapprox.surrogates.kernels.kernels import Kernel


class CholeskySampler:
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        nugget: float = 0.0,
        econ: bool = True,
    ):
        if not isinstance(variable, IndependentMarginalsVariable):
            raise ValueError(
                "variable must be an instance of IndependentMarginalsVariable"
            )
        self._bkd = variable._bkd
        self._variable = variable
        self._nugget = nugget
        self._econ = econ
        self._train_samples = self._bkd.zeros((self._variable.nvars(), 0))
        self._kernel_changed = True

    def default_candidate_samples(self, ncandidates: int) -> Array:
        nhalton_candidates = ncandidates // 2
        nrandom_candidates = ncandidates - nhalton_candidates
        seq = HaltonSequence(self._variable.nvars(), variable=self._variable)
        halton_samples = seq.rvs(nhalton_candidates)
        random_samples = self._variable.rvs(nrandom_candidates)
        return self._bkd.hstack((halton_samples, random_samples))

    def set_candidate_samples(
        self, candidate_samples: Array, init_pivots: Array = None
    ):
        if not hasattr(self, "_gp"):
            raise RuntimeError("Must call set_gaussian_process first")
        self._candidate_samples = candidate_samples
        self._canonical_candidate_samples = (
            self._gp._in_trans.map_to_canonical(self._candidate_samples)
        )
        self._candidate_samples_changed = True
        self._set_initial_pivots(init_pivots)

    def set_weight_function(self, weight_function: callable):
        self._weight_function = weight_function
        # weight function must take samples in user space (not canonical space)
        # This is the correct thing to do even though kernel is evaluated in
        # canonical space (see note in restart factorization). User should not
        # have to know about canonical space.
        self._weight_function_changed = True

    def set_gaussian_process(self, gp):
        self._gp = gp

    def _set_initial_pivots(self, init_pivots: Array):
        if not hasattr(self, "_candidate_samples"):
            raise RuntimeError("must call set_candidate_samples first")
        self._init_pivots = init_pivots
        if init_pivots is not None:
            # return samples in user space
            self._train_samples = self._candidate_samples[:, self._init_pivots]
        self._init_pivots_changed = True

    def ntrain_samples(self) -> int:
        return self._train_samples.shape[1]

    def _add_nugget(self):
        self._Amat[
            self._bkd.arange(self._Amat.shape[0]),
            self._bkd.arange(self._Amat.shape[1]),
        ] += self._nugget

    def _restart_factorization(self, nsamples: int):
        # GP optimizes kernel hyperparameters when the training samples
        # are mapped to the canonical space. So, to be consistent
        # must used canonical candidate samples to evaluate kernel
        if self._kernel_changed:
            self._Kmatrix = self._gp.kernel()(
                self._canonical_candidate_samples
            )
        if self._weight_function_changed:
            print(self._weight_function)
            self._pivot_weights = self._weight_function(
                self._candidate_samples
            )
            if self._pivot_weights.ndim != 1:
                raise RuntimeError("weight_function must return 1D array")

        if not self._econ:
            sqrt_weights = self._bkd.sqrt(self._pivot_weights)
            self.Kmatrix = sqrt_weights[:, None] * self._Kmatrix * sqrt_weights
            self._pivot_weights = None
        self._factorizer = PivotedCholeskyFactorizer(
            self._Kmatrix, econ=self._econ, bkd=self._bkd
        )
        try:
            self._factorizer.factorize(
                nsamples,
                init_pivots=self._init_pivots,
                pivot_weights=self._pivot_weights,
            )
        except RuntimeError:
            raise RuntimeError(
                "Too many samples requested. Likely need to reduce "
                "kernel lenscale to generated the specified number of samples."
                f" Alternatively request at most {self._factorizer.npivots()} "
                "total number samples"
            )

        self._weight_function_changed = False
        self._kernel_changed = False
        self._init_pivots_changed = False
        self._candidate_samples_changed = False

    def _update_factorization(self, nsamples: int):
        try:
            self._factorizer.update(nsamples)
        except RuntimeError:
            raise RuntimeError(
                "Too many samples requested. Likely need to reduce "
                "kernel lenscale to generated the specified number of samples."
                f" Alternatively request at most {self._factorizer.npivots()} "
                "total number samples"
            )

    def _uniform_weight_function(self, samples: Array) -> Array:
        return self._bkd.ones(samples.shape[1])

    def _setup_before_first_step(self):
        if not hasattr(self, "_candidate_samples"):
            self.set_candidate_samples(
                self.default_candidate_samples(1000), None
            )
        if not hasattr(self, "_weight_function"):
            self.set_weight_function(self._uniform_weight_function)

    def __call__(self, nsamples: int):
        if not hasattr(self, "_gp"):
            raise RuntimeError("Must call set_gaussian_process")
        if self.ntrain_samples() == 0:
            self._setup_before_first_step()

        if nsamples < self.ntrain_samples():
            msg = f"Requesting number of samples {nsamples} which is less "
            "than number of train samples already generated "
            f"{self.ntrain_samples()}"
            raise ValueError(msg)

        if (
            self._weight_function_changed
            or self._kernel_changed
            or self._init_pivots_changed
            or self._candidate_samples_changed
        ):
            self._restart_factorization(nsamples)
        else:
            self._update_factorization(nsamples)

        nprev_train_samples = self.ntrain_samples()
        # return samples in user space
        new_samples = self._candidate_samples[
            :, self._factorizer.pivots()[nprev_train_samples:nsamples]
        ]
        self._train_samples = self._bkd.hstack(
            (self._train_samples, new_samples)
        )
        # update pivots so that they are available if kernel matrix
        # needs to be recomputed
        self._set_initial_pivots(self._factorizer.pivots())
        return new_samples


class SamplingSchedule(ABC):
    def __init__(self):
        self._nsamples_list = []

    @abstractmethod
    def _nnew_samples(self) -> int:
        raise NotImplementedError

    def nnew_samples(self) -> int:
        nsamples_inc = self._nnew_samples()
        self._nsamples_list.append(nsamples_inc)
        return nsamples_inc

    def sample_schedule(self) -> Array:
        """Return the sample increments computed so far"""
        return self.array(self._nsamples_list)


class ConstantSamplingSchedule(SamplingSchedule):
    def __init__(self, nsamples_increment: int, max_nsamples: int):
        super().__init__()
        self._nsamples = 0
        self._max_nsamples = max_nsamples
        self._nsamples_increment = nsamples_increment

    def _nnew_samples(self) -> int:
        if self._nsamples >= self._max_nsamples:
            return 0
        self._nsamples += self._nsamples_increment
        return self._nsamples_increment


class SamplingScheduleFromList(SamplingSchedule):
    def __init__(self, nbatch_samples_list: Union[List, Array]):
        super().__init__()
        self._idx = 0
        self._nbatch_samples_list = nbatch_samples_list

    def _nnew_samples(self) -> int:
        if self._idx >= len(self._nbatch_samples_list):
            return 0
        nnew_samples = self._nbatch_samples_list[self._idx]
        self._idx += 1
        return nnew_samples


class AdaptiveGaussianProcess(ExactGaussianProcess, AdaptiveRegressorMixin):
    def __init__(
        self,
        nvars: int,
        kernel: Kernel,
        trend: BasisExpansion = None,
        kernel_reg: float = 0,
        sampling_schedule=ConstantSamplingSchedule(10, 100),
    ):
        super().__init__(nvars, kernel, trend, kernel_reg)
        self._sampling_schedule = sampling_schedule

    def set_sampler(self, sampler: CholeskySampler):
        self._sampler = sampler
        self._sampler.set_gaussian_process(self)

    def step_samples(self) -> Array:
        if not hasattr(self, "_sampler"):
            raise ValueError("must call set_sampler")
        nnew_samples = self._sampling_schedule.nnew_samples()
        if nnew_samples == 0:
            return None
        new_samples = self._sampler(
            self._sampler.ntrain_samples() + nnew_samples
        )
        # TODO make sure there is not an inconsitency
        # using ctrain samples and how kernel samples are generated
        if not hasattr(self, "_train_samples_history"):
            self._train_samples_history = new_samples
        else:
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
        self._sampler._kernel_changed = True
