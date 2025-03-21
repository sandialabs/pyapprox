from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.linearalgebra.linalg import PivotedCholeskyFactorizer
from pyapprox.expdesign.sequences import HaltonSequence
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.autogp.stats import KernelStatistics
from pyapprox.surrogates.regressor import AdaptiveRegressorMixin
from pyapprox.surrogates.bases.basisexp import BasisExpansion
from pyapprox.surrogates.kernels.kernels import Kernel


class GreedyGaussianProcessSampler(ABC):
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
        self._pivots = None
        self._kernel_changed = True
        self._initialized = False
        self._init_pivots = None

    def default_candidate_samples(self, ncandidates: int) -> Array:
        nhalton_candidates = ncandidates // 2
        nrandom_candidates = ncandidates - nhalton_candidates
        seq = HaltonSequence(self._variable.nvars(), variable=self._variable)
        halton_samples = seq.rvs(nhalton_candidates)
        random_samples = self._variable.rvs(nrandom_candidates)
        return self._bkd.hstack((halton_samples, random_samples))

    def set_candidate_samples(self, candidate_samples: Array):
        if not hasattr(self, "_gp"):
            raise RuntimeError("Must call set_gaussian_process first")
        self._candidate_samples = candidate_samples
        self._canonical_candidate_samples = (
            self._gp._in_trans.map_to_canonical(self._candidate_samples)
        )
        self._candidate_samples_changed = True

    def ncandidates(self) -> int:
        return self._candidate_samples.shape[1]

    def set_gaussian_process(self, gp):
        self._gp = gp

    def set_initial_pivots(self, init_pivots: Array):
        if not hasattr(self, "_candidate_samples"):
            raise RuntimeError("must call set_candidate_samples first")
        self._init_pivots = init_pivots

    def ntrain_samples(self) -> int:
        return self._train_samples.shape[1]

    def _add_nugget(self):
        self._Kmatrix[
            self._bkd.arange(self._Kmatrix.shape[0]),
            self._bkd.arange(self._Kmatrix.shape[1]),
        ] += self._nugget

    @abstractmethod
    def pivots(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _update_pivots(self, nsamples: int) -> Array:
        raise NotImplementedError

    def _setup_before_first_step(self):
        if not hasattr(self, "_gp"):
            raise RuntimeError("Must call set_gaussian_process")
        if not hasattr(self, "_candidate_samples"):
            self.set_candidate_samples(self.default_candidate_samples(2000))
        self._initialized = True

    def __call__(self, nsamples: int):
        if not self._initialized:
            self._setup_before_first_step()
        if nsamples < self.ntrain_samples():
            msg = f"Requesting number of samples {nsamples} which is less "
            "than number of train samples already generated "
            f"{self.ntrain_samples()}"
            raise ValueError(msg)
        new_pivots = self._update_pivots(nsamples)
        # return samples in user space
        new_samples = self._candidate_samples[:, new_pivots]
        self._train_samples = self._bkd.hstack(
            (self._train_samples, new_samples)
        )
        self._train_samples = self._candidate_samples[:, self.pivots()]
        return new_samples


class CholeskySampler(GreedyGaussianProcessSampler):

    def set_initial_pivots(self, init_pivots: Array):
        super().set_initial_pivots(init_pivots)
        if init_pivots is not None:
            # return samples in user space
            self._train_samples = self._candidate_samples[:, self._init_pivots]
            self._pivots = self._init_pivots

    def _restart_factorization(self, nsamples: int):
        # GP optimizes kernel hyperparameters when the training samples
        # are mapped to the canonical space. So, to be consistent
        # must used canonical candidate samples to evaluate kernel
        if self._kernel_changed:
            self._Kmatrix = self._gp.kernel()(
                self._canonical_candidate_samples
            )
            self._add_nugget()
        if self._weight_function_changed:
            self._pivot_weights = self._weight_function(
                self._candidate_samples
            )
            if self._pivot_weights.ndim != 1:
                raise RuntimeError("weight_function must return 1D array")

        if not self._econ:
            sqrt_weights = self._bkd.sqrt(self._pivot_weights)
            self._Kmatrix = (
                sqrt_weights[:, None] * self._Kmatrix * sqrt_weights
            )
            self._pivot_weights = None
        self._factorizer = PivotedCholeskyFactorizer(
            self._Kmatrix, econ=self._econ, bkd=self._bkd
        )
        try:
            self._factorizer.factorize(
                nsamples,
                init_pivots=self._pivots,
                pivot_weights=self._pivot_weights,
            )
            self._pivots = self._factorizer.pivots()
        except RuntimeError:
            raise RuntimeError(
                "Too many samples requested. Likely need to reduce "
                "kernel lenscale to generated the specified number of samples."
                " Alternatively request at most "
                f"{self._factorizer.npivots()-1} total number samples"
            )

        self._weight_function_changed = False
        self._kernel_changed = False
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

    def set_weight_function(self, weight_function: callable):
        self._weight_function = weight_function
        # weight function must take samples in user space (not canonical space)
        # This is the correct thing to do even though kernel is evaluated in
        # canonical space (see note in restart factorization). User should not
        # have to know about canonical space.
        self._weight_function_changed = True

    def _setup_before_first_step(self):
        super()._setup_before_first_step()
        if not hasattr(self, "_weight_function"):
            self.set_weight_function(self._uniform_weight_function)

    def _update_pivots(self, nsamples: int) -> Array:
        if (
            not self._initialized
            or self._weight_function_changed
            or self._kernel_changed
            or self._candidate_samples_changed
        ):
            self._restart_factorization(nsamples)
        else:
            self._update_factorization(nsamples)
        nprev_train_samples = self.ntrain_samples()
        return self._factorizer.pivots()[nprev_train_samples:nsamples]

    def pivots(self) -> Array:
        return self._factorizer.pivots()


class GreedyIntegratedVarianceSampler(GreedyGaussianProcessSampler):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        nugget: float = 0.0,
        nquad_nodes_1d: List[int] = None,
    ):
        super().__init__(variable, nugget, True)
        self._nquad_nodes_1d = nquad_nodes_1d

    def _priorities(self) -> Array:
        if self._L_inv.shape[0] == 0:
            return -self._bkd.diag(self._P) / self._bkd.diag(self._Kmatrix)

        A_12 = self._bkd.atleast2d(self._Kmatrix[self._pivots, :])
        L_12 = self._bkd.solve_triangular(self._L, A_12, lower=True)
        J = self._bkd.where(
            (
                self._bkd.diag(self._Kmatrix)
                - self._bkd.sum(L_12 * L_12, axis=0)
            )
            <= 0
        )[0]
        self._temp = self._bkd.diag(self._Kmatrix) - self._bkd.sum(
            L_12 * L_12, axis=0
        )
        useful_candidates = self._bkd.ones(
            (self._candidate_samples.shape[1]), dtype=bool
        )
        useful_candidates[J] = False
        useful_candidates[self._pivots] = False
        L_12 = L_12[:, useful_candidates]
        L_22 = self._bkd.sqrt(
            self._bkd.diag(self._Kmatrix)[useful_candidates]
            - self._bkd.sum(L_12 * L_12, axis=0)
        )

        P_11 = self._P[np.ix_(self._pivots, self._pivots)]
        P_12 = self._P[np.ix_(self._pivots, useful_candidates)]
        P_22 = self._bkd.diag(self._P)[useful_candidates]

        C = -self._bkd.dot((L_12 / L_22).T, self._L_inv)
        vals = self._bkd.full((self._candidate_samples.shape[1],), np.inf)

        # reimplementaion is incomplete. Need to finish porting over old code in gaussianprocess/gaussian_process. Consider creating seperate class for econominal version rather than just providing a flag
        vals[useful_candidates] = -(
            -self._best_priorities[-1]
            + self._bkd.sum(C.T * P_11.dot(C.T), axis=0)
            + 2 * self._bkd.sum(C.T / L_22 * P_12, axis=0)
            + 1 / L_22**2 * P_22
        )

        return vals

    def _update_cholesky_factorization(self, pivot: int):
        if self._L_inv.shape[0] == 0:
            self._L = self._bkd.atleast2d(
                self._bkd.sqrt(self._Kmatrix[pivot, pivot])
            )
            self._L_inv = 1 / self._L
            return

        A_12 = self._Kmatrix[self._pivots, pivot : pivot + 1]
        L_12 = self._bkd.solve_triangular(self._L, A_12, lower=True)
        L_22_sq = self._Kmatrix[pivot, pivot] - L_12.T.dot(L_12)
        if L_22_sq <= 0:
            # recompute Cholesky from scratch to make sure roundoff error
            # is not causing L_22_sq to be negative
            indices = self._bkd.hstack(
                [self._pivots, self._bkd.array([pivot], dtype=int)]
            )
            try:
                self._L = self._bkd.cholesky(
                    self._Kmatrix[np.ix_(indices, indices)]
                )
            except Exception:
                raise RuntimeError(
                    "cholesky factorization failed. Decrease correlation "
                    "length to add more points"
                )
            self._L_inv = self._bkd.inv(self._L)
            return

        L_22 = self._bkd.sqrt(L_22_sq)

        self._L = self._bkd.block(
            [[self._L, self._bkd.zeros(L_12.shape)], [L_12.T, L_22]]
        )
        indices = self._bkd.hstack(
            [self._pivots, self._bkd.array([pivot], dtype=int)]
        )
        L_22_inv = self._bkd.inv(L_22)
        self._L_inv = self._bkd.block(
            [
                [self._L_inv, self._bkd.zeros(L_12.shape)],
                [-(L_22_inv @ L_12.T) @ self._L_inv, L_22_inv],
            ]
        )

    def _update_pivots(self, nsamples: int) -> Array:
        if self._kernel_changed:
            self._Kmatrix = self._gp.kernel()(
                self._canonical_candidate_samples
            )
            self._add_nugget()

        new_pivots = self._bkd.zeros(
            nsamples - self.ntrain_samples(), dtype=int
        )
        nprev_train_samples = self.ntrain_samples()
        if self._init_pivots is not None:
            for ii in range(
                self.ntrain_samples(),
                min(self._init_pivots.shape[0], nsamples),
            ):
                self._update_cholesky_factorization(self._init_pivots[ii])
                if ii < min(self._init_pivots.shape[0], nsamples) - 1:
                    self._best_priorities.append(None)
                else:
                    self._best_priorities.append(
                        self._brute_force_priority(self._init_pivots[ii])
                    )
                self._pivots = self._bkd.hstack(
                    [self._pivots, self._init_pivots[ii]]
                )

        for ii in range(nsamples - self._pivots.shape[0]):
            priorities = self._priorities()
            new_pivots[ii] = self._bkd.argmin(priorities)
            self._best_priorities.append(priorities[new_pivots[ii]])
            self._update_cholesky_factorization(new_pivots[ii])
            self._pivots = self._bkd.hstack([self._pivots, new_pivots[ii]])
        return self._pivots[nprev_train_samples:nsamples]

    def pivots(self) -> Array:
        return self._pivots

    def _setup_before_first_step(self):
        super()._setup_before_first_step()
        self._stat = KernelStatistics(
            self._gp,
            self._variable,
            self._canonical_candidate_samples,
            self._nquad_nodes_1d,
        )
        self._P = self._stat._tau_P()[1]
        self._Kmatrix = self._gp.kernel()(self._canonical_candidate_samples)
        self._add_nugget()
        if self._pivots is None:
            self._pivots = self._bkd.zeros((0,), dtype=int)

        self._L = self._bkd.zeros((0, 0))
        self._L_inv = self._bkd.zeros((0, 0))
        self._Kmatrix_inv = self._bkd.zeros((0, 0))
        self._best_priorities = []

    def _brute_force_priority(self, new_idx: int) -> float:
        indices = self._bkd.hstack(
            [self._pivots, self._bkd.array([new_idx], dtype=int)]
        )
        Kmat = self._Kmatrix[np.ix_(indices, indices)]
        Pmat = self._P[np.ix_(indices, indices)]
        return -self._bkd.trace(self._bkd.solve(Kmat, Pmat))


class BruteForceGreedyIntegratedVarianceSampler(
    GreedyIntegratedVarianceSampler
):
    # Should only be used for testing default implementation which
    # is much faster but has more complicated code
    def _priorities(self) -> Array:
        priorities = self._bkd.full((self.ncandidates(),), np.inf)
        for mm in range(self.ncandidates()):
            if mm not in self._pivots:
                priorities[mm] = self._brute_force_priority(mm)
        return priorities


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
