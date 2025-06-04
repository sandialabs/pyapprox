from abc import ABC, abstractmethod

import numpy as np

from pyapprox.util.pya_numba import njit
from pyapprox.util.misc import get_first_n_primes
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


class LowDiscrepancySequence(ABC):
    def __init__(
        self,
        nvars: int,
        start_idx: int = 0,
        variable: IndependentMarginalsVariable = None,
        bkd: BackendMixin = NumpyMixin,
        unbounded_eps: float = 0,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self.set_start_idx(start_idx)
        if variable is not None:
            if not isinstance(variable, IndependentMarginalsVariable):
                raise ValueError(
                    "variable must be an instance of "
                    "IndependentMarginalsVariable"
                )
            if variable.nvars() != nvars:
                raise ValueError("nvars and variable are inconsistent")
        self._variable = variable
        self._eps = unbounded_eps

    def set_start_idx(self, idx: int):
        self._start_idx = idx

    def __repr__(self) -> str:
        return "{0}(startd_idx={1})".format(
            self.__class__.__name__, self._start_idx
        )

    @abstractmethod
    def _canonical_samples(self, nsamples: int) -> Array:
        # return samples in [0, 1]
        raise NotImplementedError

    def rvs(self, nsamples: int) -> Array:
        can_samples = self._canonical_samples(int(nsamples))
        if self._variable is None:
            return can_samples
        # can samples on [0, 1]
        # map unbounded variables to [eps, 1-eps] to avoid singularities
        for ii, marginal in enumerate(self._variable.marginals()):
            if not marginal.is_bounded():
                can_samples[ii] = (
                    can_samples[ii] * (1 - 2 * self._eps) + self._eps
                )
        return self._variable.ppf(can_samples)

    def nvars(self) -> int:
        return self._nvars


class HaltonSequence(LowDiscrepancySequence):
    def __init__(
        self,
        nvars: int,
        start_idx: int = 0,
        variable: IndependentMarginalsVariable = None,
        bkd: BackendMixin = NumpyMixin,
        unbounded_eps: float = 0,
    ):
        super().__init__(nvars, start_idx, variable, bkd, unbounded_eps)
        self._primes = get_first_n_primes(self._nvars)

    @staticmethod
    @njit(cache=True)
    def _sequence(nvars, index1, index2, primes):
        nsamples = index2 - index1
        sequence = np.zeros((nvars, nsamples))
        ones = np.ones(nvars)

        kk = 0
        for ii in range(index1, index2):
            ff = ii * ones
            prime_inv = 1.0 / primes
            summand = ii * nvars
            while summand > 0:
                remainder = np.remainder(ff, primes)
                sequence[:, kk] += remainder * prime_inv
                prime_inv /= primes
                ff = ff // primes
                summand = ff.sum()
            kk += 1
        return sequence

    def _canonical_samples(self, nsamples: int) -> Array:
        samples = self._sequence(
            self._nvars,
            self._start_idx,
            self._start_idx + nsamples,
            self._primes,
        )
        return self._bkd.asarray(samples)


class SobolSequence(LowDiscrepancySequence):
    """
    Compute Sobol sequence using
    Algorithm 659: Implementing Sobol’s quasirandom sequence generator
    with direction_numbers obtain from
    https://web.maths.unsw.edu.au/~fkuo/sobol/

    See Section 1 of notes at
    https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf
    """

    def _index_of_first_zero_bit_moving_right_to_left(self, ii):
        """
        Return the first the index of the first zero bit of the integer ii
        when moving right to left. Returns index in [0,...,nbits(ii)-1]

        # x >> y Returns x with the bits shifted to the right by y places.
        # This is the same as //'ing x by 2**y.

        # x & y Return bitwise and
        """

        jj = 1
        while (ii & 1) > 0:
            ii >>= 1
            jj += 1
        return jj - 1

    def _load_direction_sequence(self):
        """
        Read direction_numbers from
        https://web.maths.unsw.edu.au/~fkuo/sobol/
        """
        import os

        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_seq_file = open(
            os.path.join(dir_path, "sobol_direction_sequence.txt"), "r"
        )
        a_vals = np.empty(self._nvars - 1, dtype=np.int64)
        dir_seq = []
        ii = 1  # file does not store values for 0th dimension
        line = dir_seq_file.readline()  # skip header
        while ii < self._nvars:
            line = dir_seq_file.readline()
            if not line:
                msg = "Requested to many dimension."
                raise Exception(msg)
            line = line.split()
            dim, s, a_vals[ii - 1] = line[:3]
            dir_seq.append(np.asarray(line[3:], dtype=np.int64))
            assert int(dim) == ii + 1
            assert len(dir_seq[-1]) == int(s)
            ii += 1
        dir_seq_file.close()
        return a_vals, dir_seq

    def _compute_direction_numbers(self, seq, max_nbits, power, a_val):
        """Scale the direction sequence by 2**32

        x << y Returns x with the bits shifted to the left by y places.
        This is the same as multiplying x by 2**y.

        x ^ y  Returns a "bitwise exclusive or"
        """
        if seq is None:
            seq = np.ones(max_nbits, dtype=np.int64)

        dir_nums = np.zeros((max_nbits), dtype=np.int64)
        size = len(seq)

        if max_nbits < size:
            raise RuntimeError(f"{max_nbits}<{size}")
        # TODO change loop not to be over min(max_nbits, size) to just be to
        # max_nbits

        for ll in range(min(max_nbits, size)):
            dir_nums[ll] = seq[ll] << (power - (ll + 1))

        for ll in range(min(max_nbits, size), max_nbits):
            dir_nums[ll] = dir_nums[ll - size] ^ (dir_nums[ll - size] >> size)
            for ii in range(size - 1):
                dir_nums[ll] ^= ((a_val >> (size - ii - 2)) & 1) * dir_nums[
                    ll - ii - 1
                ]
        return dir_nums

    def _canonical_samples(self, nsamples: int) -> Array:
        # this function creates all samples up to nsamples + start_idx
        # then removes first start_idx samples
        nsamples = self._start_idx + nsamples
        power = 32
        max_nbits = np.int64(np.ceil(np.log2(nsamples)))
        if max_nbits > power:
            msg = "Requested to many samples. "
            msg += f"Can only compute {max_nbits} samples."
            raise Exception(msg)

        a_vals, dir_seq = self._load_direction_sequence()

        # compute the first right zero bit of each sample index
        indices = np.array(
            [
                self._index_of_first_zero_bit_moving_right_to_left(ii)
                for ii in range(nsamples)
            ]
        )

        const = np.double(1 << power)  # 2**power
        samples = np.empty((self._nvars, nsamples))
        samples[:, 0] = 0

        tmp1 = 0
        # seq = np.ones((max_nbits), dtype=np.int64)
        dir_nums = self._compute_direction_numbers(
            None, max_nbits, power, None
        )
        for ii in range(1, nsamples):
            tmp2 = tmp1 ^ dir_nums[indices[ii - 1]]
            samples[0, ii] = tmp2 / const
            tmp1 = tmp2
        for dd in range(1, self._nvars):
            dir_nums = self._compute_direction_numbers(
                dir_seq[dd - 1], max_nbits, power, a_vals[dd - 1]
            )
            tmp1 = 0
            for ii in range(1, nsamples):
                tmp2 = tmp1 ^ dir_nums[indices[ii - 1]]
                samples[dd, ii] = tmp2 / const
                tmp1 = tmp2
        assert samples.max() <= 1 and samples.min() >= 0
        # assume we do not need to differentiate through sequence creation
        return self._bkd.asarray(samples[:, self._start_idx :])
