from abc import ABC, abstractmethod

import numpy as np

from pyapprox.util.pya_numba import njit
from pyapprox.util.misc import get_first_n_primes
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    IndependentGroupsVariable,
)
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


class LowDiscrepancySequence(ABC):
    """
    Abstract base class for generating low-discrepancy sequences.

    This class defines an interface for generating low-discrepancy sequences, which
    are deterministic sequences designed to uniformly sample the unit hypercube.
    Derived classes must implement the `_canonical_samples` method to generate
    specific types of low-discrepancy sequences.

    Parameters
    ----------
    nvars : int
        Number of variables (dimensions) in the sequence.
    start_idx : int, optional
        Starting index for the sequence. Default is 0.
    variable : IndependentMarginalsVariable, optional
        Variable defining the marginal distributions for transforming the sequence.
        Must be an instance of `IndependentMarginalsVariable` or `IndependentGroupsVariable`.
        Default is None.
    bkd : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    unbounded_eps : float, optional
        Small value to avoid singularities when mapping unbounded variables. Default is 0.
    increment_start_index : bool, optional
        Whether to increment the starting index after each call to `rvs`. Default is False.

    """

    def __init__(
        self,
        nvars: int,
        start_idx: int = 0,
        variable: IndependentMarginalsVariable = None,
        bkd: BackendMixin = NumpyMixin,
        unbounded_eps: float = 0,
        increment_start_index: bool = False,
    ):
        """
        Initialize the low-discrepancy sequence generator.

        Parameters
        ----------
        nvars : int
            Number of variables (dimensions) in the sequence.
        start_idx : int, optional
            Starting index for the sequence. Default is 0.
        variable : IndependentMarginalsVariable, optional
            Variable defining the marginal distributions for transforming the sequence.
            Must be an instance of `IndependentMarginalsVariable` or `IndependentGroupsVariable`.
            Default is None.
        bkd : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        unbounded_eps : float, optional
            Small value to avoid singularities when mapping unbounded variables. Default is 0.
        increment_start_index : bool, optional
            Whether to increment the starting index after each call to `rvs`. Default is False.

        Raises
        ------
        ValueError
            If `variable` is not an instance of `IndependentMarginalsVariable` or `IndependentGroupsVariable`.
        ValueError
            If `nvars` and the number of variables in `variable` are inconsistent.
        """
        self._bkd = bkd
        self._nvars = nvars
        self.set_start_idx(start_idx)
        self._increment_start_index = increment_start_index
        if variable is not None:
            if not isinstance(
                variable,
                (IndependentMarginalsVariable, IndependentGroupsVariable),
            ):
                raise ValueError(
                    "variable must be an instance of "
                    "IndependentMarginalsVariable or IndependentGroupsVariable"
                )
            if variable.nvars() != nvars:
                raise ValueError("nvars and variable are inconsistent")

        if isinstance(variable, IndependentGroupsVariable):
            marginals = variable.marginals()
            variable = IndependentMarginalsVariable(
                marginals, backend=variable._bkd
            )

        if not self._bkd.bkd_equal(self._bkd, variable._bkd):
            raise ValueError(
                "variable backend {0} and backend {1} are "
                "inconsistent".format(
                    variable._bkd.__name__, self._bkd.__name__
                )
            )

        self._variable = variable
        self._eps = unbounded_eps

    def set_start_idx(self, idx: int):
        """
        Set the starting index for the sequence.

        Parameters
        ----------
        idx : int
            Starting index for the sequence.

        Returns
        -------
        None
        """
        self._start_idx = idx

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns
        -------
        repr : str
            String representation of the class, including the starting index.
        """
        return "{0}(start_idx={1})".format(
            self.__class__.__name__, self._start_idx
        )

    @abstractmethod
    def _canonical_samples(self, nsamples: int) -> Array:
        """
        Generate canonical samples in the unit hypercube.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Canonical samples in the unit hypercube.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the low-discrepancy sequence.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Random samples transformed according to the marginal distributions.

        Notes
        -----
        - Canonical samples are generated in the unit hypercube.
        - Unbounded variables are mapped to the interval `[eps, 1 - eps]` to avoid singularities.
        - If `increment_start_index` is True, the starting index is incremented after generating samples.
        """
        can_samples = self._canonical_samples(int(nsamples))
        if self._variable is None:
            return can_samples
        for ii, marginal in enumerate(self._variable.marginals()):
            if not marginal.is_bounded():
                can_samples[ii] = (
                    can_samples[ii] * (1 - 2 * self._eps) + self._eps
                )
        if self._increment_start_index:
            self.set_start_idx(self._start_idx + nsamples)
        return self._variable.ppf(can_samples)

    def nvars(self) -> int:
        """
        Return the number of variables (dimensions) in the sequence.

        Returns
        -------
        nvars : int
            Number of variables in the sequence.
        """
        return self._nvars


class HaltonSequence(LowDiscrepancySequence):
    """
    Halton sequence generator.

    This class implements the Halton sequence, a type of low-discrepancy sequence
    that uses prime numbers to generate samples in the unit hypercube.

    Parameters
    ----------
    nvars : int
        Number of variables (dimensions) in the sequence.
    start_idx : int, optional
        Starting index for the sequence. Default is 0.
    variable : IndependentMarginalsVariable, optional
        Variable defining the marginal distributions for transforming the sequence.
        Default is None.
    bkd : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    unbounded_eps : float, optional
        Small value to avoid singularities when mapping unbounded variables. Default is 0.
    increment_start_index : bool, optional
        Whether to increment the starting index after each call to `rvs`. Default is False.
    """

    def __init__(
        self,
        nvars: int,
        start_idx: int = 0,
        variable: IndependentMarginalsVariable = None,
        bkd: BackendMixin = NumpyMixin,
        unbounded_eps: float = 0,
        increment_start_index: bool = False,
    ):
        """
        Initialize the Halton sequence generator.

        Parameters
        ----------
        nvars : int
            Number of variables (dimensions) in the sequence.
        start_idx : int, optional
            Starting index for the sequence. Default is 0.
        variable : IndependentMarginalsVariable, optional
            Variable defining the marginal distributions for transforming the sequence.
            Default is None.
        bkd : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        unbounded_eps : float, optional
            Small value to avoid singularities when mapping unbounded variables. Default is 0.
        increment_start_index : bool, optional
            Whether to increment the starting index after each call to `rvs`. Default is False.
        """
        super().__init__(
            nvars,
            start_idx,
            variable,
            bkd,
            unbounded_eps,
            increment_start_index,
        )
        self._primes = get_first_n_primes(self._nvars)

    @staticmethod
    @njit(cache=True)
    def _sequence(nvars, index1, index2, primes):
        """
        Generate the Halton sequence for the given range of indices.

        Parameters
        ----------
        nvars : int
            Number of variables (dimensions) in the sequence.
        index1 : int
            Starting index for the sequence.
        index2 : int
            Ending index for the sequence.
        primes : Array
            Array of prime numbers used to generate the sequence.

        Returns
        -------
        sequence : Array
            Halton sequence samples in the unit hypercube.
        """
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
        """
        Generate canonical samples using the Halton sequence.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Canonical samples in the unit hypercube.
        """
        samples = self._sequence(
            self._nvars,
            self._start_idx,
            self._start_idx + nsamples,
            self._primes,
        )
        return self._bkd.asarray(samples)


class SobolSequence(LowDiscrepancySequence):
    """
    Sobol sequence generator.

    This class implements the Sobol sequence, a type of low-discrepancy sequence
    using direction numbers. The implementation is based on Algorithm 659 and
    uses direction numbers obtained from external sources.

    Notes
    -----
    - Direction numbers are obtained from:
      https://web.maths.unsw.edu.au/~fkuo/sobol/
    - See Section 1 of notes at:
      https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf
    """

    def _index_of_first_zero_bit_moving_right_to_left(self, ii):
        """
        Return the index of the first zero bit of the integer `ii` when moving right to left.

        Parameters
        ----------
        ii : int
            Input integer.

        Returns
        -------
        index : int
            Index of the first zero bit in `[0, ..., nbits(ii) - 1]`.

        Notes
        -----
        - `x >> y` shifts the bits of `x` to the right by `y` places (equivalent to integer division by `2**y`).
        - `x & y` performs a bitwise AND operation.
        """
        jj = 1
        while (ii & 1) > 0:
            ii >>= 1
            jj += 1
        return jj - 1

    def _load_direction_sequence(self):
        """
        Load direction numbers from an external file.

        Returns
        -------
        a_vals : Array
            Array of coefficients `a` for each dimension.
        dir_seq : List[Array]
            List of direction numbers for each dimension.

        Raises
        ------
        Exception
            If the requested number of dimensions exceeds the available direction numbers.

        Notes
        -----
        - Direction numbers are read from `sobol_direction_sequence.txt`.
        - The file must be located in the same directory as this script.
        """
        import os

        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_seq_file = open(
            os.path.join(dir_path, "sobol_direction_sequence.txt"), "r"
        )
        a_vals = np.empty(self._nvars - 1, dtype=np.int64)
        dir_seq = []
        ii = 1  # File does not store values for the 0th dimension
        line = dir_seq_file.readline()  # Skip header
        while ii < self._nvars:
            line = dir_seq_file.readline()
            if not line:
                msg = "Requested too many dimensions."
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
        """
        Compute direction numbers scaled by `2**power`.

        Parameters
        ----------
        seq : Array or None
            Sequence of direction numbers for the current dimension.
        max_nbits : int
            Maximum number of bits for the direction numbers.
        power : int
            Scaling factor (e.g., `2**power`).
        a_val : int or None
            Coefficient `a` for the current dimension.

        Returns
        -------
        dir_nums : Array
            Array of direction numbers scaled by `2**power`.

        Notes
        -----
        - `x << y` shifts the bits of `x` to the left by `y` places (equivalent to multiplying `x` by `2**y`).
        - `x ^ y` performs a bitwise exclusive OR operation.
        """
        if seq is None:
            seq = np.ones(max_nbits, dtype=np.int64)

        dir_nums = np.zeros((max_nbits), dtype=np.int64)
        size = len(seq)

        if max_nbits < size:
            raise RuntimeError(f"{max_nbits} < {size}")

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
        """
        Generate canonical samples using the Sobol sequence.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Canonical samples in the unit hypercube.

        Raises
        ------
        Exception
            If the requested number of samples exceeds the maximum supported by the algorithm.

        Notes
        -----
        - This method generates all samples up to `nsamples + start_idx` and removes the first `start_idx` samples.
        - The Sobol sequence is computed using direction numbers and bitwise operations.
        """
        nsamples = self._start_idx + nsamples
        power = 32
        max_nbits = np.int64(np.ceil(np.log2(nsamples)))
        if max_nbits > power:
            msg = "Requested too many samples. "
            msg += f"Can only compute {max_nbits} samples."
            raise Exception(msg)

        a_vals, dir_seq = self._load_direction_sequence()

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
        return self._bkd.asarray(samples[:, self._start_idx :])
