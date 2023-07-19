import numpy as np

from pyapprox.util.pya_numba import njit
from pyapprox.util.utilities import get_first_n_primes
from pyapprox.util.sys_utilities import trace_error_with_msg


@njit(cache=True)
def __halton_sequence(num_vars, index1, index2, primes):
    num_samples = index2-index1
    sequence = np.zeros((num_vars, num_samples))
    ones = np.ones(num_vars)

    kk = 0
    for ii in range(index1, index2):
        ff = ii*ones
        prime_inv = 1./primes
        summand = ii*num_vars
        while summand > 0:
            remainder = np.remainder(ff, primes)
            sequence[:, kk] += remainder*prime_inv
            prime_inv /= primes
            ff = ff//primes
            summand = ff.sum()
        kk += 1
    return sequence


def transformed_halton_sequence(marginal_icdfs, num_vars, num_samples,
                                start_index=1):
    """
    Generate a Halton sequence using inverse transform sampling.

    Deprecated: pass a variable argument to
   :func:`pyapprox.expdesign.low_discrepancy_sequences.halton_sequence`
    """
    assert start_index > 0
    # sample with index 0 is [0,..0] this can cause problems for icdfs of
    # unbounded random variables so start with index 1 in halton sequence
    samples = halton_sequence(num_vars, num_samples, start_index)
    if marginal_icdfs is None:
        return samples

    if callable(marginal_icdfs):
        marginal_icdfs = [marginal_icdfs]*num_vars
    else:
        assert len(marginal_icdfs) == num_vars

    for ii in range(num_vars):
        samples[ii, :] = marginal_icdfs[ii](samples[ii, :])
    return samples


def __index_of_first_zero_bit_moving_right_to_left(ii):
    """
    Return the first the index of the first zero bit of the integer ii
    when moving right to left. Returns index in [0,...,nbits(ii)-1]

    # x >> y Returns x with the bits shifted to the right by y places.
    # This is the same as //'ing x by 2**y.

    # x & y Return bitwise and
    """

    jj = 1
    while ((ii & 1) > 0):
        ii >>= 1
        jj += 1
        # print(ii, "{0:b}".format(ii), jj, (ii&1)>0)
    return jj-1


def __load_direction_sequence(nvars):
    """
    Read direction_numbers from
    https://web.maths.unsw.edu.au/~fkuo/sobol/
    """
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_seq_file = open(
        os.path.join(dir_path, 'sobol_direction_sequence.txt'), 'r')
    a_vals = np.empty(nvars-1, dtype=np.int64)
    dir_seq = []
    ii = 1  # file does not store values for 0th dimension
    line = dir_seq_file.readline()  # skip header
    while ii < nvars:
        line = dir_seq_file.readline()
        if not line:
            msg = 'Requested to many dimension.'
            raise Exception(msg)
        line = line.split()
        dim, s, a_vals[ii-1] = line[:3]
        dir_seq.append(np.asarray(line[3:], dtype=np.int64))
        assert int(dim) == ii+1
        assert len(dir_seq[-1]) == int(s)
        ii += 1
    dir_seq_file.close()
    return a_vals, dir_seq


def __compute_direction_numbers(seq, max_nbits, power, a_val):
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
        dir_nums[ll] = seq[ll] << (power-(ll+1))

    for ll in range(min(max_nbits, size), max_nbits):
        dir_nums[ll] = dir_nums[ll-size] ^ (dir_nums[ll-size] >> size)
        for ii in range(size-1):
            dir_nums[ll] ^= ((a_val >> (size-ii-2)) & 1)*dir_nums[ll-ii-1]
    return dir_nums


def _sobol_sequence(nvars, nsamples):
    """
    Compute Sobol sequence using
    Algorithm 659: Implementing Sobol’s quasirandom sequence generator
    with direction_numbers obtain from
    https://web.maths.unsw.edu.au/~fkuo/sobol/

    See Section 1 of notes at
    https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf
    """
    power = 32
    max_nbits = np.int64(np.ceil(np.log2(nsamples)))
    if max_nbits > power:
        msg = 'Requested to many samples. '
        msg += f'Can only compute {max_nbits} samples.'
        raise Exception(msg)

    a_vals, dir_seq = __load_direction_sequence(nvars)

    # compute the first right zero bit of each sample index
    indices = np.array(
        [__index_of_first_zero_bit_moving_right_to_left(ii)
         for ii in range(nsamples)])

    const = np.double(1 << power)  # 2**power
    samples = np.empty((nvars, nsamples))
    samples[:, 0] = 0

    tmp1 = 0
    # seq = np.ones((max_nbits), dtype=np.int64)
    dir_nums = __compute_direction_numbers(None, max_nbits, power, None)
    for ii in range(1, nsamples):
        tmp2 = tmp1 ^ dir_nums[indices[ii-1]]
        samples[0, ii] = tmp2/const
        tmp1 = tmp2
    for dd in range(1, nvars):
        dir_nums = __compute_direction_numbers(
            dir_seq[dd-1], max_nbits, power, a_vals[dd-1])
        tmp1 = 0
        for ii in range(1, nsamples):
            tmp2 = tmp1 ^ dir_nums[indices[ii-1]]
            samples[dd, ii] = tmp2/const
            tmp1 = tmp2
    assert samples.max() <= 1 and samples.min() >= 0
    return samples


def sobol_sequence(nvars, nsamples, start_index=0, variable=None):
    """
    Generate a multivariate Sobol sequence

    Parameters
    ----------
    nvars : integer
        The number of dimensions

    nsamples : integer
        The number of samples needed

    start_index : integer
        The number of initial samples in the Sobol sequence to skip

    variable : :class:pyapprox.variabels.IndependentMarginalsVariable
        If provided will be used for inverse transform sampling

    Returns
    -------
    samples : np.ndarray (nvars, nsamples)
        The low-discrepancy samples
    """
    nsamples = int(nsamples)
    samples = _sobol_sequence(nvars, nsamples+start_index)[:, start_index:]
    if variable is None:
        return samples
    assert variable.num_vars() == nvars
    samples = variable.evaluate('ppf', samples)
    return samples


def halton_sequence(num_vars, nsamples, start_index=0, variable=None):
    """
    Generate a multivariate Halton sequence

        Parameters
    ----------
    nvars : integer
        The number of dimensions

    nsamples : integer
        The number of samples needed

    start_index : integer
        The number of initial samples in the Sobol sequence to skip

    variable : :class:pyapprox.variabels.IndependentMarginalsVariable
        If provided will be used for inverse transform sampling

    Returns
    -------
    samples : np.ndarray (nvars, nsamples)
        The low-discrepancy samples
    """
    index1, index2 = start_index, start_index + nsamples
    assert index1 < index2, "Index 1 must be < Index 2"
    assert num_vars <= 100, "Number of variables must be <= 100"

    primes = get_first_n_primes(num_vars)

    try:
        from pyapprox.cython.utilities import halton_sequence_pyx
        samples = halton_sequence_pyx(primes, index1, index2)
    except Exception as e:
        trace_error_with_msg('halton_sequence extension failed', e)
        samples = __halton_sequence(num_vars, index1, index2, primes)

    if variable is None:
        return samples
    return variable.evaluate('ppf', samples)
