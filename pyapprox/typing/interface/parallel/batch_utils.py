"""Utilities for batch splitting and combining in parallel execution.

This module provides BatchSplitter for splitting sample batches into
chunks for parallel processing and combining the results.
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend


class BatchSplitter(Generic[Array]):
    """Split and combine array batches for parallel processing.

    Handles splitting input samples into chunks for parallel evaluation
    and combining the results back into proper array shapes.

    Parameters
    ----------
    bkd : Backend[Array]
        The array backend (NumPy or PyTorch).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> splitter = BatchSplitter(bkd)
    >>> samples = bkd.randn(3, 100)  # 100 samples, 3 vars
    >>> chunks = splitter.split_samples(samples, n_chunks=4)
    >>> # Process chunks in parallel...
    >>> # results = [process(chunk) for chunk in chunks]
    >>> # combined = splitter.combine_outputs(results)
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        return self._bkd

    def split_samples(
        self,
        samples: Array,
        n_chunks: int,
    ) -> List[Array]:
        """Split samples into chunks for parallel processing.

        Parameters
        ----------
        samples : Array
            Samples with shape (nvars, nsamples).
        n_chunks : int
            Number of chunks to split into.

        Returns
        -------
        List[Array]
            List of sample arrays, each with shape (nvars, chunk_size).
        """
        nsamples = samples.shape[1]
        if n_chunks <= 0:
            raise ValueError("n_chunks must be positive")
        if n_chunks > nsamples:
            n_chunks = nsamples

        chunk_size = (nsamples + n_chunks - 1) // n_chunks
        chunks = []
        for i in range(0, nsamples, chunk_size):
            end = min(i + chunk_size, nsamples)
            chunks.append(samples[:, i:end])
        return chunks

    def split_to_singles(self, samples: Array) -> List[Array]:
        """Split samples into individual column vectors.

        Each sample becomes a (nvars, 1) array for single-sample processing.

        Parameters
        ----------
        samples : Array
            Samples with shape (nvars, nsamples).

        Returns
        -------
        List[Array]
            List of single-sample arrays, each (nvars, 1).
        """
        nsamples = samples.shape[1]
        return [samples[:, i : i + 1] for i in range(nsamples)]

    def combine_outputs(
        self,
        outputs: List[Array],
        axis: int = 1,
    ) -> Array:
        """Combine outputs from parallel processing.

        Parameters
        ----------
        outputs : List[Array]
            List of output arrays to concatenate.
        axis : int, optional
            Axis along which to concatenate. Default is 1 (samples axis).

        Returns
        -------
        Array
            Combined output array.
        """
        if not outputs:
            raise ValueError("outputs list cannot be empty")
        return self._bkd.concatenate(outputs, axis=axis)

    def combine_jacobians(self, jacobians: List[Array]) -> Array:
        """Combine single-sample jacobians into batch format.

        Takes jacobians from single-sample evaluations (nqoi, nvars) each
        and stacks them into batch format (nsamples, nqoi, nvars).

        Parameters
        ----------
        jacobians : List[Array]
            List of jacobian arrays, each (nqoi, nvars).

        Returns
        -------
        Array
            Stacked jacobians with shape (nsamples, nqoi, nvars).
        """
        if not jacobians:
            raise ValueError("jacobians list cannot be empty")
        # Stack along new first axis: (nqoi, nvars) -> (nsamples, nqoi, nvars)
        return self._bkd.stack(jacobians, axis=0)

    def combine_hessians(self, hessians: List[Array]) -> Array:
        """Combine single-sample hessians into batch format.

        Takes hessians from single-sample evaluations (nvars, nvars) each
        and stacks them into batch format (nsamples, nvars, nvars).

        Parameters
        ----------
        hessians : List[Array]
            List of hessian arrays, each (nvars, nvars).

        Returns
        -------
        Array
            Stacked hessians with shape (nsamples, nvars, nvars).
        """
        if not hessians:
            raise ValueError("hessians list cannot be empty")
        return self._bkd.stack(hessians, axis=0)

    def combine_hvps(self, hvps: List[Array]) -> Array:
        """Combine single-sample HVP results into batch format.

        Takes HVP results from single-sample evaluations (nvars, 1) each
        and combines them into batch format (nsamples, nvars).

        Parameters
        ----------
        hvps : List[Array]
            List of HVP result arrays, each (nvars, 1).

        Returns
        -------
        Array
            Combined HVPs with shape (nsamples, nvars).
        """
        if not hvps:
            raise ValueError("hvps list cannot be empty")
        # Each hvp is (nvars, 1), reshape to (nvars,) and stack
        nvars = hvps[0].shape[0]
        reshaped = [self._bkd.reshape(h, (nvars,)) for h in hvps]
        return self._bkd.stack(reshaped, axis=0)
