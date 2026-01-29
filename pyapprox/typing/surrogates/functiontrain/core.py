"""FunctionTrain core - a single tensor in the train decomposition."""

from typing import Generic, List, Self, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import BasisExpansionProtocol


class FunctionTrainCore(Generic[Array]):
    """A single core of a FunctionTrain.

    A core is a 2D grid of basis expansions indexed by (left_rank, right_rank).
    Each basis expansion maps a single variable to nqoi outputs.

    The core tensor has shape (r_left, r_right) where each element is a
    univariate basis expansion.

    Parameters
    ----------
    basisexps : List[List[BasisExpansionProtocol[Array]]]
        2D list of basis expansions. Shape: [r_left][r_right]
        Each BasisExpansion should be univariate (nvars=1).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        basisexps: List[List[BasisExpansionProtocol[Array]]],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._basisexps = basisexps
        self._r_left = len(basisexps)
        self._r_right = len(basisexps[0]) if basisexps else 0

        # Validate structure
        for row in basisexps:
            if len(row) != self._r_right:
                raise ValueError("All rows must have same number of elements")

        # Store shapes for parameter flattening
        self._param_shapes: List[Tuple[int, int]] = []
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                bexp = self._basisexps[ii][jj]
                self._param_shapes.append((bexp.nterms(), bexp.nqoi()))

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def ranks(self) -> Tuple[int, int]:
        """Return (left_rank, right_rank)."""
        return (self._r_left, self._r_right)

    def nparams(self) -> int:
        """Return total number of parameters in this core."""
        total = 0
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                total += self._basisexps[ii][jj].nparams()
        return total

    def nqoi(self) -> int:
        """Return number of QoIs (from first basis expansion)."""
        return self._basisexps[0][0].nqoi()

    def __call__(self, sample_1d: Array) -> Array:
        """Evaluate core at univariate samples.

        Parameters
        ----------
        sample_1d : Array
            Univariate samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Core values. Shape: (r_left, r_right, nsamples, nqoi)
        """
        nsamples = sample_1d.shape[1]
        nqoi = self.nqoi()

        values = []
        for ii in range(self._r_left):
            row_values = []
            for jj in range(self._r_right):
                # BasisExpansion.__call__ returns (nqoi, nsamples)
                # We want (nsamples, nqoi) for einsum compatibility
                val = self._basisexps[ii][jj](sample_1d).T  # (nsamples, nqoi)
                row_values.append(val)
            # Stack along axis 0: (r_right, nsamples, nqoi)
            values.append(self._bkd.stack(row_values, axis=0))
        # Stack along axis 0: (r_left, r_right, nsamples, nqoi)
        return self._bkd.stack(values, axis=0)

    def basis_matrix(self, sample_1d: Array, ii: int, jj: int) -> Array:
        """Get basis matrix for a specific basis expansion.

        Parameters
        ----------
        sample_1d : Array
            Univariate samples. Shape: (1, nsamples)
        ii : int
            Left rank index.
        jj : int
            Right rank index.

        Returns
        -------
        Array
            Basis matrix. Shape: (nsamples, nterms)
        """
        return self._basisexps[ii][jj].basis_matrix(sample_1d)

    def get_nparams(self, ii: int, jj: int) -> int:
        """Get number of trainable parameters for a specific expansion.

        Parameters
        ----------
        ii : int
            Left rank index.
        jj : int
            Right rank index.

        Returns
        -------
        int
            Number of trainable parameters for this expansion.
        """
        return self._basisexps[ii][jj].nparams()

    def get_nterms(self, ii: int, jj: int) -> int:
        """Get number of basis terms for a specific expansion.

        Parameters
        ----------
        ii : int
            Left rank index.
        jj : int
            Right rank index.

        Returns
        -------
        int
            Number of basis terms for this expansion.
        """
        return self._basisexps[ii][jj].nterms()

    def total_nterms(self) -> int:
        """Return total number of basis terms across all expansions.

        This is the number of columns in the full Jacobian.
        """
        total = 0
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                total += self._basisexps[ii][jj].nterms()
        return total

    def get_trainable_indices(self) -> List[int]:
        """Get column indices in full Jacobian that correspond to trainable params.

        Returns
        -------
        List[int]
            Indices of trainable columns in the full Jacobian.
        """
        indices = []
        col = 0
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                bexp = self._basisexps[ii][jj]
                nterms = bexp.nterms()
                if bexp.nparams() > 0:
                    # This expansion is trainable
                    indices.extend(range(col, col + nterms))
                col += nterms
        return indices

    def with_params(self, params: Array) -> Self:
        """Return NEW core with given parameters.

        Parameters
        ----------
        params : Array
            Flattened parameters. Shape: (nparams,) or (nparams, 1)

        Returns
        -------
        Self
            New FunctionTrainCore with parameters set.
        """
        params_flat = self._bkd.flatten(params)
        if params_flat.shape[0] != self.nparams():
            raise ValueError(
                f"Expected {self.nparams()} params, got {params_flat.shape[0]}"
            )

        # Distribute params to basis expansions
        new_basisexps: List[List[BasisExpansionProtocol[Array]]] = []
        idx = 0
        for ii in range(self._r_left):
            row: List[BasisExpansionProtocol[Array]] = []
            for jj in range(self._r_right):
                old_bexp = self._basisexps[ii][jj]
                nparams_bexp = old_bexp.nparams()
                if nparams_bexp == 0:
                    # Fixed expansion (e.g., ConstantExpansion) - just copy
                    new_bexp = old_bexp.with_params(old_bexp.get_coefficients())
                    row.append(new_bexp)
                else:
                    bexp_params = params_flat[idx : idx + nparams_bexp]
                    # Reshape to (nterms, nqoi)
                    bexp_params_2d = self._bkd.reshape(
                        bexp_params, (old_bexp.nterms(), old_bexp.nqoi())
                    )
                    new_bexp = old_bexp.with_params(bexp_params_2d)
                    row.append(new_bexp)
                    idx += nparams_bexp
            new_basisexps.append(row)

        return self.__class__(new_basisexps, self._bkd)

    def _flatten_params(self) -> Array:
        """Flatten all trainable parameters to single vector.

        Only includes parameters from expansions with nparams > 0.

        Returns
        -------
        Array
            Flattened parameters. Shape: (nparams,)
        """
        params_list = []
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                bexp = self._basisexps[ii][jj]
                if bexp.nparams() > 0:
                    coef = bexp.get_coefficients()
                    params_list.append(self._bkd.flatten(coef))
        if len(params_list) == 0:
            return self._bkd.zeros((0,))
        return self._bkd.hstack(params_list)

    def __repr__(self) -> str:
        return f"FunctionTrainCore(ranks={self.ranks()})"
