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

        # Pre-allocate result tensor to avoid list appending and stacking
        result = self._bkd.zeros((self._r_left, self._r_right, nsamples, nqoi))
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                # BasisExpansion.__call__ returns (nqoi, nsamples)
                # We want (nsamples, nqoi) for einsum compatibility
                result[ii, jj] = self._basisexps[ii][jj](sample_1d).T
        return result

    def get_basisexp(self, ii: int, jj: int) -> BasisExpansionProtocol[Array]:
        """Get univariate expansion at position (ii, jj).

        Parameters
        ----------
        ii : int
            Left rank index.
        jj : int
            Right rank index.

        Returns
        -------
        BasisExpansionProtocol[Array]
            The univariate basis expansion.

        Raises
        ------
        IndexError
            If indices are out of bounds.
        """
        if ii < 0 or ii >= self._r_left:
            raise IndexError(
                f"Left rank index {ii} out of bounds [0, {self._r_left})"
            )
        if jj < 0 or jj >= self._r_right:
            raise IndexError(
                f"Right rank index {jj} out of bounds [0, {self._r_right})"
            )
        return self._basisexps[ii][jj]

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

    def jacobian_wrt_params(self, sample_1d: Array) -> Array:
        """Jacobian of core output w.r.t. trainable parameters.

        This is used for gradient-based optimization (MSEFitter).

        Parameters
        ----------
        sample_1d : Array
            Univariate samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (r_left, r_right, nsamples, nqoi, nparams_this_core)

        Notes
        -----
        The Jacobian is sparse: d(core[i,j])/d(theta_k) is zero unless theta_k
        belongs to the expansion at position (i,j).
        """
        nsamples = sample_1d.shape[1]
        nparams = self.nparams()
        nqoi = self.nqoi()

        if nparams == 0:
            return self._bkd.zeros(
                (self._r_left, self._r_right, nsamples, nqoi, 0)
            )

        # Build Jacobian using list accumulation then stack
        # We need shape (r_left, r_right, nsamples, nqoi, nparams)
        jac_rows = []
        for ii in range(self._r_left):
            jac_cols = []
            for jj in range(self._r_right):
                # Each position (ii, jj) contributes to specific param indices
                # Initialize a (nsamples, nqoi, nparams) array of zeros
                # and fill in the part for this expansion
                pos_jac = self._bkd.zeros((nsamples, nqoi, nparams))

                bexp = self._basisexps[ii][jj]
                bexp_nparams = bexp.nparams()

                if bexp_nparams > 0:
                    # Get jacobian for this expansion
                    if hasattr(bexp, "jacobian_wrt_params"):
                        bexp_jac = bexp.jacobian_wrt_params(sample_1d)
                    else:
                        bexp_jac = self._linear_jacobian_wrt_params(
                            bexp, sample_1d
                        )
                    # bexp_jac has shape (nsamples, nqoi, bexp_nparams)

                    # Find starting index for this expansion's params
                    param_idx = self._get_param_start_index(ii, jj)

                    # Build the full jacobian for this position by concatenation
                    # Left padding: zeros of shape (nsamples, nqoi, param_idx)
                    # Middle: bexp_jac of shape (nsamples, nqoi, bexp_nparams)
                    # Right padding: zeros of shape (nsamples, nqoi, remaining)
                    remaining = nparams - param_idx - bexp_nparams

                    parts = []
                    if param_idx > 0:
                        parts.append(
                            self._bkd.zeros((nsamples, nqoi, param_idx))
                        )
                    parts.append(bexp_jac)
                    if remaining > 0:
                        parts.append(
                            self._bkd.zeros((nsamples, nqoi, remaining))
                        )

                    pos_jac = self._bkd.concatenate(parts, axis=2)

                jac_cols.append(pos_jac)

            # Stack along axis 0 to get (r_right, nsamples, nqoi, nparams)
            jac_rows.append(self._bkd.stack(jac_cols, axis=0))

        # Stack along axis 0 to get (r_left, r_right, nsamples, nqoi, nparams)
        return self._bkd.stack(jac_rows, axis=0)

    def _get_param_start_index(self, target_ii: int, target_jj: int) -> int:
        """Get the starting parameter index for expansion at (target_ii, target_jj).

        Parameters
        ----------
        target_ii : int
            Left rank index.
        target_jj : int
            Right rank index.

        Returns
        -------
        int
            Starting index in the flattened parameter vector.
        """
        param_idx = 0
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                if ii == target_ii and jj == target_jj:
                    return param_idx
                bexp = self._basisexps[ii][jj]
                param_idx += bexp.nparams()
        return param_idx

    def _linear_jacobian_wrt_params(
        self, bexp: BasisExpansionProtocol[Array], sample_1d: Array
    ) -> Array:
        """Compute Jacobian for linear parameterization from basis matrix.

        For f(x) = sum_l theta_{l,q} * phi_l(x), we have:
            df_q/d(theta_{l,q'}) = phi_l(x) if q == q' else 0

        Parameters
        ----------
        bexp : BasisExpansionProtocol
            The basis expansion.
        sample_1d : Array
            Univariate samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, nqoi, nparams)
        """
        phi = bexp.basis_matrix(sample_1d)  # (nsamples, nterms)
        nsamples = phi.shape[0]
        nterms = bexp.nterms()
        nqoi = bexp.nqoi()
        nparams = bexp.nparams()

        # For linear, nparams = nterms * nqoi (row-major: all terms for q=0, then q=1, etc.)
        # jac[s, q, l + q*nterms] = phi[s, l]
        jac_parts = []
        for qq in range(nqoi):
            # For QoI qq, the jacobian w.r.t. its nterms params is phi
            # But we need to place it in the right position in the nparams dimension
            # Params for qq are at indices [qq*nterms : (qq+1)*nterms]
            # Before: zeros of shape (nsamples, 1, qq*nterms)
            # Middle: phi reshaped to (nsamples, 1, nterms)
            # After: zeros of shape (nsamples, 1, (nqoi-qq-1)*nterms)
            before = self._bkd.zeros((nsamples, 1, qq * nterms))
            middle = self._bkd.reshape(phi, (nsamples, 1, nterms))
            after = self._bkd.zeros((nsamples, 1, (nqoi - qq - 1) * nterms))

            row_parts = []
            if qq > 0:
                row_parts.append(before)
            row_parts.append(middle)
            if qq < nqoi - 1:
                row_parts.append(after)

            if len(row_parts) == 1:
                qoi_jac = row_parts[0]
            else:
                qoi_jac = self._bkd.concatenate(row_parts, axis=2)
            jac_parts.append(qoi_jac)

        # Stack along axis 1 to get (nsamples, nqoi, nparams)
        return self._bkd.concatenate(jac_parts, axis=1)

    def supports_input_jacobian(self) -> bool:
        """Check if all basis expansions support input Jacobian.

        Returns
        -------
        bool
            True if all expansions have jacobian_batch method.
        """
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                if not hasattr(self._basisexps[ii][jj], "jacobian_batch"):
                    return False
        return True

    def jacobian_wrt_input(self, sample_1d: Array) -> Array:
        """Jacobian of core output w.r.t. univariate input.

        This computes the derivative of each element in the core matrix
        with respect to the univariate input x_k.

        Parameters
        ----------
        sample_1d : Array
            Univariate samples. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (r_left, r_right, nsamples, nqoi)
            Same shape as __call__ output since each core is univariate.

        Raises
        ------
        RuntimeError
            If any basis expansion does not support jacobian_batch.
        """
        nsamples = sample_1d.shape[1]
        nqoi = self.nqoi()

        # Pre-allocate result tensor to avoid list appending and stacking
        result = self._bkd.zeros((self._r_left, self._r_right, nsamples, nqoi))
        for ii in range(self._r_left):
            for jj in range(self._r_right):
                bexp = self._basisexps[ii][jj]
                if not hasattr(bexp, "jacobian_batch"):
                    raise RuntimeError(
                        f"Basis expansion at ({ii}, {jj}) does not support "
                        "jacobian_batch. Input Jacobian requires differentiable "
                        "bases."
                    )
                # jacobian_batch returns (nsamples, nqoi, nvars=1)
                jac = bexp.jacobian_batch(sample_1d)
                # Extract the single variable dimension: (nsamples, nqoi)
                result[ii, jj] = jac[:, :, 0]
        return result

    def __repr__(self) -> str:
        return f"FunctionTrainCore(ranks={self.ranks()})"
