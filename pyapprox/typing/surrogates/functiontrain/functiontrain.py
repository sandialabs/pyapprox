"""FunctionTrain - tensor train decomposition surrogate."""

from typing import Generic, List, Self

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.functiontrain.core import FunctionTrainCore


class FunctionTrain(Generic[Array]):
    """Function train tensor decomposition surrogate.

    Represents a multivariate function as a sequence of univariate
    basis expansions connected via tensor contractions:

        f(x_1, ..., x_d) = G_1(x_1) · G_2(x_2) · ... · G_d(x_d)

    where each G_i is a matrix of univariate basis expansions.

    Parameters
    ----------
    cores : List[FunctionTrainCore]
        List of cores, one per input variable.
        - cores[0].ranks() must be (1, r_1)
        - cores[-1].ranks() must be (r_{d-1}, 1)
        - cores[i].ranks()[1] == cores[i+1].ranks()[0] for interior cores
    bkd : Backend[Array]
        Computational backend.
    nqoi : int
        Number of quantities of interest. Default: 1.
    """

    def __init__(
        self,
        cores: List[FunctionTrainCore[Array]],
        bkd: Backend[Array],
        nqoi: int = 1,
    ):
        self._cores = cores
        self._bkd = bkd
        self._nqoi = nqoi
        self._validate_ranks()

    def _validate_ranks(self) -> None:
        """Validate that core ranks are compatible."""
        if len(self._cores) == 0:
            raise ValueError("FunctionTrain requires at least one core")

        # First core must have left rank = 1
        if self._cores[0].ranks()[0] != 1:
            raise ValueError(
                f"First rank of first core must be 1, "
                f"got {self._cores[0].ranks()[0]}"
            )

        # Last core must have right rank = 1
        if self._cores[-1].ranks()[1] != 1:
            raise ValueError(
                f"Second rank of last core must be 1, "
                f"got {self._cores[-1].ranks()[1]}"
            )

        # Interior cores must have matching ranks
        for ii in range(len(self._cores) - 1):
            r_right = self._cores[ii].ranks()[1]
            r_left_next = self._cores[ii + 1].ranks()[0]
            if r_right != r_left_next:
                raise ValueError(
                    f"Core {ii} right rank ({r_right}) doesn't match "
                    f"core {ii+1} left rank ({r_left_next})"
                )

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return len(self._cores)

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return self._nqoi

    def nparams(self) -> int:
        """Return total number of parameters across all cores."""
        return sum(core.nparams() for core in self._cores)

    def cores(self) -> List[FunctionTrainCore[Array]]:
        """Return list of cores."""
        return self._cores

    def __call__(self, samples: Array) -> Array:
        """Evaluate FunctionTrain at samples.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Values at samples. Shape: (nqoi, nsamples)
        """
        # Evaluate first core: (1, r_1, nsamples, nqoi)
        values = self._cores[0](samples[:1])

        # Contract with remaining cores
        for ii in range(1, self.nvars()):
            # Core evaluation: (r_{i-1}, r_i, nsamples, nqoi)
            core_val = self._cores[ii](samples[ii : ii + 1])
            # Contraction: "ijkl, jmkl->imkl"
            # i = left rank of accumulated (starts at 1)
            # j = right rank of accumulated = left rank of new core
            # m = right rank of new core
            # k = sample index
            # l = QoI index
            values = self._bkd.einsum("ijkl, jmkl->imkl", values, core_val)

        # Final shape: (1, 1, nsamples, nqoi) -> (nqoi, nsamples)
        return values[0, 0].T

    def with_params(self, params: Array) -> Self:
        """Return NEW FunctionTrain with given parameters.

        Parameters
        ----------
        params : Array
            Flattened parameters. Shape: (nparams,) or (nparams, 1)

        Returns
        -------
        Self
            New FunctionTrain with parameters set.
        """
        params_flat = self._bkd.flatten(params)
        if params_flat.shape[0] != self.nparams():
            raise ValueError(
                f"Expected {self.nparams()} params, got {params_flat.shape[0]}"
            )

        # Distribute params to cores
        new_cores = []
        idx = 0
        for core in self._cores:
            core_nparams = core.nparams()
            core_params = params_flat[idx : idx + core_nparams]
            new_cores.append(core.with_params(core_params))
            idx += core_nparams

        return self.__class__(
            cores=new_cores,
            bkd=self._bkd,
            nqoi=self._nqoi,
        )

    def with_cores(self, cores: List[FunctionTrainCore[Array]]) -> Self:
        """Return NEW FunctionTrain with given cores.

        Parameters
        ----------
        cores : List[FunctionTrainCore]
            New cores to use.

        Returns
        -------
        Self
            New FunctionTrain with given cores.
        """
        return self.__class__(
            cores=cores,
            bkd=self._bkd,
            nqoi=self._nqoi,
        )

    def _flatten_params(self) -> Array:
        """Flatten all core parameters to single vector.

        Returns
        -------
        Array
            Flattened parameters. Shape: (nparams,)
        """
        return self._bkd.hstack([core._flatten_params() for core in self._cores])

    # =========================================================================
    # Per-core Jacobian methods for Alternating Least Squares
    # =========================================================================

    def _eval_left_cores(self, core_id: int, samples: Array) -> Array:
        """Evaluate product of cores [0, core_id).

        Parameters
        ----------
        core_id : int
            Core index (must be > 0).
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Left product. Shape: (1, r_{core_id-1}, nsamples, nqoi)
        """
        if core_id < 1 or core_id >= self.nvars():
            raise ValueError(f"core_id must be in [1, nvars-1], got {core_id}")

        values = self._cores[0](samples[:1])
        for ii in range(1, core_id):
            values = self._bkd.einsum(
                "ijkl, jmkl->imkl",
                values,
                self._cores[ii](samples[ii : ii + 1]),
            )
        return values

    def _eval_right_cores(self, core_id: int, samples: Array) -> Array:
        """Evaluate product of cores [core_id+1, nvars).

        Parameters
        ----------
        core_id : int
            Core index (must be < nvars-1).
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Right product. Shape: (r_{core_id+1}, 1, nsamples, nqoi)
        """
        if core_id >= self.nvars() - 1 or core_id < 0:
            raise ValueError(
                f"core_id must be in [0, nvars-2], got {core_id}"
            )

        values = self._cores[core_id + 1](samples[core_id + 1 : core_id + 2])
        for ii in range(core_id + 2, self.nvars()):
            values = self._bkd.einsum(
                "ijkl, jmkl->imkl",
                values,
                self._cores[ii](samples[ii : ii + 1]),
            )
        return values

    def _core_function_jacobians(
        self, core_id: int, samples: Array
    ) -> List[List[Array]]:
        """Get basis matrices for all basis expansions in a core.

        Parameters
        ----------
        core_id : int
            Core index.
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        List[List[Array]]
            2D list of basis matrices. Shape: [r_left][r_right]
            Each matrix has shape (nsamples, nterms).
        """
        core = self._cores[core_id]
        r_left, r_right = core.ranks()
        sample_1d = samples[core_id : core_id + 1]

        jacs: List[List[Array]] = []
        for ii in range(r_left):
            row: List[Array] = []
            for jj in range(r_right):
                row.append(core.basis_matrix(sample_1d, ii, jj))
            jacs.append(row)
        return jacs

    def _core_jacobian(self, samples: Array, core_id: int) -> List[Array]:
        """Compute Jacobian of output w.r.t. core parameters.

        This is used by ALS to solve local least squares problems.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        core_id : int
            Which core to compute Jacobian for.

        Returns
        -------
        List[Array]
            List of Jacobians, one per QoI.
            Each has shape (nsamples, nparams_this_core).
        """
        if core_id == 0:
            return self._first_core_jacobian(samples)
        if core_id < self.nvars() - 1:
            return self._interior_core_jacobian(samples, core_id)
        return self._final_core_jacobian(samples)

    def _first_core_jacobian(self, samples: Array) -> List[Array]:
        """Jacobian for first core (only right product).

        Shape: f = G_0 · R where R = G_1 · G_2 · ... · G_{d-1}
        df/dG_0[0,j] = R[j,0] * basis_j

        Includes all basis functions (even constants) for correct lstsq solution.
        """
        core = self._cores[0]
        _, r_right = core.ranks()
        fun_jacs = self._core_function_jacobians(0, samples)
        Rmat = self._eval_right_cores(0, samples)  # (r_right, 1, nsamples, nqoi)

        jacs = []
        for qq in range(self.nqoi()):
            jac_parts = []
            for jj in range(r_right):
                # R[jj, 0, :, qq] has shape (nsamples,)
                # fun_jacs[0][jj] has shape (nsamples, nterms)
                jac_parts.append(
                    Rmat[jj, 0, :, qq : qq + 1] * fun_jacs[0][jj]
                )
            jacs.append(self._bkd.hstack(jac_parts))
        return jacs

    def _interior_core_jacobian(
        self, samples: Array, core_id: int
    ) -> List[Array]:
        """Jacobian for interior core (both left and right products).

        Shape: f = L · G_k · R
        df/dG_k[i,j] = L[0,i] * R[j,0] * basis_{i,j}

        Includes all basis functions (even constants) for correct lstsq solution.
        """
        core = self._cores[core_id]
        r_left, r_right = core.ranks()
        Lmat = self._eval_left_cores(core_id, samples)  # (1, r_left, nsamples, nqoi)
        fun_jacs = self._core_function_jacobians(core_id, samples)
        Rmat = self._eval_right_cores(core_id, samples)  # (r_right, 1, nsamples, nqoi)

        jacs = []
        for qq in range(self.nqoi()):
            jac_parts = []
            for ii in range(r_left):
                for jj in range(r_right):
                    # L[0,ii,:,qq] * R[jj,0,:,qq] has shape (nsamples,)
                    # fun_jacs[ii][jj] has shape (nsamples, nterms)
                    weight = (Lmat[0, ii, :, qq] * Rmat[jj, 0, :, qq])[:, None]
                    jac_parts.append(weight * fun_jacs[ii][jj])
            jacs.append(self._bkd.hstack(jac_parts))
        return jacs

    def _final_core_jacobian(self, samples: Array) -> List[Array]:
        """Jacobian for final core (only left product).

        Shape: f = L · G_{d-1} where L = G_0 · G_1 · ... · G_{d-2}
        df/dG_{d-1}[i,0] = L[0,i] * basis_i

        Includes all basis functions (even constants) for correct lstsq solution.
        """
        core = self._cores[self.nvars() - 1]
        r_left, _ = core.ranks()
        Lmat = self._eval_left_cores(
            self.nvars() - 1, samples
        )  # (1, r_left, nsamples, nqoi)
        fun_jacs = self._core_function_jacobians(self.nvars() - 1, samples)

        jacs = []
        for qq in range(self.nqoi()):
            jac_parts = []
            for ii in range(r_left):
                # L[0,ii,:,qq] has shape (nsamples,)
                # fun_jacs[ii][0] has shape (nsamples, nterms)
                jac_parts.append(Lmat[0, ii, :, qq : qq + 1] * fun_jacs[ii][0])
            jacs.append(self._bkd.hstack(jac_parts))
        return jacs

    # =========================================================================
    # Full Jacobian w.r.t. parameters for gradient-based optimization
    # =========================================================================

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """Compute Jacobian of output w.r.t. ALL trainable parameters.

        This is used for gradient-based optimization (MSEFitter).
        Uses the forward-backward sweep algorithm for efficiency.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, nqoi, nparams)

        Notes
        -----
        The gradient computation uses:
            df/d(theta_k) = F_{<k} * dF_k/d(theta) * F_{>k}

        where F_{<k} is the left product and F_{>k} is the right product.
        """
        nsamples = samples.shape[1]
        nparams = self.nparams()
        nqoi = self.nqoi()

        # Edge case: no trainable params
        if nparams == 0:
            return self._bkd.zeros((nsamples, nqoi, 0))

        # Edge case: single variable FT (nvars=1)
        if self.nvars() == 1:
            # Only one core, no left/right products needed
            core_jac = self._cores[0].jacobian_wrt_params(samples[0:1])
            # core_jac shape: (r_left, r_right, nsamples, nqoi, core_nparams)
            # For first/last core: r_left=1, r_right=1
            return core_jac[0, 0]  # (nsamples, nqoi, nparams)

        # Normal case: forward-backward sweep
        # 1. Forward sweep: compute and cache left products L_k = F_0 * ... * F_{k-1}
        left_products: List[Array] = []
        # L_0 doesn't exist (no cores to the left of core 0)
        # L_1 = F_0
        left_products.append(self._cores[0](samples[:1]))
        for kk in range(1, self.nvars() - 1):
            # L_{k+1} = L_k * F_k
            left_products.append(
                self._bkd.einsum(
                    "ijkl, jmkl->imkl",
                    left_products[-1],
                    self._cores[kk](samples[kk : kk + 1]),
                )
            )

        # 2. Backward sweep: compute right products R_k = F_{k+1} * ... * F_{d-1}
        # R_{d-1} doesn't exist (no cores to the right of last core)
        # R_{d-2} = F_{d-1}
        right_products: List[Array] = [
            self._bkd.zeros((1,))
        ] * self.nvars()  # placeholder
        right_products[self.nvars() - 2] = self._cores[self.nvars() - 1](
            samples[self.nvars() - 1 : self.nvars()]
        )
        for kk in range(self.nvars() - 3, -1, -1):
            # R_k = F_{k+1} * R_{k+1}
            right_products[kk] = self._bkd.einsum(
                "ijkl, jmkl->imkl",
                self._cores[kk + 1](samples[kk + 1 : kk + 2]),
                right_products[kk + 1],
            )

        # 3. Assemble Jacobian: contribution from each core
        jac_parts = []
        param_offset = 0

        for kk in range(self.nvars()):
            core = self._cores[kk]
            core_nparams = core.nparams()

            if core_nparams == 0:
                continue

            # Get core Jacobian: (r_left, r_right, nsamples, nqoi, core_nparams)
            core_jac = core.jacobian_wrt_params(samples[kk : kk + 1])
            r_left, r_right = core.ranks()

            # Compute contribution: L_k * dF_k/dtheta * R_k
            # Initialize contribution tensor
            core_contribution = self._bkd.zeros((nsamples, nqoi, core_nparams))

            for ii in range(r_left):
                for jj in range(r_right):
                    # Get weight from left product
                    if kk == 0:
                        # No left product for first core
                        L_weight = self._bkd.ones((nsamples, nqoi))
                    else:
                        # left_products[kk-1] is L_k: (1, r_left, nsamples, nqoi)
                        L_weight = left_products[kk - 1][0, ii, :, :]

                    # Get weight from right product
                    if kk == self.nvars() - 1:
                        # No right product for last core
                        R_weight = self._bkd.ones((nsamples, nqoi))
                    else:
                        # right_products[kk]: (r_right, 1, nsamples, nqoi)
                        R_weight = right_products[kk][jj, 0, :, :]

                    # Combined weight: (nsamples, nqoi)
                    weight = L_weight * R_weight

                    # core_jac[ii, jj] has shape (nsamples, nqoi, core_nparams)
                    # Multiply by weight and accumulate
                    contribution = weight[:, :, None] * core_jac[ii, jj]
                    core_contribution = core_contribution + contribution

            jac_parts.append(core_contribution)
            param_offset += core_nparams

        # Concatenate all core contributions along the parameter dimension
        return self._bkd.concatenate(jac_parts, axis=2)

    # =========================================================================
    # Jacobian w.r.t. input variables
    # =========================================================================

    def jacobian_batch(self, samples: Array) -> Array:
        """Compute Jacobian of output w.r.t. input variables.

        Uses the forward-backward sweep algorithm:
            df/dx_k = L_k * (dG_k/dx_k) * R_k

        where L_k and R_k are left/right products of cores.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nsamples, nqoi, nvars)

        Raises
        ------
        RuntimeError
            If any core does not support input Jacobian computation.
        """
        nsamples = samples.shape[1]
        nvars = self.nvars()
        nqoi = self.nqoi()

        # Check that all cores support input jacobian
        for kk, core in enumerate(self._cores):
            if not core.supports_input_jacobian():
                raise RuntimeError(
                    f"Core {kk} does not support input Jacobian. "
                    "All basis expansions must have jacobian_batch method."
                )

        # Edge case: nvars=1
        if nvars == 1:
            # df/dx_0 = dG_0/dx_0 (no left or right products)
            core_jac = self._cores[0].jacobian_wrt_input(samples[0:1])
            # core_jac shape: (1, 1, nsamples, nqoi)
            # Return shape: (nsamples, nqoi, nvars=1)
            result = core_jac[0, 0]  # (nsamples, nqoi)
            return self._bkd.reshape(result, (nsamples, nqoi, 1))

        # Forward sweep: compute and cache left products L_k = G_0 * ... * G_{k-1}
        left_products: List[Array] = []
        # L_1 = G_0
        left_products.append(self._cores[0](samples[:1]))
        for kk in range(1, nvars - 1):
            # L_{k+1} = L_k * G_k
            left_products.append(
                self._bkd.einsum(
                    "ijkl, jmkl->imkl",
                    left_products[-1],
                    self._cores[kk](samples[kk : kk + 1]),
                )
            )

        # Backward sweep: compute right products R_k = G_{k+1} * ... * G_{d-1}
        # R_{d-1} doesn't exist (no cores to the right of last core)
        # R_{d-2} = G_{d-1}
        right_products: List[Array] = [
            self._bkd.zeros((1,))
        ] * nvars  # placeholder
        right_products[nvars - 2] = self._cores[nvars - 1](
            samples[nvars - 1 : nvars]
        )
        for kk in range(nvars - 3, -1, -1):
            # R_k = G_{k+1} * R_{k+1}
            right_products[kk] = self._bkd.einsum(
                "ijkl, jmkl->imkl",
                self._cores[kk + 1](samples[kk + 1 : kk + 2]),
                right_products[kk + 1],
            )

        # Assemble Jacobian: one column per input variable
        jac_columns = []

        for kk in range(nvars):
            # Get core input jacobian: (r_left, r_right, nsamples, nqoi)
            core_jac = self._cores[kk].jacobian_wrt_input(samples[kk : kk + 1])
            r_left, r_right = self._cores[kk].ranks()

            # Compute contribution: L_k * dG_k/dx_k * R_k
            contribution = self._bkd.zeros((nsamples, nqoi))

            for ii in range(r_left):
                for jj in range(r_right):
                    # Get weight from left product
                    if kk == 0:
                        # No left product for first core
                        L_weight = self._bkd.ones((nsamples, nqoi))
                    else:
                        # left_products[kk-1] is L_k: (1, r_left, nsamples, nqoi)
                        L_weight = left_products[kk - 1][0, ii, :, :]

                    # Get weight from right product
                    if kk == nvars - 1:
                        # No right product for last core
                        R_weight = self._bkd.ones((nsamples, nqoi))
                    else:
                        # right_products[kk]: (r_right, 1, nsamples, nqoi)
                        R_weight = right_products[kk][jj, 0, :, :]

                    # Combined weight: (nsamples, nqoi)
                    weight = L_weight * R_weight

                    # core_jac[ii, jj] has shape (nsamples, nqoi)
                    contribution = contribution + weight * core_jac[ii, jj]

            jac_columns.append(contribution)

        # Stack columns: (nvars, nsamples, nqoi) -> transpose to (nsamples, nqoi, nvars)
        jac_stacked = self._bkd.stack(jac_columns, axis=0)  # (nvars, nsamples, nqoi)
        return self._bkd.transpose(jac_stacked, (1, 2, 0))  # (nsamples, nqoi, nvars)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample point. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars)
        """
        jac_batch = self.jacobian_batch(sample)  # (1, nqoi, nvars)
        return jac_batch[0, :, :]  # (nqoi, nvars)

    def __repr__(self) -> str:
        return (
            f"FunctionTrain(nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"nparams={self.nparams()})"
        )
