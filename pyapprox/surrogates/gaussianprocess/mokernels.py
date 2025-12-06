from abc import abstractmethod
from typing import List, Tuple

from pyapprox.surrogates.kernels.kernels import Kernel, SphericalCovariance
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.surrogates.affine.basis import MultiIndexBasis, Basis
from pyapprox.surrogates.affine.basisexp import BasisExpansion


class MultiOutputKernel(Kernel):
    def __init__(self, kernels: List[Kernel], noutputs: int):
        self._bkd = kernels[0]._bkd
        for kernel in kernels[1:]:
            if type(kernel._bkd) is not type(self._bkd):
                raise ValueError("kernels do not have the same backend")
        self._kernels = kernels
        self._nkernels = len(kernels)
        self._noutputs = noutputs
        self._hyp_list = sum([kernel.hyp_list() for kernel in kernels])

        self._nsamples_per_output_0 = None
        self._nsamples_per_output_1 = None

    def noutputs(self) -> int:
        return self._noutputs

    def kernels(self) -> List[Kernel]:
        return self._kernels

    def nkernels(self) -> int:
        return self._nkernels

    @abstractmethod
    def _scale_block(
        self,
        samples_per_output_ii: Array,
        ii: int,
        samples_per_output_jj: Array,
        jj: int,
        kk: int,
        symmetric: bool,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _scale_diag(
        self, samples_per_output_ii: Array, ii: int, kk: int
    ) -> Array:
        raise NotImplementedError

    def _evaluate_block(
        self,
        samples_per_output_ii: Array,
        ii: int,
        samples_per_output_jj: Array,
        jj: int,
        block_format: bool,
        symmetric: bool,
    ) -> Array:
        block = 0
        nonzero = False
        for kk in range(self._nkernels):
            block_kk = self._scale_block(
                samples_per_output_ii,
                ii,
                samples_per_output_jj,
                jj,
                kk,
                symmetric,
            )
            if block_kk is not None:
                block += block_kk
                nonzero = True
        if not block_format:
            if nonzero:
                return block
            return self._bkd.full(
                (
                    samples_per_output_ii.shape[1],
                    samples_per_output_jj.shape[1],
                ),
                0.0,
            )
        if nonzero:
            return block
        return None

    def __call__(
        self,
        samples_0: List[Array],
        samples_1: List[Array] = None,
        block_format: bool = False,
    ) -> Array:
        """
        Parameters
        ----------
        block_format : bool
            only return upper-traingular blocks, and set lower-triangular
            blocks to None
        """
        # samples_0 = [s for s in samples_0]
        if samples_1 is None:
            samples_1 = samples_0
            symmetric = True
        else:
            symmetric = False
        nsamples_0 = self._bkd.asarray([s.shape[1] for s in samples_0])
        nsamples_1 = self._bkd.asarray([s.shape[1] for s in samples_1])
        active_outputs_0 = self._bkd.where(nsamples_0 > 0)[0]
        active_outputs_1 = self._bkd.where(nsamples_1 > 0)[0]
        noutputs_0 = active_outputs_0.shape[0]
        noutputs_1 = active_outputs_1.shape[0]
        matrix_blocks = [
            [None for jj in range(noutputs_1)] for ii in range(noutputs_0)
        ]
        for ii in range(noutputs_0):
            idx0 = active_outputs_0[ii]
            for jj in range(noutputs_1):
                idx1 = active_outputs_1[jj]
                matrix_blocks[ii][jj] = self._evaluate_block(
                    samples_0[idx0],
                    idx0,
                    samples_1[idx1],
                    idx1,
                    block_format,
                    symmetric,
                )
        if not block_format:
            rows = [
                self._bkd.hstack(matrix_blocks[ii]) for ii in range(noutputs_0)
            ]
            return self._bkd.vstack(rows)
        return matrix_blocks

    def diag(self, samples_0: Array) -> Array:
        # samples_0 = [asarray(s) for s in samples_0]
        nsamples_0 = self._bkd.asarray([s.shape[1] for s in samples_0])
        active_outputs_0 = self._bkd.where(nsamples_0 > 0)[0]
        noutputs_0 = active_outputs_0.shape[0]
        diags = []
        for ii in range(noutputs_0):
            diag_ii = 0
            idx = active_outputs_0[ii]
            for kk in range(self._nkernels):
                diag_iikk = self._scale_diag(samples_0[idx], idx, kk)
                if diag_iikk is not None:
                    diag_ii += diag_iikk
            diags.append(diag_ii)
        return self._bkd.hstack(diags)

    def __repr__(self) -> str:
        if self._nsamples_per_output_0 is None:
            return super().__repr__()
        return "{0}({1}, nsamples_per_output={2})".format(
            self.__class__.__name__,
            self._hyp_list._short_repr(),
            self._nsamples_per_output_0,
        )


class SpatiallyScaledMultiOutputKernel(MultiOutputKernel):
    def __init__(self, kernels: List[Kernel], scalings: List[BasisExpansion]):
        super().__init__(kernels, len(kernels))
        self._validate_kernels_and_scalings(kernels, scalings)
        self._scalings = scalings
        self._hyp_list = self._hyp_list + sum(
            [scaling.hyp_list() for scaling in scalings]
        )

    @abstractmethod
    def _validate_kernels_and_scalings(
        self, kernels: List[Kernel], scalings: List[BasisExpansion]
    ):
        raise NotImplementedError

    @abstractmethod
    def _get_kernel_combination_matrix_entry(
        self, samples: Array, ii: int, kk: int
    ):
        raise NotImplementedError

    def _scale_block(
        self,
        samples_per_output_ii,
        ii,
        samples_per_output_jj,
        jj,
        kk,
        symmetric: bool,
    ) -> Array:
        wmat_iikk = self._get_kernel_combination_matrix_entry(
            samples_per_output_ii, ii, kk
        )
        wmat_jjkk = self._get_kernel_combination_matrix_entry(
            samples_per_output_jj, jj, kk
        )
        if wmat_iikk is not None and wmat_jjkk is not None:
            if ii != jj or not symmetric:
                kmat = self.kernels()[kk](
                    samples_per_output_ii, samples_per_output_jj
                )
            else:
                kmat = self.kernels()[kk](samples_per_output_ii)
            return wmat_iikk * kmat * wmat_jjkk.T
        return None

    def _scale_diag(
        self, samples_per_output_ii: Array, ii: int, kk: int
    ) -> Array:
        wmat_iikk = self._get_kernel_combination_matrix_entry(
            samples_per_output_ii, ii, kk
        )
        if wmat_iikk is not None:
            return wmat_iikk[:, 0] ** 2 * self.kernels()[kk].diag(
                samples_per_output_ii
            )
        return None


class MultiPeerKernel(SpatiallyScaledMultiOutputKernel):
    def _validate_kernels_and_scalings(
        self, kernels: List[Kernel], scalings: List[BasisExpansion]
    ):
        if len(scalings) != len(kernels) - 1:
            msg = "The number of scalings {0} must be one less than ".format(
                len(scalings)
            )
            msg += "the number of kernels {0}".format(len(kernels))
            raise ValueError(msg)

    def _get_kernel_combination_matrix_entry(
        self, samples: Array, ii: int, kk: int
    ) -> Array:
        if ii == self.noutputs() - 1:
            if kk < self.noutputs() - 1:
                return self._scalings[kk](samples)
            return self._bkd.full((samples.shape[1], 1), 1.0)
        if ii == kk:
            return self._bkd.full((samples.shape[1], 1), 1.0)
        return None

    @staticmethod
    def _cholesky(
        noutputs: int,
        blocks: List[List[Array]],
        bkd: BackendMixin,
        block_format: bool = False,
    ) -> Array:
        chol_blocks = []
        L_A_inv_B_list = []
        for ii in range(noutputs - 1):
            row = [None for ii in range(noutputs)]
            for jj in range(noutputs):
                if jj == ii:
                    row[ii] = bkd.cholesky(blocks[ii][ii])
                elif not block_format:
                    row[jj] = bkd.full(
                        (
                            blocks[ii][ii].shape[0],
                            blocks[jj][noutputs - 1].shape[0],
                        ),
                        0.0,
                    )
            chol_blocks.append(row)
            L_A_inv_B_list.append(
                bkd.solve_triangular(row[ii], blocks[ii][-1])
            )
        B = bkd.vstack([blocks[jj][-1] for jj in range(noutputs - 1)]).T
        D = blocks[-1][-1]
        L_A_inv_B = bkd.vstack(L_A_inv_B_list)
        if not block_format:
            L_A = bkd.vstack([bkd.hstack(row[:-1]) for row in chol_blocks])
            return bkd.block_cholesky_engine(
                L_A, L_A_inv_B, B, D, block_format
            )
        return bkd.block_cholesky_engine(
            chol_blocks, L_A_inv_B, B, D, block_format
        )

    @staticmethod
    def _cholesky_blocks_to_dense(
        A: Array, C: Array, D: Array, bkd: BackendMixin
    ) -> Array:
        shape = sum([A[ii][ii].shape[0] for ii in range(len(A))])
        L = bkd.full((shape + C.shape[0], shape + D.shape[1]), 0.0)
        cnt = 0
        for ii in range(len(A)):
            L[
                cnt : cnt + A[ii][ii].shape[0], cnt : cnt + A[ii][ii].shape[0]
            ] = A[ii][ii]
            cnt += A[ii][ii].shape[0]
        L[cnt:, :cnt] = C
        L[cnt:, cnt:] = D
        return L

    @staticmethod
    def _logdet(A: Array, C: Array, D: Array, bkd: BackendMixin) -> float:
        log_det = 0
        for ii, row in enumerate(A):
            log_det += 2 * bkd.log(bkd.get_diagonal(row[ii])).sum()
        log_det += 2 * bkd.log(bkd.get_diagonal(D)).sum()
        return log_det

    @staticmethod
    def _lower_solve_triangular(
        A: Array, C: Array, D: Array, values: Array, bkd: BackendMixin
    ) -> Array:
        # Solve Lx=y when L is the cholesky factor
        # of a peer kernel
        coefs = []
        cnt = 0
        for ii, row in enumerate(A):
            coefs.append(
                bkd.solve_triangular(
                    row[ii], values[cnt : cnt + row[ii].shape[0]], lower=True
                )
            )
            cnt += row[ii].shape[0]
        coefs = bkd.vstack(coefs)
        coefs = bkd.vstack(
            (
                coefs,
                bkd.solve_triangular(D, values[cnt:] - C @ coefs, lower=True),
            )
        )
        return coefs

    @staticmethod
    def _upper_solve_triangular(
        A: Array, C: Array, D: Array, values: Array, bkd: BackendMixin
    ) -> Array:
        # Solve L^Tx=y when L is the cholesky factor
        # of a peer kernel.
        # A, C, D all are from lower-triangular factor L (not L^T)
        # so must take transpose of all blocks
        idx1 = values.shape[0]
        idx0 = idx1 - D.shape[1]
        coefs = [bkd.solve_triangular(D.T, values[idx0:idx1], lower=False)]
        for ii, row in reversed(list(enumerate(A))):
            idx1 = idx0
            idx0 -= row[ii].shape[1]
            C_sub = C[:, idx0:idx1]
            coefs = [
                bkd.solve_triangular(
                    row[ii].T,
                    values[idx0:idx1] - C_sub.T @ coefs[-1],
                    lower=False,
                )
            ] + coefs
        coefs = bkd.vstack(coefs)
        return coefs

    @staticmethod
    def _cholesky_solve(
        A: Array, C: Array, D: Array, values: Array, bkd: BackendMixin
    ) -> Array:
        gamma = MultiPeerKernel._lower_solve_triangular(A, C, D, values, bkd)
        return MultiPeerKernel._upper_solve_triangular(A, C, D, gamma, bkd)


class MultiLevelKernel(SpatiallyScaledMultiOutputKernel):
    def _validate_kernels_and_scalings(
        self, kernels: List[Kernel], scalings: List[BasisExpansion]
    ):
        if len(scalings) != len(kernels) - 1:
            msg = "The number of scalings {0} must be one less than ".format(
                len(scalings)
            )
            msg += "the number of kernels {0}".format(len(kernels))
            raise ValueError(msg)

    def _get_kernel_combination_matrix_entry(
        self, samples: Array, ii: int, kk: int
    ) -> Array:
        if ii == kk:
            return self._bkd.full((samples.shape[1], 1), 1.0)
        if ii < kk:
            return None
        val = self._scalings[kk](samples)
        for jj in range(kk + 1, ii):
            val *= self._scalings[jj](samples)
        return val


class LMCKernel(MultiOutputKernel):
    """
    Linear model of coregionalization (LMC)
    """

    def __init__(
        self, kernels: List[Kernel], output_kernels: List[Kernel], noutputs
    ):
        super().__init__(kernels, noutputs)
        self.output_kernels = output_kernels
        self._validate_kernels()
        self._hyp_list = self._hyp_list + sum(
            [kernel.hyp_list() for kernel in self.output_kernels]
        )

    def _validate_kernels(self):
        if len(self.output_kernels) != len(self.kernels()):
            msg = "The number of kernels {0} and output_kernels {1}".format(
                len(self.kernels()), len(self.output_kernels)
            ) + (" are inconsistent")
            raise ValueError(msg)
        for ii, kernel in enumerate(self.output_kernels):
            if not isinstance(kernel, SphericalCovariance):
                raise ValueError(
                    f"The {ii}-th output kernel is not a SphericalCovariance"
                )

    def _scale_block(
        self,
        samples_per_output_ii: Array,
        ii: int,
        samples_per_output_jj: Array,
        jj: int,
        kk: int,
        symmetric: bool,
    ) -> Array:
        if ii != jj or not symmetric:
            kmat = self.kernels()[kk](
                samples_per_output_ii, samples_per_output_jj
            )
        else:
            kmat = self.kernels()[kk](samples_per_output_ii)
        return self.output_kernels[kk](ii, jj) * kmat

    def _scale_diag(self, samples_per_output_ii: Array, ii: int, kk: int):
        return self.output_kernels[kk](ii, ii) * self.kernels()[kk].diag(
            samples_per_output_ii
        )

    def get_output_kernel_correlations_from_psi(self, kk: int) -> Array:
        """
        Compute the correlation between the first output and all other outputs.

        This can easily be expressed in terms of psi. The other relationships
        are more complicated.

        This function is used for testing only
        """
        hyp_values = self.output_kernels[kk].hyp_list().get_values()
        psi = self.output_kernels[kk]._trans.map_theta_to_spherical(hyp_values)
        return self._bkd.cos(psi[1:, 1])


class ICMKernel(LMCKernel):
    """
    Intrinsic coregionalization model (ICM)
    """

    def __init__(self, latent_kernel, output_kernel, noutputs):
        super().__init__([latent_kernel], [output_kernel], noutputs)


class CollaborativeKernel(LMCKernel):
    def __init__(
        self,
        latent_kernels: List[Kernel],
        output_kernels: List[Kernel],
        discrepancy_kernels: List[Kernel],
        noutputs: int,
    ):
        super().__init__(
            latent_kernels + discrepancy_kernels, output_kernels, noutputs
        )

    def _validate_kernels(self):
        if len(self.output_kernels) + self.noutputs() != len(self.kernels()):
            msg = "The number of kernels {0} and output_kernels {1}".format(
                len(self.kernels()), len(self.output_kernels)
            ) + (" are inconsistent")
            raise ValueError(msg)
        for ii, kernel in enumerate(self.output_kernels):
            if not isinstance(kernel, SphericalCovariance):
                raise ValueError(
                    f"The {ii}-th output kernel is not a SphericalCovariance"
                )

    def _scale_block(
        self,
        samples_per_output_ii: Array,
        ii: int,
        samples_per_output_jj: Array,
        jj: int,
        kk: int,
        symmetric: bool,
    ) -> Array:
        if kk < self._nkernels - self.noutputs():
            # Evaluate latent kernel
            return super()._scale_block(
                samples_per_output_ii,
                ii,
                samples_per_output_jj,
                jj,
                kk,
                symmetric,
            )
        if kk - self._nkernels + self.noutputs() == min(ii, jj):
            # evaluate discrepancy kernel
            if ii != jj or not symmetric:
                return self.kernels()[kk](
                    samples_per_output_ii, samples_per_output_jj
                )
            return self.kernels()[kk](samples_per_output_ii)
        return None

    def _scale_diag(self, samples_per_output_ii: Array, ii, kk):
        if kk < self._nkernels - self.noutputs():
            return super()._scale_diag(samples_per_output_ii, ii, kk)
        if kk - self._nkernels + self.noutputs() == ii:
            return self.kernels()[kk].diag(samples_per_output_ii)
        return None


def _recursive_latent_coefs(
    scaling_vals: Array, noutputs: int, level: int, bkd: BackendMixin
) -> Array:
    val = bkd.zeros(noutputs)
    val[level] = 1.0
    if level == 0:
        return val
    return val + scaling_vals[level - 1] * _recursive_latent_coefs(
        scaling_vals, noutputs, level - 1, bkd
    )


def _get_recursive_scaling_matrix(
    scaling_vals: Array, noutputs: int, bkd: BackendMixin
) -> Array:
    # for scalar scalings only
    rows = [
        _recursive_latent_coefs(scaling_vals, noutputs, ll, bkd)[None, :]
        for ll in range(noutputs)
    ]
    return bkd.vstack(rows)


def _construct_scaling_from_basis(
    basis: Basis, bounds: Tuple[float, float], val: float, fixed: bool
) -> BasisExpansion:
    # set bounds on scaling
    bexp = BasisExpansion(basis, None, 1, bounds, fixed)
    bexp.set_coefficients(basis._bkd.full((bexp.basis().nterms(), 1), val))
    return bexp


def construct_tensor_product_monomial_scaling(
    nvars: int,
    nterms_1d: List[int],
    val: float,
    bounds: Tuple[float, float],
    fixed: bool = False,
    bkd: BackendMixin = NumpyMixin,
) -> BasisExpansion:
    basis = MultiIndexBasis([Monomial1D(backend=bkd) for ii in range(nvars)])
    basis.set_tensor_product_indices(nterms_1d)
    return _construct_scaling_from_basis(basis, bounds, val, fixed)

