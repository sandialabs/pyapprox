from abc import abstractmethod
import numpy as np

from pyapprox.surrogates.autogp.kernels import Kernel
from pyapprox.surrogates.autogp._torch_wrappers import (
    full, asarray, hstack, vstack, cholesky, solve_triangular, multidot,
    cos, to_numpy, atleast1d, repeat)
from pyapprox.surrogates.autogp.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform)
from pyapprox.surrogates.autogp.transforms import (
    SphericalCorrelationTransform)


class MultiOutputKernel(Kernel):
    def __init__(self, kernels, noutputs):
        self.kernels = kernels
        self.nkernels = len(kernels)
        self.noutputs = noutputs
        self.hyp_list = sum([kernel.hyp_list for kernel in kernels])

        self.nsamples_per_output = None

    def set_nsamples_per_output(self, nsamples_per_output):
        if len(nsamples_per_output) != self.noutputs:
            msg = "length of nsamples_per_output {0} ".format(
                nsamples_per_output)
            msg += "does not match self.noutputs {0}".format(
                self.noutputs)
            raise ValueError(msg)
        self.nsamples_per_output = np.asarray(nsamples_per_output, dtype=int)

    @staticmethod
    def _unpack_samples(samples, nsamples_per_output):
        samples_per_output = []
        lb, ub = 0, 0
        for nn in nsamples_per_output:
            lb = ub
            ub += nn
            samples_per_output.append(samples[:, lb:ub])
        return samples_per_output

    @abstractmethod
    def _scale_off_diag_block(self, samples_per_output_ii, ii,
                              samples_per_output_jj, jj, kk):
        raise NotImplementedError

    @abstractmethod
    def _scale_diag_block(self, samples_per_output_ii, ii, kk):
        raise NotImplementedError

    @abstractmethod
    def _scale_diag(self, samples_per_output_ii, ii, kk):
        raise NotImplementedError

    def _evaluate_diagonal_block(self, samples_per_output_ii, ii):
        block = 0
        for kk in range(self.nkernels):
            block_kk = self._scale_diag_block(samples_per_output_ii, ii, kk)
            if block_kk is not None:
                block += block_kk
        return block

    def _evaluate_off_diagonal_block(self, samples_per_output_ii, ii,
                                     samples_per_output_jj, jj,
                                     block_format):
        block = 0
        nonzero = False
        for kk in range(self.nkernels):
            block_kk = self._scale_off_diag_block(
                samples_per_output_ii, ii, samples_per_output_jj, jj, kk)
            if block_kk is not None:
                block += block_kk
                nonzero = True
        if not block_format:
            if nonzero:
                return block
            return full((samples_per_output_ii.shape[1],
                         samples_per_output_jj.shape[1]), 0.)
        if nonzero:
            return block
        return None

    def __call__(self, X, Y=None, block_format=False):
        """
        Parameters
        ----------
        block_format : list[list]
            only return upper-traingular blocks, and set lower-triangular
            blocks to None
        """
        if self.nsamples_per_output is None:
            raise ValueError("Must call self.set_nsamples_per_output")
        X = asarray(X)
        if Y is None:
            Y = X
        else:
            Y = asarray(Y)
        samples_per_output = self._unpack_samples(X, self.nsamples_per_output)
        matrix_blocks = [[None for jj in range(self.noutputs)]
                         for ii in range(self.noutputs)]
        for ii in range(self.noutputs):
            matrix_blocks[ii][ii] = self._evaluate_diagonal_block(
                samples_per_output[ii], ii)
            for jj in range(ii+1, self.noutputs):
                matrix_blocks[ii][jj] = self._evaluate_off_diagonal_block(
                    samples_per_output[ii], ii, samples_per_output[jj], jj,
                    block_format)
                if not block_format:
                    matrix_blocks[jj][ii] = matrix_blocks[ii][jj].T
        if not block_format:
            rows = [hstack(matrix_blocks[ii]) for ii in range(self.noutputs)]
            return vstack(rows)
        return matrix_blocks

    def diag(self, X):
        X = asarray(X)
        samples_per_output = self._unpack_samples(X, self.nsamples_per_output)
        diags = []
        for ii in range(self.noutputs):
            diag_ii = 0
            for kk in range(self.nkernels):
                diag_iikk = self._scale_diag(samples_per_output[ii], ii, kk)
                if diag_iikk is not None:
                    diag_ii += diag_iikk
            diags.append(diag_ii)
        return hstack(diags)


class SpatiallyScaledMultiOutputKernel(MultiOutputKernel):
    def __init__(self, kernels, scalings):
        super().__init__(kernels, len(kernels))
        self._validate_kernels_and_scalings(kernels, scalings)
        self.scalings = scalings
        self.hyp_list = (
            self.hyp_list+sum([scaling.hyp_list for scaling in scalings]))

    @abstractmethod
    def _validate_kernels_and_scalings(self, kernels, scalings):
        raise NotImplementedError

    @abstractmethod
    def _get_kernel_combination_matrix_entry(self, samples, ii, kk):
        raise NotImplementedError

    def _scale_off_diag_block(self, samples_per_output_ii, ii,
                              samples_per_output_jj, jj, kk):
        wmat_iikk = self._get_kernel_combination_matrix_entry(
            samples_per_output_ii, ii, kk)
        wmat_jjkk = self._get_kernel_combination_matrix_entry(
            samples_per_output_jj, jj, kk)
        if wmat_iikk is not None and wmat_jjkk is not None:
            kmat = self.kernels[kk](
                samples_per_output_ii, samples_per_output_jj)
            return wmat_iikk*kmat*wmat_jjkk.T
        return None

    def _scale_diag_block(self, samples_per_output_ii, ii, kk):
        wmat_iikk = self._get_kernel_combination_matrix_entry(
            samples_per_output_ii, ii, kk)
        if wmat_iikk is not None:
            kmat = self.kernels[kk](samples_per_output_ii)
            return wmat_iikk*kmat*wmat_iikk.T
        return None

    def _scale_diag(self, samples_per_output_ii, ii, kk):
        wmat_iikk = self._get_kernel_combination_matrix_entry(
            samples_per_output_ii, ii, kk)
        if wmat_iikk is not None:
            return wmat_iikk[:, 0]**2*self.kernels[kk].diag(
                samples_per_output_ii)
        return None


def _block_cholesky(L_A, L_A_inv_B, B, D, return_blocks):
    schur_comp = D-multidot((L_A_inv_B.T, L_A_inv_B))
    L_S = cholesky(schur_comp)
    chol_blocks = [L_A, L_A_inv_B.T, L_S]
    if return_blocks:
        return chol_blocks
    return vstack([
        hstack([chol_blocks[0], 0*L_A_inv_B]),
        hstack([chol_blocks[1], chol_blocks[2]])])


def block_cholesky(blocks, return_blocks=False):
    A, B = blocks[0]
    D = blocks[1][1]
    L_A = cholesky(A)
    L_A_inv_B = solve_triangular(L_A, B)
    return _block_cholesky(L_A, L_A_inv_B, B, D, return_blocks)


class MultiPeerKernel(SpatiallyScaledMultiOutputKernel):
    def _validate_kernels_and_scalings(self, kernels, scalings):
        if len(scalings) != len(kernels)-1:
            msg = "The number of scalings {0} must be one less than ".format(
                len(scalings))
            msg += "the number of kernels {0}".format(len(kernels))
            raise ValueError(msg)

    def _get_kernel_combination_matrix_entry(self, samples, ii, kk):
        if ii == self.noutputs-1:
            if kk < self.noutputs-1:
                return self.scalings[kk](samples)
            return full((samples.shape[1], 1), 1.)
        if ii == kk:
            return full((samples.shape[1], 1), 1.)
        return None

    def _cholesky(self, blocks, block_format=False):
        chol_blocks = []
        L_A_inv_B_list = []
        for ii in range(self.noutputs-1):
            row = [None for ii in range(self.noutputs)]
            for jj in range(self.noutputs):
                if jj == ii:
                    row[ii] = cholesky(blocks[ii][ii])
                elif not block_format:
                    row[jj] = full(
                        (blocks[ii][ii].shape[0],
                         blocks[jj][self.noutputs-1].shape[0]), 0.)
            chol_blocks.append(row)
            L_A_inv_B_list.append(solve_triangular(row[ii], blocks[ii][-1]))
        B = np.vstack([blocks[jj][-1] for jj in range(self.noutputs-1)]).T
        D = blocks[-1][-1]
        L_A_inv_B = vstack(L_A_inv_B_list)
        if not block_format:
            L_A = vstack([hstack(row[:-1]) for row in chol_blocks])
            return _block_cholesky(
                L_A, L_A_inv_B, B, D, block_format)
        return _block_cholesky(
                chol_blocks, L_A_inv_B, B, D, block_format)

    @staticmethod
    def _cholesky_blocks_to_dense(A, C, D):
        shape = sum([A[ii][ii].shape[0] for ii in range(len(A))])
        L = np.zeros((shape+C.shape[0], shape+D.shape[1]))
        cnt = 0
        for ii in range(len(A)):
            L[cnt:cnt+A[ii][ii].shape[0], cnt:cnt+A[ii][ii].shape[0]] = (
                A[ii][ii])
            cnt += A[ii][ii].shape[0]
        L[cnt:, :cnt] = C
        L[cnt:, cnt:] = D
        return L


class MultiLevelKernel(SpatiallyScaledMultiOutputKernel):
    def _validate_kernels_and_scalings(self, kernels, scalings):
        if len(scalings) != len(kernels)-1:
            msg = "The number of scalings {0} must be one less than ".format(
                len(scalings))
            msg += "the number of kernels {0}".format(len(kernels))
            raise ValueError(msg)

    def _get_kernel_combination_matrix_entry(self, samples, ii, kk):
        if ii == kk:
            return full((samples.shape[1], 1), 1.)
        if ii < kk:
            return None
        val = self.scalings[kk](samples)
        for jj in range(kk+1, ii):
            val *= self.scalings[jj](samples)
        return val


class LMCKernel(MultiOutputKernel):
    """
    Linear model of coregionalization (LMC)
    """
    def __init__(self, kernels, output_kernels, noutputs):
        super().__init__(kernels, noutputs)
        self.output_kernels = output_kernels
        self._validate_kernels()
        self.hyp_list = (
            self.hyp_list +
            sum([kernel.hyp_list for kernel in self.output_kernels]))

    def _validate_kernels(self):
        if len(self.output_kernels) != len(self.kernels):
            msg = "The number of kernels {0} and output_kernels {1}".format(
                len(self.kernels), len(self.output_kernels)) + (
                    " are inconsistent")
            raise ValueError(msg)
        for ii, kernel in enumerate(self.output_kernels):
            if not isinstance(kernel, SphericalCovariance):
                raise ValueError(
                    f"The {ii}-th output kernel is not a SphericalCovariance")

    def _scale_off_diag_block(self, samples_per_output_ii, ii,
                              samples_per_output_jj, jj, kk):
        kmat = self.kernels[kk](
            samples_per_output_ii, samples_per_output_jj)
        return self.output_kernels[kk](ii, jj)*kmat

    def _scale_diag_block(self, samples_per_output_ii, ii, kk):
        return self._scale_off_diag_block(
            samples_per_output_ii, ii, samples_per_output_ii, ii, kk)

    def _scale_diag(self, samples_per_output_ii, ii, kk):
        return self.output_kernels[kk](ii, ii)*self.kernels[kk].diag(
            samples_per_output_ii)

    def get_output_kernel_correlations_from_psi(self, kk):
        """
        Compute the correlation between the first output and all other outputs.

        This can easily be expressed in terms of psi. The other relationships
        are more complicated.

        This function is used for testing only
        """
        hyp_values = self.output_kernels[kk].hyp_list.get_values()
        psi = self.output_kernels[kk]._trans.map_theta_to_spherical(hyp_values)
        return cos(psi[1:, 1])


class ICMKernel(LMCKernel):
    """
    Intrinsic coregionalization model (ICM)
    """
    def __init__(self, latent_kernel, output_kernel, noutputs):
        super().__init__([latent_kernel], [output_kernel], noutputs)


class SphericalCovariance():
    def __init__(self, noutputs, radii=1, radii_bounds=[1e-1, 1],
                 angles=np.pi/2, angle_bounds=[0, np.pi],
                 radii_transform=IdentityHyperParameterTransform(),
                 angle_transform=IdentityHyperParameterTransform()):
        self.noutputs = noutputs
        self._trans = SphericalCorrelationTransform(self.noutputs)
        self._validate_bounds(radii_bounds, angle_bounds)
        self._radii = HyperParameter(
            "radii", self.noutputs, radii, radii_bounds, radii_transform)
        self._angles = HyperParameter(
            "angles", self._trans.ntheta-self.noutputs, angles, angle_bounds,
            angle_transform)
        self.hyp_list = HyperParameterList([self._radii, self._angles])
        self._cov_matrix = None
        self._previous_hyp_values = None

    def _validate_bounds(self, radii_bounds, angle_bounds):
        bounds = asarray(self._trans.get_spherical_bounds())
        # all theoretical radii_bounds are the same so just check one
        radii_bounds = atleast1d(radii_bounds)
        if radii_bounds.shape[0] == 2:
            radii_bounds = repeat(radii_bounds, self.noutputs)
        radii_bounds = radii_bounds.reshape((radii_bounds.shape[0]//2, 2))
        if (np.any(to_numpy(radii_bounds[:, 0] < bounds[:self.noutputs, 0])) or
                np.any(to_numpy(
                    radii_bounds[:, 1] > bounds[:self.noutputs, 1]))):
            raise ValueError("radii bounds are inconsistent")
        # all theoretical angle_bounds are the same so just check one
        angle_bounds = atleast1d(angle_bounds)
        if angle_bounds.shape[0] == 2:
            angle_bounds = repeat(
                angle_bounds, self._trans.ntheta-self.noutputs)
        angle_bounds = angle_bounds.reshape((angle_bounds.shape[0]//2, 2))
        if (np.any(to_numpy(angle_bounds[:, 0] < bounds[self.noutputs:, 0])) or
                np.any(to_numpy(
                    angle_bounds[:, 1] > bounds[self.noutputs:, 1]))):
            # print(angle_bounds)
            # print(bounds[self.noutputs:])
            raise ValueError("angle bounds are inconsistent")

    def __call__(self, ii, jj):
        hyp_values = self.hyp_list.get_values()
        if (self._previous_hyp_values is None or
            not np.allclose(self._previous_hyp_values,
                            hyp_values.detach(), atol=1e-14)):
            chol_factor = self._trans.map_to_cholesky(hyp_values)
            self._cov_matrix = multidot((chol_factor, chol_factor.T))
            self._previous_hyp_values = hyp_values.detach().numpy().copy()
        return self._cov_matrix[ii, jj]

    def __repr__(self):
        return "{0}(name={1}, nvars={2}, degree={3}, nterms={4})".format(
            self.__class__.__name__, self._coef.name, self.nvars,
            self.self.degree, self.nterms)


class CollaborativeKernel(LMCKernel):
    def __init__(self, latent_kernels, output_kernels, discrepancy_kernels,
                 noutputs):
        super().__init__(
            latent_kernels+discrepancy_kernels, output_kernels, noutputs)

    def _validate_kernels(self):
        if len(self.output_kernels)+self.noutputs != len(self.kernels):
            msg = "The number of kernels {0} and output_kernels {1}".format(
                len(self.kernels), len(self.output_kernels)) + (
                    " are inconsistent")
            raise ValueError(msg)
        for ii, kernel in enumerate(self.output_kernels):
            if not isinstance(kernel, SphericalCovariance):
                raise ValueError(
                    f"The {ii}-th output kernel is not a SphericalCovariance")

    def _scale_off_diag_block(self, samples_per_output_ii, ii,
                              samples_per_output_jj, jj, kk):
        if kk < self.nkernels-self.noutputs:
            return super()._scale_off_diag_block(
                samples_per_output_ii, ii, samples_per_output_jj, jj, kk)
        if kk-self.nkernels+self.noutputs == ii:
            return self.kernels[kk](
                samples_per_output_ii, samples_per_output_jj)
        return None

    def _scale_diag_block(self, samples_per_output_ii, ii, kk):
        if kk < self.nkernels-self.noutputs:
            return super()._scale_diag_block(
                samples_per_output_ii, ii, kk)
        if kk-self.nkernels+self.noutputs == ii:
            return self.kernels[kk](samples_per_output_ii)
        return None

    def _scale_diag(self, samples_per_output_ii, ii, kk):
        if kk < self.nkernels-self.noutputs:
            return super()._scale_diag(
                samples_per_output_ii, ii, kk)
        if kk-self.nkernels+self.noutputs == ii:
            return self.kernels[kk].diag(samples_per_output_ii)
        return None


def _recursive_latent_coefs(scaling_vals, noutputs, level):
    val = np.zeros(noutputs)
    val[level] = 1.
    if level == 0:
        return val
    return val + scaling_vals[level-1]*_recursive_latent_coefs(
        scaling_vals, noutputs, level-1)


def _get_recursive_scaling_matrix(scaling_vals, noutputs):
    # for scalar scalings only
    rows = [_recursive_latent_coefs(scaling_vals, noutputs, ll)[None, :]
            for ll in range(noutputs)]
    return np.vstack(rows)
