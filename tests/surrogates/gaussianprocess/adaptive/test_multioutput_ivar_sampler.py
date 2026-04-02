"""Tests for multioutput IVAR adaptive sampler.

Ports the legacy test_multioutput_ivar_update from
pyapprox/surrogates/gaussianprocess/tests/test_activelearning.py.

Tests cover:
- Basic selection: correct shapes, samples from candidates
- Objective validation: IVAR objective matches integrated variance reduction
  computed directly by fitting a MultiOutputGP and comparing prior vs
  posterior variance at quadrature points
"""

from typing import List

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.adaptive.multioutput_ivar_sampler import (
    MultiOutputIVARSampler,
)
from pyapprox.surrogates.gaussianprocess.multioutput import (
    MultiOutputGP,
)
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kernels.multioutput.multilevel import (
    MultiLevelKernel,
)
from pyapprox.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.util.test_utils import slow_test


class TestMultiOutputIVARSampler:
    """Tests for MultiOutputIVARSampler.

    Ports the legacy test_multioutput_ivar_update test from
    test_activelearning.py, adapted to the typing module API:
    - MultiLevelKernel replaces legacy mokernels.MultiLevelKernel
    - PolynomialScaling replaces construct_tensor_product_monomial_scaling
    - MultiOutputGP replaces MOExactGaussianProcess
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def _make_kernel(self, bkd) -> MultiLevelKernel:
        """Create a 3-level MultiLevelKernel with fixed hyperparameters.

        Structure: f_0 ~ GP(0, k_0), f_1 = 2.0*f_0 + d_1, f_2 = -3.0*f_1 + d_2
        where k_i are SE kernels with length scale 0.3.
        """
        nvars = 1
        nmodels = 3

        kernels = [
            SquaredExponentialKernel([0.3], (0.1, 1.0), nvars, bkd)
            for _ in range(nmodels)
        ]
        # Constant scalings: rho_0(x) = 2.0, rho_1(x) = -3.0
        scalings = [
            PolynomialScaling([2.0], (-3.0, 3.0), bkd, nvars=nvars, fixed=True),
            PolynomialScaling([-3.0], (-3.0, 3.0), bkd, nvars=nvars, fixed=True),
        ]
        kernel = MultiLevelKernel(kernels, scalings)
        # Fix all hyperparameters (no optimization)
        kernel.hyp_list().set_all_inactive()
        return kernel

    def _make_candidates(self, bkd, ncandidates_per_model: int = 21) -> List:
        """Create uniform candidate sets for each output in [0, 1]."""
        nmodels = 3
        return [
            bkd.linspace(0, 1, ncandidates_per_model)[None, :] for _ in range(nmodels)
        ]

    def test_multioutput_ivar_select_samples(self, bkd) -> None:
        """Selected samples have correct shapes and are from candidates.

        Creates a 3-output multi-fidelity IVAR sampler with costs [1, 2, 3],
        initializes with the midpoint of the highest-fidelity output,
        and selects 7 samples. Verifies:
        - Result is a list of 3 arrays
        - Each array has shape (nvars, n_selected_i)
        - Total selected count = 7
        - All selected samples are subsets of the corresponding candidates
        """
        nmodels = 3
        ncandidates_per_model = 21
        kernel = self._make_kernel(bkd)
        candidates_list = self._make_candidates(bkd, ncandidates_per_model)
        costs = bkd.asarray([0.05, 0.1, 1.0])

        sampler = MultiOutputIVARSampler(
            candidates_list, costs, bkd, nugget=0, nquad_samples=10000
        )
        # Initial pivot: midpoint of highest-fidelity output (last output)
        init_pivot = ncandidates_per_model * (nmodels - 1) + ncandidates_per_model // 2
        sampler.set_initial_pivots([init_pivot])
        sampler.set_kernel(kernel)

        nsamples = 7
        samples_list = sampler.select_samples(nsamples)

        # Check result is a list with correct length
        assert len(samples_list) == nmodels

        # Check each output has shape (nvars, n_i)
        total_selected = 0
        for ii, samples in enumerate(samples_list):
            assert samples.shape[0] == 1  # nvars = 1
            total_selected += samples.shape[1]
        assert total_selected == nsamples

        # Check all samples are from candidates
        for ii, (samples, candidates) in enumerate(zip(samples_list, candidates_list)):
            cand_np = bkd.to_numpy(candidates)
            samp_np = bkd.to_numpy(samples)
            for j in range(samp_np.shape[1]):
                found = any(
                    np.allclose(samp_np[:, j], cand_np[:, k])
                    for k in range(cand_np.shape[1])
                )
                assert found, \
                    f"Output {ii}, sample {j} not found in candidates"

    @slow_test
    def test_multioutput_ivar_objective_matches_integrated_variance(
        self, bkd,
    ) -> None:
        """IVAR objective matches integrated variance reduction.

        This is the key correctness test from the legacy
        test_multioutput_ivar_update. After selecting samples:
        1. Fit a MultiOutputGP with zero training values
        2. Compute prior variance (from kernel diagonal) at HF quad points
        3. Compute posterior variance (from GP prediction) at HF quad points
        4. Verify: best_obj_val approx -(E[prior_var] - E[posterior_var])

        Uses a strong cost profile [0.05, 0.1, 1.0] to ensure samples
        spread across outputs (not all going to HF). HF is the last
        output (model nmodels-1), which is the most expensive.
        """
        nmodels = 3
        ncandidates_per_model = 21
        kernel = self._make_kernel(bkd)
        candidates_list = self._make_candidates(bkd, ncandidates_per_model)
        # Strong cost profile: LF outputs are very cheap relative to HF
        costs = bkd.asarray([0.05, 0.1, 1.0])

        sampler = MultiOutputIVARSampler(
            candidates_list, costs, bkd, nugget=0, nquad_samples=5000
        )
        # Initial pivot: midpoint of highest-fidelity (last) output
        init_pivot = ncandidates_per_model * (nmodels - 1) + ncandidates_per_model // 2
        sampler.set_initial_pivots([init_pivot])
        sampler.set_kernel(kernel)

        nsamples = 7
        sampler.select_samples(nsamples)

        # Collect all selected samples per output (including initial pivots)
        all_indices = sampler.selected_indices()
        all_samples_list = sampler._partition_samples(all_indices)

        # Ensure all outputs have samples for GP fitting; add dummy
        # midpoint for any output with zero selected samples
        nvars = 1
        train_X_list = []
        train_y_list = []
        for ii in range(nmodels):
            if all_samples_list[ii].shape[1] == 0:
                dummy_x = bkd.asarray([[0.5]])
                train_X_list.append(dummy_x)
                train_y_list.append(bkd.zeros((1, 1)))
            else:
                train_X_list.append(all_samples_list[ii])
                train_y_list.append(bkd.zeros((1, all_samples_list[ii].shape[1])))

        # Fit GP with zero values (mimics legacy test)
        gp = MultiOutputGP(kernel, nugget=1e-10)
        kernel.hyp_list().set_all_inactive()
        gp._fit_internal(train_X_list, train_y_list)

        # Generate HF quadrature points for validation
        # Use same seed as P matrix computation in the sampler
        np.random.seed(42)
        nquad = 5000
        hf_quad_pts = bkd.asarray(np.random.rand(nvars, nquad))

        # For multi-level kernel, need list input with empty for LF
        empty = bkd.zeros((nvars, 0))
        quad_list = [empty for _ in range(nmodels - 1)] + [hf_quad_pts]

        # Prior variance at HF quadrature points: diagonal of K(HF, HF)
        K_diag_blocks = kernel(quad_list, block_format=True)
        prior_var_hf = bkd.diag(K_diag_blocks[nmodels - 1][nmodels - 1])

        # Posterior variance from fitted GP
        _, std_list = gp.predict_with_uncertainty(quad_list)
        posterior_var_hf = bkd.flatten(std_list[nmodels - 1]) ** 2

        # Integrated variance reduction:
        # obj_val = -(E[prior_var] - E[posterior_var])
        # Legacy: -(prior_std**2).mean() + (posterior_std**2).mean()
        prior_integrated = bkd.mean(prior_var_hf)
        posterior_integrated = bkd.mean(posterior_var_hf)
        expected_obj = -(prior_integrated - posterior_integrated)

        # Compare with sampler's tracked best objective
        actual_obj = sampler.best_obj_vals()[-1]
        bkd.assert_allclose(
            bkd.asarray([actual_obj]),
            bkd.asarray([float(bkd.to_numpy(bkd.reshape(expected_obj, (1,)))[0])]),
            rtol=5e-2,
        )
