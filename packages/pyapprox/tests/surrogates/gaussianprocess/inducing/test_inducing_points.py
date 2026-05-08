"""Tests for InducingPoints."""

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.inducing.inducing_points import (
    InducingPoints,
)


class TestInducingPoints:
    def test_shape_and_values(self, bkd):
        nvars, M = 2, 5
        locs = bkd.array(np.random.randn(nvars, M))
        ip = InducingPoints(nvars, M, bkd, locs, (-10.0, 10.0))
        bkd.assert_allclose(ip.get_samples(), locs)
        assert ip.get_samples().shape == (nvars, M)

    def test_hyp_list_nparams(self, bkd):
        nvars, M = 3, 4
        locs = bkd.array(np.random.randn(nvars, M))
        ip = InducingPoints(nvars, M, bkd, locs, (-10.0, 10.0))
        assert ip.hyp_list().nparams() == nvars * M

    def test_active_inactive_toggling(self, bkd):
        nvars, M = 2, 3
        locs = bkd.array(np.random.randn(nvars, M))
        ip = InducingPoints(nvars, M, bkd, locs, (-10.0, 10.0))

        assert ip.hyp_list().nactive_params() == nvars * M
        ip.hyp_list().set_all_inactive()
        assert ip.hyp_list().nactive_params() == 0
        ip.hyp_list().set_all_active()
        assert ip.hyp_list().nactive_params() == nvars * M

    def test_nvars_and_num_inducing(self, bkd):
        nvars, M = 2, 7
        locs = bkd.array(np.random.randn(nvars, M))
        ip = InducingPoints(nvars, M, bkd, locs, (-10.0, 10.0))
        assert ip.nvars() == nvars
        assert ip.num_inducing() == M

    def test_invalid_shape_raises(self, bkd):
        with pytest.raises(ValueError, match="inducing_locations must have shape"):
            InducingPoints(
                2, 3, bkd, bkd.array(np.random.randn(3, 2)), (-10.0, 10.0)
            )

    def test_set_active_values_roundtrip(self, bkd):
        nvars, M = 2, 3
        locs = bkd.array(np.random.randn(nvars, M))
        ip = InducingPoints(nvars, M, bkd, locs, (-10.0, 10.0))

        new_vals = bkd.array(np.ones(nvars * M) * 0.5)
        ip.hyp_list().set_active_values(new_vals)
        bkd.assert_allclose(
            ip.get_samples(), bkd.full((nvars, M), 0.5)
        )
