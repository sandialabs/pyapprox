"""
Tests for summary statistic protocols and implementations.

All tests use dual-backend testing via the ``bkd`` fixture.
"""

import numpy as np
import pytest

from pyapprox.inverse.variational.summary import (
    Aggregation,
    FlattenAggregation,
    IdentityTransform,
    MaxAggregation,
    MeanAggregation,
    MeanAndVarianceAggregation,
    SummaryStatistic,
    Transform,
    TransformAggregateSummary,
)


class TestIdentityTransform:
    def test_returns_input_unchanged(self, bkd) -> None:
        t = IdentityTransform(nobs_dim=2)
        obs = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = t(obs)
        bkd.assert_allclose(result, obs, rtol=1e-12)

    def test_nfeatures(self, bkd) -> None:
        t = IdentityTransform(nobs_dim=3)
        assert t.nfeatures() == 3

    def test_satisfies_transform_protocol(self, bkd) -> None:
        t = IdentityTransform(nobs_dim=2)
        assert isinstance(t, Transform)

    def test_single_observation(self, bkd) -> None:
        t = IdentityTransform(nobs_dim=2)
        obs = bkd.asarray([[1.0], [2.0]])
        result = t(obs)
        assert result.shape == (2, 1)
        bkd.assert_allclose(result, obs, rtol=1e-12)


class TestMeanAggregation:
    def test_mean_of_single_row(self, bkd) -> None:
        agg = MeanAggregation(nfeatures=1, bkd=bkd)
        features = bkd.asarray([[1.0, 3.0, 5.0]])
        result = agg(features)
        assert result.shape == (1, 1)
        bkd.assert_allclose(result, bkd.asarray([[3.0]]), rtol=1e-12)

    def test_mean_of_multiple_rows(self, bkd) -> None:
        agg = MeanAggregation(nfeatures=2, bkd=bkd)
        features = bkd.asarray([[1.0, 3.0], [2.0, 4.0]])
        result = agg(features)
        assert result.shape == (2, 1)
        bkd.assert_allclose(result, bkd.asarray([[2.0], [3.0]]), rtol=1e-12)

    def test_single_observation(self, bkd) -> None:
        agg = MeanAggregation(nfeatures=2, bkd=bkd)
        features = bkd.asarray([[5.0], [7.0]])
        result = agg(features)
        assert result.shape == (2, 1)
        bkd.assert_allclose(result, bkd.asarray([[5.0], [7.0]]), rtol=1e-12)

    def test_nlabel_dims(self, bkd) -> None:
        agg = MeanAggregation(nfeatures=3, bkd=bkd)
        assert agg.nlabel_dims() == 3

    def test_satisfies_aggregation_protocol(self, bkd) -> None:
        agg = MeanAggregation(nfeatures=1, bkd=bkd)
        assert isinstance(agg, Aggregation)


class TestMeanAndVarianceAggregation:
    def test_output_shape(self, bkd) -> None:
        agg = MeanAndVarianceAggregation(nfeatures=2, bkd=bkd)
        features = bkd.asarray([[1.0, 3.0], [2.0, 4.0]])
        result = agg(features)
        assert result.shape == (4, 1)

    def test_values(self, bkd) -> None:
        agg = MeanAndVarianceAggregation(nfeatures=1, bkd=bkd)
        features = bkd.asarray([[1.0, 3.0, 5.0]])
        result = agg(features)
        # mean=3, var=8/3
        expected_mean = 3.0
        expected_var = np.var([1.0, 3.0, 5.0])
        bkd.assert_allclose(
            result,
            bkd.asarray([[expected_mean], [expected_var]]),
            rtol=1e-6,
        )

    def test_nlabel_dims(self, bkd) -> None:
        agg = MeanAndVarianceAggregation(nfeatures=3, bkd=bkd)
        assert agg.nlabel_dims() == 6

    def test_satisfies_aggregation_protocol(self, bkd) -> None:
        agg = MeanAndVarianceAggregation(nfeatures=1, bkd=bkd)
        assert isinstance(agg, Aggregation)


class TestFlattenAggregation:
    def test_output_shape(self, bkd) -> None:
        agg = FlattenAggregation(nfeatures=2, n_obs=3, bkd=bkd)
        features = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = agg(features)
        assert result.shape == (6, 1)

    def test_values(self, bkd) -> None:
        agg = FlattenAggregation(nfeatures=2, n_obs=2, bkd=bkd)
        features = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        result = agg(features)
        # Flatten: [[1,2],[3,4]] -> [1,2,3,4] (row-major)
        expected = bkd.asarray([[1.0], [2.0], [3.0], [4.0]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_wrong_n_obs_raises(self, bkd) -> None:
        agg = FlattenAggregation(nfeatures=2, n_obs=3, bkd=bkd)
        features = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="expects 3 observations"):
            agg(features)

    def test_nlabel_dims(self, bkd) -> None:
        agg = FlattenAggregation(nfeatures=2, n_obs=5, bkd=bkd)
        assert agg.nlabel_dims() == 10

    def test_satisfies_aggregation_protocol(self, bkd) -> None:
        agg = FlattenAggregation(nfeatures=1, n_obs=3, bkd=bkd)
        assert isinstance(agg, Aggregation)


class TestMaxAggregation:
    def test_output_shape(self, bkd) -> None:
        agg = MaxAggregation(nfeatures=2, bkd=bkd)
        features = bkd.asarray([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        result = agg(features)
        assert result.shape == (2, 1)

    def test_values(self, bkd) -> None:
        agg = MaxAggregation(nfeatures=2, bkd=bkd)
        features = bkd.asarray([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        result = agg(features)
        bkd.assert_allclose(result, bkd.asarray([[5.0], [6.0]]), rtol=1e-12)

    def test_nlabel_dims(self, bkd) -> None:
        agg = MaxAggregation(nfeatures=3, bkd=bkd)
        assert agg.nlabel_dims() == 3

    def test_satisfies_aggregation_protocol(self, bkd) -> None:
        agg = MaxAggregation(nfeatures=1, bkd=bkd)
        assert isinstance(agg, Aggregation)


class TestTransformAggregateSummary:
    def test_identity_mean(self, bkd) -> None:
        """IdentityTransform + MeanAggregation = observation mean."""
        summary = TransformAggregateSummary(
            IdentityTransform(nobs_dim=1),
            MeanAggregation(nfeatures=1, bkd=bkd),
            bkd,
        )
        obs = bkd.asarray([[1.0, 3.0, 5.0]])
        result = summary(obs)
        assert result.shape == (1, 1)
        bkd.assert_allclose(result, bkd.asarray([[3.0]]), rtol=1e-12)

    def test_identity_flatten(self, bkd) -> None:
        """IdentityTransform + FlattenAggregation = raw data vector."""
        summary = TransformAggregateSummary(
            IdentityTransform(nobs_dim=2),
            FlattenAggregation(nfeatures=2, n_obs=3, bkd=bkd),
            bkd,
        )
        obs = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = summary(obs)
        assert result.shape == (6, 1)

    def test_identity_mean_and_variance(self, bkd) -> None:
        """IdentityTransform + MeanAndVarianceAggregation."""
        summary = TransformAggregateSummary(
            IdentityTransform(nobs_dim=1),
            MeanAndVarianceAggregation(nfeatures=1, bkd=bkd),
            bkd,
        )
        obs = bkd.asarray([[1.0, 3.0, 5.0]])
        result = summary(obs)
        assert result.shape == (2, 1)
        expected_mean = 3.0
        expected_var = np.var([1.0, 3.0, 5.0])
        bkd.assert_allclose(
            result,
            bkd.asarray([[expected_mean], [expected_var]]),
            rtol=1e-6,
        )

    def test_nlabel_dims_matches_aggregation(self, bkd) -> None:
        summary = TransformAggregateSummary(
            IdentityTransform(nobs_dim=2),
            MeanAggregation(nfeatures=2, bkd=bkd),
            bkd,
        )
        assert summary.nlabel_dims() == 2

    def test_satisfies_summary_protocol(self, bkd) -> None:
        summary = TransformAggregateSummary(
            IdentityTransform(nobs_dim=1),
            MeanAggregation(nfeatures=1, bkd=bkd),
            bkd,
        )
        assert isinstance(summary, SummaryStatistic)

    def test_multivariate_mean(self, bkd) -> None:
        """2D observations, mean aggregation."""
        summary = TransformAggregateSummary(
            IdentityTransform(nobs_dim=2),
            MeanAggregation(nfeatures=2, bkd=bkd),
            bkd,
        )
        obs = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = summary(obs)
        assert result.shape == (2, 1)
        bkd.assert_allclose(result, bkd.asarray([[2.0], [5.0]]), rtol=1e-12)

    def test_variable_n_obs(self, bkd) -> None:
        """Mean summary handles different observation counts."""
        summary = TransformAggregateSummary(
            IdentityTransform(nobs_dim=1),
            MeanAggregation(nfeatures=1, bkd=bkd),
            bkd,
        )
        r2 = summary(bkd.asarray([[1.0, 3.0]]))
        r4 = summary(bkd.asarray([[1.0, 2.0, 3.0, 4.0]]))
        bkd.assert_allclose(r2, bkd.asarray([[2.0]]), rtol=1e-12)
        bkd.assert_allclose(r4, bkd.asarray([[2.5]]), rtol=1e-12)
