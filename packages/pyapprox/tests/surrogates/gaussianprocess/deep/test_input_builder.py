"""Tests for InputBuilder strategies."""

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.deep.input_builder import (
    InputBuilder,
    PureCompositionBuilder,
    RootBuilder,
    SkipConnectedBuilder,
)


class TestSkipConnectedBuilder:
    def test_no_parents_returns_X(self, bkd):
        builder = SkipConnectedBuilder()
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        result = builder.build(X, [], bkd)
        bkd.assert_allclose(result, X)

    def test_concatenates_X_and_parents(self, bkd):
        builder = SkipConnectedBuilder()
        X = bkd.array([[1.0, 2.0]])
        p1 = bkd.array([[10.0, 20.0]])
        p2 = bkd.array([[100.0, 200.0]])
        result = builder.build(X, [p1, p2], bkd)
        expected = bkd.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
        bkd.assert_allclose(result, expected)

    def test_input_dim_no_parents(self):
        builder = SkipConnectedBuilder()
        assert builder.input_dim(3, []) == 3

    def test_input_dim_with_parents(self):
        builder = SkipConnectedBuilder()
        assert builder.input_dim(2, [1, 1]) == 4

    def test_satisfies_protocol(self):
        assert isinstance(SkipConnectedBuilder(), InputBuilder)


class TestPureCompositionBuilder:
    def test_no_parents_returns_X(self, bkd):
        builder = PureCompositionBuilder()
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        result = builder.build(X, [], bkd)
        bkd.assert_allclose(result, X)

    def test_parents_only(self, bkd):
        builder = PureCompositionBuilder()
        X = bkd.array([[1.0, 2.0]])
        p1 = bkd.array([[10.0, 20.0]])
        p2 = bkd.array([[100.0, 200.0]])
        result = builder.build(X, [p1, p2], bkd)
        expected = bkd.array([[10.0, 20.0], [100.0, 200.0]])
        bkd.assert_allclose(result, expected)

    def test_input_dim_no_parents(self):
        builder = PureCompositionBuilder()
        assert builder.input_dim(3, []) == 3

    def test_input_dim_with_parents(self):
        builder = PureCompositionBuilder()
        assert builder.input_dim(3, [1, 1]) == 2

    def test_satisfies_protocol(self):
        assert isinstance(PureCompositionBuilder(), InputBuilder)


class TestRootBuilder:
    def test_returns_X(self, bkd):
        builder = RootBuilder()
        X = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        result = builder.build(X, [], bkd)
        bkd.assert_allclose(result, X)

    def test_raises_if_parents_given(self, bkd):
        builder = RootBuilder()
        X = bkd.array([[1.0, 2.0]])
        p1 = bkd.array([[10.0, 20.0]])
        with pytest.raises(ValueError, match="non-root"):
            builder.build(X, [p1], bkd)

    def test_input_dim(self):
        builder = RootBuilder()
        assert builder.input_dim(5, []) == 5

    def test_input_dim_raises_with_parents(self):
        builder = RootBuilder()
        with pytest.raises(ValueError, match="parent"):
            builder.input_dim(5, [1])

    def test_satisfies_protocol(self):
        assert isinstance(RootBuilder(), InputBuilder)
