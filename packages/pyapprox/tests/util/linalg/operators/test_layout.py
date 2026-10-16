"""Tests for layouts and their mismatch checks."""

import pytest

from pyapprox.util.linalg.operators import Layout, LayoutMismatchError


class TestLayout:
    def test_equal_layouts_match(self) -> None:
        Layout("u", 5).require_same(Layout("u", 5), "x")

    def test_version_mismatch_raises(self) -> None:
        """A space rebuilt to the same size is still a different space."""
        old = Layout("u", 5, version=0)
        new = Layout("u", 5, version=1)
        with pytest.raises(LayoutMismatchError, match="version 0 != 1"):
            old.require_same(new, "operator domain")

    def test_name_mismatch_raises(self) -> None:
        """Equal sizes do not make two spaces the same space."""
        with pytest.raises(LayoutMismatchError, match="name 'u' != 'v'"):
            Layout("u", 5).require_same(Layout("v", 5), "rhs")

    def test_message_names_every_difference(self) -> None:
        with pytest.raises(LayoutMismatchError) as info:
            Layout("u", 5).require_same(Layout("u", 6, dtype="float32"), "rhs")
        message = str(info.value)
        assert "rhs" in message
        assert "size 5 != 6" in message
        assert "dtype 'float64' != 'float32'" in message
        assert "version" not in message
        assert "name" not in message

    def test_mismatch_is_a_value_error(self) -> None:
        assert issubclass(LayoutMismatchError, ValueError)

    def test_is_frozen_and_hashable(self) -> None:
        layout = Layout("u", 3)
        with pytest.raises(AttributeError):
            setattr(layout, "size", 4)
        assert len({layout, Layout("u", 3)}) == 1

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Layout("", 2)
        with pytest.raises(ValueError, match="size"):
            Layout("u", -1)
        with pytest.raises(ValueError, match="version"):
            Layout("u", 2, version=-1)
        with pytest.raises(ValueError, match="dtype"):
            Layout("u", 2, dtype="int64")
