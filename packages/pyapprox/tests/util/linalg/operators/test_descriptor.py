"""Tests for descriptors and their mismatch checks."""

import pytest

from pyapprox.util.linalg.operators import SpaceDescriptor, SpaceMismatchError


class TestLayout:
    def test_equal_layouts_match(self) -> None:
        SpaceDescriptor("u", 5).require_same(SpaceDescriptor("u", 5), "x")

    def test_version_mismatch_raises(self) -> None:
        """A space rebuilt to the same size is still a different space."""
        old = SpaceDescriptor("u", 5, version=0)
        new = SpaceDescriptor("u", 5, version=1)
        with pytest.raises(SpaceMismatchError, match="version 0 != 1"):
            old.require_same(new, "operator domain")

    def test_name_mismatch_raises(self) -> None:
        """Equal sizes do not make two spaces the same space."""
        with pytest.raises(SpaceMismatchError, match="name 'u' != 'v'"):
            SpaceDescriptor("u", 5).require_same(SpaceDescriptor("v", 5), "rhs")

    def test_message_names_every_difference(self) -> None:
        with pytest.raises(SpaceMismatchError) as info:
            SpaceDescriptor("u", 5).require_same(
                SpaceDescriptor("u", 6, dtype="float32"), "rhs"
            )
        message = str(info.value)
        assert "rhs" in message
        assert "size 5 != 6" in message
        assert "dtype 'float64' != 'float32'" in message
        assert "version" not in message
        assert "name" not in message

    def test_mismatch_is_a_value_error(self) -> None:
        assert issubclass(SpaceMismatchError, ValueError)

    def test_is_frozen_and_hashable(self) -> None:
        descriptor = SpaceDescriptor("u", 3)
        with pytest.raises(AttributeError):
            setattr(descriptor, "size", 4)
        assert len({descriptor, SpaceDescriptor("u", 3)}) == 1

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="name"):
            SpaceDescriptor("", 2)
        with pytest.raises(ValueError, match="size"):
            SpaceDescriptor("u", -1)
        with pytest.raises(ValueError, match="version"):
            SpaceDescriptor("u", 2, version=-1)
        with pytest.raises(ValueError, match="dtype"):
            SpaceDescriptor("u", 2, dtype="int64")
