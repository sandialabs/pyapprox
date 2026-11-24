"""The floating-point precision a backend is configured with."""

from enum import Enum


class Precision(Enum):
    """The width of a backend's working real dtype: ``SINGLE`` is 32 bits,
    ``DOUBLE`` 64. The complex working dtype pairs with it (64 or 128
    bits) and is never configured separately."""

    SINGLE = "single"
    DOUBLE = "double"
