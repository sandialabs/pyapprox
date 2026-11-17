"""PyApprox: High-dimensional function approximation and estimation."""

from pyapprox._moved import install as _install_moved

name = "pyapprox"

_install_moved()

try:
    from pyapprox._version import __version__
except ImportError:
    __version__ = "unknown"
