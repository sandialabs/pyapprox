"""Exceptions every backend raises the same way."""

import numpy as np


class LinAlgError(np.linalg.LinAlgError):
    """A linear-algebra failure (a singular matrix, a matrix that is not
    positive definite), raised by every backend whatever its library.

    Catch this class to handle the failure on any backend. It subclasses
    ``numpy.linalg.LinAlgError``, so handlers written for NumPy's class
    also catch it; it is not a ``torch.linalg.LinAlgError``, which a
    backend-generic module cannot import. The library's own error is
    chained as ``__cause__``, so its message stays in the traceback.
    """
