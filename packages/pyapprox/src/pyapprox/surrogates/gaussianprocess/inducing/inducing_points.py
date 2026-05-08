"""Inducing point locations for variational Gaussian Processes.

Manages inducing point coordinates as optimizable hyperparameters.
Noise is NOT included here — it belongs to the likelihood.
"""

from typing import Generic, Tuple, Union

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameter, HyperParameterList


class InducingPoints(Generic[Array]):
    """Inducing point locations for sparse GP inference.

    Parameters
    ----------
    nvars : int
        Number of input variables.
    num_inducing : int
        Number of inducing points.
    bkd : Backend[Array]
        Backend for numerical operations.
    inducing_locations : Array
        Initial inducing point locations, shape (nvars, num_inducing).
    inducing_bounds : Array or Tuple[float, float]
        Bounds for inducing point coordinates, shape
        (nvars * num_inducing, 2) or (2,) to broadcast.
    """

    def __init__(
        self,
        nvars: int,
        num_inducing: int,
        bkd: Backend[Array],
        inducing_locations: Array,
        inducing_bounds: Union[Array, Tuple[float, float]],
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._num_inducing = num_inducing

        if inducing_locations.shape != (nvars, num_inducing):
            raise ValueError(
                f"inducing_locations must have shape ({nvars}, {num_inducing}), "
                f"got {inducing_locations.shape}"
            )

        n_flat = nvars * num_inducing
        bounds_arr = bkd.atleast_1d(bkd.asarray(inducing_bounds))
        if bounds_arr.ndim == 1:
            if bounds_arr.shape[0] != 2:
                raise ValueError(
                    "inducing_bounds must have shape (2,) or "
                    f"({n_flat}, 2), got {bounds_arr.shape}"
                )
            bounds_arr = bkd.tile(
                bkd.reshape(bounds_arr, (1, 2)),
                (n_flat, 1),
            )

        self._inducing = HyperParameter(
            "inducing_points",
            n_flat,
            bkd.flatten(inducing_locations),
            bounds_arr,
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._inducing])

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def get_samples(self) -> Array:
        """Return inducing point locations, shape (nvars, num_inducing)."""
        return self._bkd.reshape(
            self._inducing.get_values(),
            (self._nvars, self._num_inducing),
        )

    def nvars(self) -> int:
        return self._nvars

    def num_inducing(self) -> int:
        return self._num_inducing

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __repr__(self) -> str:
        return (
            f"InducingPoints(num_inducing={self._num_inducing}, "
            f"nvars={self._nvars})"
        )
