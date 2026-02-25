"""
OED data management utilities.

Provides save/load/subset operations for OED datasets, enabling the
two-phase workflow: generate expensive model evaluations once, then
run OED optimization many times using saved data.
"""

import pickle
from typing import Any, Dict, Generic, Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


class OEDDataManager(Generic[Array]):
    """
    Manages OED datasets: save, load, and subset.

    Supports the two-phase OED workflow where model evaluations are
    generated once (expensive) and then reused for multiple OED runs
    (cheap). Arrays are stored in the typing convention:
    (nqoi, nsamples) for values.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._data: Dict[str, Any] = {}

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def save_data(
        self,
        filename: str,
        outerloop_samples: Array,
        outerloop_shapes: Array,
        outerloop_weights: Array,
        observation_locations: Array,
        innerloop_samples: Array,
        innerloop_shapes: Array,
        innerloop_weights: Array,
        qoi_vals: Array,
        qoi_quad_weights: Array,
    ) -> None:
        """
        Save OED dataset to disk.

        Parameters
        ----------
        filename : str
            Path to save the dataset (pickle format).
        outerloop_samples : Array
            Outer loop parameter samples. Shape: (nvars, nouter)
        outerloop_shapes : Array
            Model outputs for outer samples. Shape: (nobs, nouter)
        outerloop_weights : Array
            Outer loop quadrature weights. Shape: (nouter,)
        observation_locations : Array
            Spatial coordinates of observations. Shape: (ndim, nobs)
        innerloop_samples : Array
            Inner loop parameter samples. Shape: (nvars, ninner)
        innerloop_shapes : Array
            Model outputs for inner samples. Shape: (nobs, ninner)
        innerloop_weights : Array
            Inner loop quadrature weights. Shape: (ninner,)
        qoi_vals : Array
            QoI values at inner samples. Shape: (nqoi, ninner) or
            (ninner, npred)
        qoi_quad_weights : Array
            QoI quadrature weights. Shape: (nqoi, 1) or (npred, 1)
        """
        data = {
            "outloop_samples": self._bkd.to_numpy(outerloop_samples),
            "outloop_shapes": self._bkd.to_numpy(outerloop_shapes),
            "outloop_quad_weights": self._bkd.to_numpy(outerloop_weights),
            "observation_locations": self._bkd.to_numpy(
                observation_locations
            ),
            "inloop_samples": self._bkd.to_numpy(innerloop_samples),
            "inloop_shapes": self._bkd.to_numpy(innerloop_shapes),
            "inloop_quad_weights": self._bkd.to_numpy(innerloop_weights),
            "qoi_vals": self._bkd.to_numpy(qoi_vals),
            "qoi_quad_weights": self._bkd.to_numpy(qoi_quad_weights),
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

        # Also store in memory
        self._data = {
            k: self._bkd.asarray(v) for k, v in data.items()
        }

    def load_data(self, filename: str) -> None:
        """
        Load OED dataset from disk.

        Parameters
        ----------
        filename : str
            Path to the saved dataset (pickle format).
        """
        with open(filename, "rb") as f:
            raw_data = pickle.load(f)

        self._data = {
            k: self._bkd.asarray(v) for k, v in raw_data.items()
        }

    def get(self, name: str) -> Array:
        """
        Get a named array from the loaded dataset.

        Parameters
        ----------
        name : str
            Name of the array. Valid names: "outloop_samples",
            "outloop_shapes", "outloop_quad_weights",
            "observation_locations", "inloop_samples",
            "inloop_shapes", "inloop_quad_weights",
            "qoi_vals", "qoi_quad_weights".

        Returns
        -------
        Array
            The requested array.

        Raises
        ------
        KeyError
            If the name is not found in the dataset.
        """
        if name not in self._data:
            available = list(self._data.keys())
            raise KeyError(
                f"'{name}' not found. Available: {available}"
            )
        return self._data[name]

    def nobservations(self) -> int:
        """Number of observation locations in the dataset."""
        return int(self._data["outloop_shapes"].shape[0])

    def extract_data_subset(
        self,
        active_obs_idx: Array,
        active_loc_idx: Array,
        nout_oed: int,
        nin_oed: int,
    ) -> Dict[str, Array]:
        """
        Extract a spatial and sample subset of the dataset.

        This is a cheap operation (array slicing only, no model calls).

        Parameters
        ----------
        active_obs_idx : Array
            Indices of active observations to keep. Shape: (nobs_active,)
        active_loc_idx : Array
            Indices of active spatial locations. Shape: (nloc_active,)
        nout_oed : int
            Number of outer samples to use (<= total outer samples).
        nin_oed : int
            Number of inner samples to use (<= total inner samples).

        Returns
        -------
        Dict[str, Array]
            Dictionary with the subsetted arrays:
            - "outloop_samples": (nvars, nout_oed)
            - "outloop_shapes": (nobs_active, nout_oed)
            - "outloop_quad_weights": (nout_oed,)
            - "observation_locations": (ndim, nloc_active)
            - "inloop_samples": (nvars, nin_oed)
            - "inloop_shapes": (nobs_active, nin_oed)
            - "inloop_quad_weights": (nin_oed,)
            - "qoi_vals": subsetted QoI values
            - "qoi_quad_weights": subsetted weights
        """
        obs_idx_np = self._bkd.to_numpy(active_obs_idx).astype(int)
        loc_idx_np = self._bkd.to_numpy(active_loc_idx).astype(int)

        # Subset outer loop
        outloop_samples = self._data["outloop_samples"][:, :nout_oed]
        outloop_shapes = self._data["outloop_shapes"][obs_idx_np, :nout_oed]
        outloop_weights = self._data["outloop_quad_weights"][:nout_oed]
        # Renormalize weights
        outloop_weights = outloop_weights / self._bkd.sum(outloop_weights)

        # Subset inner loop
        inloop_samples = self._data["inloop_samples"][:, :nin_oed]
        inloop_shapes = self._data["inloop_shapes"][obs_idx_np, :nin_oed]
        inloop_weights = self._data["inloop_quad_weights"][:nin_oed]
        inloop_weights = inloop_weights / self._bkd.sum(inloop_weights)

        # Subset observation locations
        obs_locs = self._data["observation_locations"][:, loc_idx_np]

        # Subset QoI values (keep all QoI dims, subset samples)
        qoi_vals = self._data["qoi_vals"]
        if qoi_vals.shape[0] == self._data["inloop_shapes"].shape[1]:
            # qoi_vals shape is (ninner, npred)
            qoi_vals = qoi_vals[:nin_oed, :]
        else:
            # qoi_vals shape is (nqoi, ninner)
            qoi_vals = qoi_vals[:, :nin_oed]

        return {
            "outloop_samples": outloop_samples,
            "outloop_shapes": outloop_shapes,
            "outloop_quad_weights": outloop_weights,
            "observation_locations": obs_locs,
            "inloop_samples": inloop_samples,
            "inloop_shapes": inloop_shapes,
            "inloop_quad_weights": inloop_weights,
            "qoi_vals": qoi_vals,
            "qoi_quad_weights": self._data["qoi_quad_weights"],
        }
