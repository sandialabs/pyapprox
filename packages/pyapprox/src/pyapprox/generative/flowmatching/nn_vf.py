"""Neural network velocity field for flow matching."""

from typing import Generic

import torch
import torch.nn as nn

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class MLPVelocityField(nn.Module, Generic[Array]):
    """MLP velocity field for flow matching.

    Satisfies FunctionProtocol (bkd, nvars, nqoi, __call__) and provides
    jacobian_batch for density evaluation via divergence tracking.

    Internally uses torch for the MLP computation. At the public
    interface (``forward``, ``jacobian_batch``), arrays are converted
    from the caller's backend via ``bkd.to_numpy`` /
    ``torch.as_tensor``, then converted back via ``bkd.asarray``.

    For training, ``forward_torch`` and ``jacobian_batch_torch`` operate
    directly on torch tensors without boundary conversion.

    Parameters
    ----------
    nvars_in : int
        Input dimension (typically 1 + d for [t; x]).
    nqoi : int
        Output dimension.
    hidden_dims : list[int]
        Hidden layer sizes.
    bkd : Backend[Array]
        Computational backend for interface arrays.
    activation : str
        Activation function: "silu", "relu", or "tanh".
    """

    def __init__(
        self,
        nvars_in: int,
        nqoi: int,
        hidden_dims: list[int],
        bkd: Backend[Array],
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self._nvars_in = nvars_in
        self._nqoi = nqoi
        self._bkd = bkd
        self._is_torch = isinstance(bkd, TorchBkd)
        self._torch_dtype = (
            bkd.default_dtype() if self._is_torch else torch.float64
        )

        act_map = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in act_map:
            raise ValueError(
                f"Unknown activation {activation!r}. "
                f"Choose from {list(act_map.keys())}"
            )
        act_cls = act_map[activation]

        layers: list[nn.Module] = []
        in_dim = nvars_in
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h, dtype=self._torch_dtype))
            layers.append(act_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, nqoi, dtype=self._torch_dtype))
        self._net = nn.Sequential(*layers)

    def _to_tensor(self, arr: Array) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr
        return torch.as_tensor(
            self._bkd.to_numpy(arr), dtype=self._torch_dtype
        )

    def _from_tensor(self, t: torch.Tensor) -> Array:
        if self._is_torch:
            return t  # type: ignore[return-value]
        return self._bkd.asarray(t.detach().cpu().numpy())

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return input dimension."""
        return self._nvars_in

    def nqoi(self) -> int:
        """Return output dimension."""
        return self._nqoi

    def forward_torch(self, vf_input: torch.Tensor) -> torch.Tensor:
        """Evaluate velocity field on torch tensors (no conversion).

        Used by fitters during training to avoid boundary conversion
        overhead and preserve the autograd graph.

        Parameters
        ----------
        vf_input : torch.Tensor
            Shape ``(nvars_in, ns)``.

        Returns
        -------
        torch.Tensor
            Shape ``(nqoi, ns)``.
        """
        return self._net(vf_input.T).T

    def forward(self, vf_input: Array) -> Array:
        """Evaluate velocity field with backend boundary conversion.

        Parameters
        ----------
        vf_input : Array
            Shape ``(nvars_in, ns)``.

        Returns
        -------
        Array
            Shape ``(nqoi, ns)``.
        """
        x = self._to_tensor(vf_input)
        result = self.forward_torch(x)
        return self._from_tensor(result)

    def jacobian_batch_torch(
        self, vf_input: torch.Tensor
    ) -> torch.Tensor:
        """Jacobian on torch tensors (no conversion).

        Parameters
        ----------
        vf_input : torch.Tensor
            Shape ``(nvars_in, ns)``.

        Returns
        -------
        torch.Tensor
            Shape ``(ns, nqoi, nvars_in)``.
        """
        x_t = vf_input.T  # (ns, nvars_in)
        fn = lambda z: self._net(z)  # noqa: E731
        return torch.func.vmap(torch.func.jacrev(fn))(x_t)

    def jacobian_batch(self, vf_input: Array) -> Array:
        """Jacobian of output w.r.t. input for each sample.

        Parameters
        ----------
        vf_input : Array
            Shape ``(nvars_in, ns)``.

        Returns
        -------
        Array
            Shape ``(ns, nqoi, nvars_in)``.
        """
        x = self._to_tensor(vf_input)
        jac = self.jacobian_batch_torch(x)
        return self._from_tensor(jac)
