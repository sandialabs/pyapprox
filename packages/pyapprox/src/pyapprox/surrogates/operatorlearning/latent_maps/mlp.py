r"""A neural latent map, which is what makes POD-DeepONet a POD method.

The paper's POD-DeepONet is an operator surrogate whose output basis is a
precomputed POD basis and whose coefficient map is a network. Both of
those already exist here -- the basis as an encoder, the surrogate as
:class:`OperatorSurrogate` -- so the only missing piece is a
:class:`LatentMapProtocol` implementation that happens to be a network.
This is that piece, and it is deliberately just a multilayer perceptron
between two coefficient spaces.

**Torch only, and it says so at construction.** The forward pass is
``nn.Module`` ops that the ``Backend`` protocol cannot express, so under
a non-torch backend the autograd graph would be severed at this object's
output and ``loss.backward()`` would train nothing while raising nothing.
The precedent
:class:`~pyapprox.generative.flowmatching.nn_vf.MLPVelocityField` accepts
any backend and converts at its boundary, which is right for a velocity
field a numpy ODE stepper may evaluate. A latent map exists to be
*fitted*, so the same tolerance would buy a silent non-convergence
instead of an error. It refuses instead.
"""

from __future__ import annotations

from typing import Generic, List

import torch
import torch.nn as nn

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class MLPLatentMap(nn.Module, Generic[Array]):
    r"""A multilayer perceptron from input codes to output codes.

    Satisfies :class:`~pyapprox.surrogates.operatorlearning.protocols.LatentMapProtocol`
    -- ``nvars``, ``nqoi`` and ``__call__`` -- and nothing wider. In
    particular it declares no ``Derivatives`` bundle: fitting
    differentiates with respect to the *parameters*, which autograd does
    from ``parameters()``, and a bundle describes derivatives with
    respect to the *inputs*, which no consumer here wants. A
    half-populated bundle would be worse than none, per the convention
    that an absent capability is absent rather than present-and-raising.

    Parameters
    ----------
    ncodes_in : int
        Number of input coefficients, matching the input encoder's
        ``latent_dim``.
    ncodes_out : int
        Number of output coefficients, matching the output encoder's
        ``latent_dim``.
    hidden_dims : list of int
        Hidden layer widths. Empty gives a single affine map, which is
        the linear case the equivalence test uses.
    bkd : Backend[Array]
        Computational backend. Must be a ``TorchBkd``.
    activation : str
        One of ``"silu"``, ``"relu"``, ``"tanh"``.

    Raises
    ------
    TypeError
        If ``bkd`` is not a ``TorchBkd``. See the module docstring: the
        alternative is training that silently does nothing.
    """

    def __init__(
        self,
        ncodes_in: int,
        ncodes_out: int,
        hidden_dims: List[int],
        bkd: Backend[Array],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        # Stored before the isinstance check, following the
        # MLPVelocityField precedent: narrowing bkd to the concrete
        # TorchBkd would fix Array to Tensor and make every accessor
        # annotated Backend[Array] a type error.
        self._bkd = bkd
        if not isinstance(bkd, TorchBkd):
            raise TypeError(
                f"{type(self).__name__} is a torch module, so its forward "
                f"pass cannot preserve an autograd graph under "
                f"{type(bkd).__name__}: a fit would run, change nothing, "
                f"and report no error. Pass a TorchBkd."
            )
        if ncodes_in < 1 or ncodes_out < 1:
            raise ValueError(
                f"ncodes_in and ncodes_out must be positive, got "
                f"{ncodes_in} and {ncodes_out}"
            )
        activations = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in activations:
            raise ValueError(
                f"unknown activation {activation!r}, choose from "
                f"{sorted(activations)}"
            )
        self._ncodes_in = ncodes_in
        self._ncodes_out = ncodes_out
        self._dtype = bkd.default_dtype()

        layers: List[nn.Module] = []
        width = ncodes_in
        for hidden in hidden_dims:
            layers.append(nn.Linear(width, hidden, dtype=self._dtype))
            layers.append(activations[activation]())
            width = hidden
        layers.append(nn.Linear(width, ncodes_out, dtype=self._dtype))
        self._net = nn.Sequential(*layers)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of input coefficients the map consumes."""
        return self._ncodes_in

    def nqoi(self) -> int:
        """Number of output coefficients the map produces."""
        return self._ncodes_out

    def forward(self, coefs: Array) -> Array:
        """Map input codes to output codes.

        ``(nvars, nsamples) -> (nqoi, nsamples)``.

        Transposes in and out because ``nn.Linear`` is
        samples-down-rows while this package is samples-down-columns.
        The graph is preserved: under ``TorchBkd`` the argument is
        already a tensor and is passed through untouched, so a fitter
        can call this and differentiate the result.
        """
        if coefs.ndim != 2:
            raise ValueError(
                f"coefs must be 2D (nvars, nsamples), got shape "
                f"{coefs.shape}"
            )
        if int(coefs.shape[0]) != self._ncodes_in:
            raise ValueError(
                f"coefs has {int(coefs.shape[0])} rows but this map takes "
                f"{self._ncodes_in} input codes"
            )
        out = self._net(coefs.T)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"expected a tensor from the network, got "
                f"{type(out).__name__}"
            )
        # asarray, not array: the former passes a tensor through
        # untouched so the graph survives, the latter detaches and would
        # silently break the fit this class exists for. It also carries
        # the Tensor back to Array without a cast, since TorchBkd binds
        # Array to Tensor.
        return self._bkd.asarray(out.T)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(ncodes_in={self._ncodes_in}, "
            f"ncodes_out={self._ncodes_out})"
        )
