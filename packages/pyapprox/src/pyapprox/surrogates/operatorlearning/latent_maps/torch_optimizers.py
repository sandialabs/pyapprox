r"""Configuring a torch optimizer before there is anything to optimize.

A ``torch.optim.Optimizer`` cannot be built until its parameters exist,
but the parameters belong to a copy a fitter makes internally, so a
caller cannot construct one. Hard-coding the optimizer inside the fitter
solves that and creates a worse problem: ``SGD(momentum=0.9)`` and
``LBFGS(line_search_fn="strong_wolfe")`` take different arguments, so
every optimizer needs a new fitter class, or a fitter grows a union of
every optimizer's knobs.

This is the deferred-binding pattern
:class:`~pyapprox.optimization.minimize.protocols.BindableOptimizerProtocol`
uses -- configure with options first, attach to a problem later --
narrowed to what torch needs. The wider protocol does not fit here: it
takes one flat parameter vector and requires bounds, while a network has
many unbounded tensors of different shapes, so satisfying it would mean
packing and unpacking every iteration to duplicate what ``torch.optim``
already does.

Chaining falls out. Adam has no line search, so it drifts away from an
exact optimum once the gradient is numerical noise; L-BFGS has one but
needs the whole sample and a good starting point. Running one then the
other is better than either, and :class:`ChainedTorchOptimizer` composes
them without a fitter knowing that it happened.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Unpack,
    runtime_checkable,
)

import torch


class TorchAdamOptions(TypedDict, total=False):
    """Adam's tunable arguments, beyond the step size.

    Typed rather than left to ``**kwargs: Any`` so a misspelled key is a
    static error instead of a ``TypeError`` thousands of epochs into a
    run. Covers what a caller tunes; the rest of torch's signature is
    performance and device plumbing, reachable by constructing a
    :class:`TorchOptimizerSpec` directly.
    """

    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool


class TorchLBFGSOptions(TypedDict, total=False):
    """L-BFGS's tunable arguments, beyond iteration count and tolerance."""

    max_eval: Optional[int]
    history_size: int
    line_search_fn: Optional[str]


@runtime_checkable
class TorchOptimizerSpecProtocol(Protocol):
    """Something that yields a configured optimizer, given parameters.

    The seam a caller injects to choose an optimizer and its settings.
    Deliberately not a name-to-class table: a caller constructs the spec
    they want and passes it, so adding an optimizer needs no change
    here.
    """

    def build(
        self, parameters: Iterable[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        """Return an optimizer over ``parameters``."""
        ...

    def nsteps(self) -> int:
        """How many times the driving loop should call ``step``."""
        ...

    def needs_full_sample(self) -> bool:
        """Whether stepping requires every sample rather than a batch.

        True for quasi-Newton methods, which build curvature from
        successive gradients and would have that estimate corrupted by a
        gradient that changes with the batch.
        """
        ...


class TorchOptimizerSpec:
    """A torch optimizer class plus the keyword arguments to build it.

    Parameters
    ----------
    optimizer_cls : type
        Any ``torch.optim.Optimizer`` subclass.
    nsteps : int
        Steps the driving loop takes.
    needs_full_sample : bool
        See :meth:`TorchOptimizerSpecProtocol.needs_full_sample`.
    **kwargs
        Passed to ``optimizer_cls``. ``Any`` here and nowhere else: this
        class is generic over every ``torch.optim`` subclass, and their
        arguments are disjoint -- ``betas`` for Adam, ``momentum`` for
        SGD, ``line_search_fn`` for L-BFGS -- so no single ``TypedDict``
        describes them. Torch's own constructors reject an unknown key,
        so the error is loud, just at construction rather than at type
        check.

        Prefer :func:`torch_adam` and :func:`torch_lbfgs`, which fix the
        class and so *can* type their options: a misspelling there is a
        static error rather than a runtime one. Reach for this
        constructor only for an optimizer those do not cover.
    """

    def __init__(
        self,
        optimizer_cls: type,
        nsteps: int = 1000,
        needs_full_sample: bool = False,
        **kwargs: Any,
    ) -> None:
        if not issubclass(optimizer_cls, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer_cls must be a torch.optim.Optimizer "
                f"subclass, got {optimizer_cls.__name__}"
            )
        if nsteps < 1:
            raise ValueError(f"nsteps must be positive, got {nsteps}")
        self._optimizer_cls = optimizer_cls
        self._nsteps = nsteps
        self._needs_full_sample = needs_full_sample
        self._kwargs: Dict[str, Any] = kwargs

    def build(
        self, parameters: Iterable[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        """Return the configured optimizer over ``parameters``."""
        return self._optimizer_cls(list(parameters), **self._kwargs)

    def nsteps(self) -> int:
        """How many times the driving loop should call ``step``."""
        return self._nsteps

    def needs_full_sample(self) -> bool:
        """Whether stepping requires every sample rather than a batch."""
        return self._needs_full_sample

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._optimizer_cls.__name__}, "
            f"nsteps={self._nsteps})"
        )


def torch_adam(
    learning_rate: float = 1e-2,
    nsteps: int = 1000,
    **options: Unpack[TorchAdamOptions],
) -> TorchOptimizerSpec:
    """Spec for Adam, the usual first stage.

    Cheap per step and tolerant of a poor starting point, which is what
    a first stage needs. It has no line search, so it does not finish
    the job -- see :func:`torch_lbfgs`.
    """
    return TorchOptimizerSpec(
        torch.optim.Adam, nsteps=nsteps, lr=learning_rate, **options
    )


def torch_lbfgs(
    nsteps: int = 100,
    tol: float = 1e-15,
    **options: Unpack[TorchLBFGSOptions],
) -> TorchOptimizerSpec:
    """Spec for L-BFGS with a strong-Wolfe line search, the usual polish.

    Converges an already-close fit far tighter than more first-order
    steps can: measured on a problem with an exact solution, Adam alone
    stalls near 5e-03 while a hundred L-BFGS iterations reach 1e-18. It
    needs the full sample, and ``nsteps`` here is ``max_iter``, so the
    driving loop calls ``step`` once.
    """
    options.setdefault("line_search_fn", "strong_wolfe")
    return TorchOptimizerSpec(
        torch.optim.LBFGS,
        nsteps=1,
        needs_full_sample=True,
        max_iter=nsteps,
        tolerance_grad=tol,
        tolerance_change=tol,
        **options,
    )


class ChainedTorchOptimizer:
    """Run several specs in order, each starting where the last stopped.

    The composition that makes a polish expressible without a fitter
    knowing what a polish is:
    ``ChainedTorchOptimizer([torch_adam(), torch_lbfgs()])`` is one spec
    as far as its consumer is concerned.

    Parameters
    ----------
    stages : list of TorchOptimizerSpecProtocol
        Run in the order given.
    """

    def __init__(self, stages: List[TorchOptimizerSpecProtocol]) -> None:
        if not stages:
            raise ValueError("stages must not be empty")
        for stage in stages:
            if not isinstance(stage, TorchOptimizerSpecProtocol):
                raise TypeError(
                    f"each stage must satisfy "
                    f"TorchOptimizerSpecProtocol, got "
                    f"{type(stage).__name__}"
                )
        self._stages = list(stages)

    def stages(self) -> List[TorchOptimizerSpecProtocol]:
        """Return the stages, in order."""
        return list(self._stages)

    def __repr__(self) -> str:
        names = ", ".join(repr(s) for s in self._stages)
        return f"{self.__class__.__name__}([{names}])"


def torch_adam_then_lbfgs(
    learning_rate: float = 1e-2,
    nepochs: int = 1000,
    npolish_iterations: int = 100,
    **options: Unpack[TorchAdamOptions],
) -> ChainedTorchOptimizer:
    """The default: Adam to get close, L-BFGS to finish.

    Named because it is the combination worth reaching for by default,
    not because the two are inseparable -- either stage alone is a valid
    spec.
    """
    return ChainedTorchOptimizer(
        [
            torch_adam(
                learning_rate=learning_rate, nsteps=nepochs, **options
            ),
            torch_lbfgs(nsteps=npolish_iterations),
        ]
    )
