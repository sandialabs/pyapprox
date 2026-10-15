r"""Energy-conserving sampling and weighting (ECSW) hyper-reduction.

The exact reduced residual is a sum over every element,
:math:`\sum_e V_e^T F_e`. ECSW keeps a few elements with non-negative
weights,

.. math:: \hat{F}(z) \approx \sum_{e \in \tilde{E}} \xi_e V_e^T F_e(o_e + V_e z),

fitted so the weighted sum reproduces the full one on training data.
Non-negative weights keep each element's contribution with its own
sign, so a reduced Jacobian assembled from positive semidefinite element
blocks stays positive semidefinite; each kept element is evaluated with
the true physics. The number of kept elements is the online cost.

Three steps, each a separate call:

1. :func:`ecsw_training_matrix` evaluates, at each training sample, what
   the injected targets ask to be reproduced (the reduced residual, its
   Jacobian, ...) for every element, one column per element, from the
   exact element-sampled residual :func:`unit_weight_residual`. This is
   the full-order cost, so it is done once; how it is computed (in
   blocks, in parallel, from stored samples) is the caller's choice.
2. :func:`fit_ecsw` solves :math:`\min_{\xi \geq 0} \|G \xi - G
   \mathbf{1}\|` with an injected NNLS solver. The kept elements are its
   positive weights: there is no separate choice of where to sample.
   Stopping the solver at a relative tolerance, not at full convergence,
   makes the weights sparse; a tolerance near the basis truncation level
   is the usual choice. Trying another tolerance or solver reuses
   :math:`G`.
3. :func:`build_ecsw` restricts the element operator to the kept
   elements and returns the hyper-reduced residual.

Scope: the operator reduced is the element-wise interior one. Natural
boundary terms are not part of it and would be reduced separately; the
lift must be time-invariant.
"""

from dataclasses import dataclass
from typing import Generic, List, Protocol, Sequence, runtime_checkable

from pyapprox.pde.elementwise import ElementSet, ElementwiseOperatorProtocol
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.pde.rom.lift import LiftProtocol
from pyapprox.pde.rom.sampled import ElementSampledResidual
from pyapprox.pde.rom.trial_space import LinearTrialSpaceProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.nnls import NNLSResult, NNLSSolverProtocol


@runtime_checkable
class ElementTargetProtocol(Protocol, Generic[Array]):
    """What one training sample asks the weights to reproduce.

    Returns rows of the training matrix: one column per element, each
    row a quantity whose element contributions the weighted sum must
    match. Which quantities (residual, Jacobian, energy, a normalized
    variant) decides where the weights land, so it is injected rather
    than selected by a flag.
    """

    def rows(
        self,
        full: ElementSampledResidual[Array],
        code: Array,
        time: float,
        param: Array,
    ) -> Array:
        """Return the rows for one sample. Shape: (nrows, nelements)."""
        ...


class ResidualTargets(Generic[Array]):
    r"""The reduced residual, :math:`V_e^T F_e`: ``ncodes`` rows per sample.

    The original ECSW target; matching it preserves the virtual work of
    the reduced residual.
    """

    def rows(
        self,
        full: ElementSampledResidual[Array],
        code: Array,
        time: float,
        param: Array,
    ) -> Array:
        """Return the residual rows. Shape: (ncodes, nelements)."""
        return full.bkd().transpose(full.element_contributions(code, time, param))


class JacobianTargets(Generic[Array]):
    r"""The reduced Jacobian, :math:`V_e^T J_e V_e`: ``ncodes**2`` rows.

    Matching it as well keeps Newton on the hyper-reduced model close to
    Newton on the exact projection.
    """

    def rows(
        self,
        full: ElementSampledResidual[Array],
        code: Array,
        time: float,
        param: Array,
    ) -> Array:
        """Return the Jacobian rows. Shape: (ncodes**2, nelements)."""
        bkd = full.bkd()
        jacobians = full.element_jacobian_contributions(code, time, param)
        nelements = int(jacobians.shape[0])
        return bkd.transpose(bkd.reshape(jacobians, (nelements, -1)))


@dataclass(frozen=True)
class ECSWFit(Generic[Array]):
    """The hyper-reduced residual and how it was fitted.

    Attributes
    ----------
    residual : ElementSampledResidual
        The hyper-reduced residual over the kept elements.
    kept : ElementSet
        The kept elements.
    solver_result : NNLSResult
        The weight solve: weights of every element (zero for those
        dropped) and the achieved relative residual.
    """

    residual: ElementSampledResidual[Array]
    kept: ElementSet[Array]
    solver_result: NNLSResult[Array]

    def nkept(self) -> int:
        """Return the number of kept elements."""
        return self.kept.nselected()

    def nelements(self) -> int:
        """Return the number of elements fitted over."""
        return int(self.solver_result.weights.shape[0])


def unit_weight_residual(
    elementwise: ElementwiseOperatorProtocol[Array],
    parameterization: ParameterizationProtocol[Array],
    trial_space: LinearTrialSpaceProtocol[Array],
    lift: LiftProtocol[Array],
) -> ElementSampledResidual[Array]:
    """The exact projection as a sum over every element (unit weights).

    Its per-element contributions are the columns of the training
    matrix; it is also the reference a fit reproduces.
    """
    if not isinstance(elementwise, ElementwiseOperatorProtocol):
        raise TypeError(
            "elementwise must satisfy ElementwiseOperatorProtocol, got "
            f"{type(elementwise).__name__}"
        )
    bkd = elementwise.bkd()
    nelements = elementwise.nelements()
    return ElementSampledResidual.with_weights(
        elementwise.restrict(ElementSet.all(nelements, bkd)),
        parameterization,
        trial_space,
        lift,
        bkd.ones((nelements,)),
    )


def ecsw_training_matrix(
    full: ElementSampledResidual[Array],
    codes: Array,
    params: Array,
    targets: Sequence[ElementTargetProtocol[Array]],
    time: float = 0.0,
) -> Array:
    """Stack every target's rows for every training sample.

    Parameters
    ----------
    full : ElementSampledResidual
        The unit-weight residual over every element
        (:func:`unit_weight_residual`), so each column holds one
        element's contributions.
    codes : Array
        Reduced states to train at, e.g. projected snapshots and Newton
        iterates. Shape: (ncodes, ntrain).
    params : Array
        The parameter of each. Shape: (nparams, ntrain).
    targets : sequence of ElementTargetProtocol
        What each sample must reproduce, e.g. ``(ResidualTargets(),)``,
        or with ``JacobianTargets()`` as well.
    time : float
        Time the data is frozen at.

    Returns
    -------
    Array
        Shape: (nrows, nelements).

    Raises
    ------
    TypeError
        If a target does not satisfy its protocol.
    ValueError
        If the training arrays disagree or no target is given.
    """
    if codes.ndim != 2 or params.ndim != 2:
        raise ValueError("codes and params must be 2D")
    if codes.shape[1] != params.shape[1]:
        raise ValueError("codes and params need the same number of columns")
    if not targets:
        raise ValueError("at least one target is required")
    for target in targets:
        if not isinstance(target, ElementTargetProtocol):
            raise TypeError(
                "targets must satisfy ElementTargetProtocol, got "
                f"{type(target).__name__}"
            )
    blocks: List[Array] = []
    for kk in range(int(codes.shape[1])):
        for target in targets:
            blocks.append(target.rows(full, codes[:, kk], time, params[:, kk]))
    return full.bkd().vstack(blocks)


def fit_ecsw(
    training_matrix: Array,
    solver: NNLSSolverProtocol[Array],
    bkd: Backend[Array],
) -> NNLSResult[Array]:
    r"""Solve for non-negative weights reproducing the full element sums.

    Solves :math:`\min_{\xi \geq 0} \|G \xi - G \mathbf{1}\|`: the target
    is the unit-weight sum, the exact projection on the training data.
    The positive weights are the kept elements.

    Raises
    ------
    TypeError
        If ``solver`` does not satisfy its protocol.
    """
    if not isinstance(solver, NNLSSolverProtocol):
        raise TypeError(
            f"solver must satisfy NNLSSolverProtocol, got {type(solver).__name__}"
        )
    return solver.solve(training_matrix, bkd.sum(training_matrix, axis=1))


def build_ecsw(
    elementwise: ElementwiseOperatorProtocol[Array],
    parameterization: ParameterizationProtocol[Array],
    trial_space: LinearTrialSpaceProtocol[Array],
    lift: LiftProtocol[Array],
    weights: NNLSResult[Array],
) -> ECSWFit[Array]:
    """Restrict to the elements with positive weight and build the residual.

    Raises
    ------
    ValueError
        If no weight is positive.
    """
    bkd = elementwise.bkd()
    weights_np = bkd.to_numpy(weights.weights)
    kept_list = [
        e for e in range(int(weights_np.shape[0])) if float(weights_np[e]) > 0.0
    ]
    if not kept_list:
        raise ValueError("the weight solve kept no element")
    kept_indices = bkd.asarray(kept_list, dtype=bkd.int64_dtype())
    kept = ElementSet(kept_indices, bkd)
    residual = ElementSampledResidual.with_weights(
        elementwise.restrict(kept),
        parameterization,
        trial_space,
        lift,
        weights.weights[kept_indices],
    )
    return ECSWFit(residual, kept, weights)
