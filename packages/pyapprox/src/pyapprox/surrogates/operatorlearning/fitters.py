r"""Weighted least-squares fitting of an operator surrogate.

Once fields are encoded the fit is a vector-valued regression: the
input coefficients are its samples, the output coefficients its
quantities of interest, and :math:`C` its coefficient array of shape
``(nterms, nqoi)``. So the fitter holds a latent map rather than
reimplementing one, and takes the same
``fit(latent_map, samples, values)`` shape as its siblings in
``affine/expansions/fitters``.

This fitter solves one linear least squares problem, which is possible
only when the map is linear in its parameters -- hence
``LinearInParamsLatentMapProtocol`` rather than the bare
``LatentMapProtocol``. A neural latent map satisfies the latter and not
the former, and wants a gradient-based fitter instead.

The design matrix reaches the solver intact. ``LinearSystemSolver``
applies :math:`\sqrt{w}` to both it and the right-hand side and then
solves once with :math:`d_{\mathrm{out}}` columns, so the weighting is
a row scaling of a single least squares problem rather than something
folded into normal equations.
"""

from __future__ import annotations

from typing import Generic, List, Optional, Sequence

from pyapprox.optimization.linear import LeastSquaresSolver
from pyapprox.surrogates.affine.protocols.solver import (
    LinearSystemSolverProtocol,
    WeightedSolverProtocol,
)
from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
    LinearInParamsLatentMapProtocol,
    is_linear_in_params,
    is_multi_index,
    require_coefficient_error_is_field_error,
)
from pyapprox.surrogates.operatorlearning.surrogate import OperatorSurrogate
from pyapprox.util.backends.protocols import Array, Backend


class OperatorFitResult(Generic[Array]):
    """Result of fitting an operator surrogate.

    Parameters
    ----------
    surrogate : OperatorSurrogate[Array]
        The fitted surrogate, mapping input fields to output fields.
    params : Array
        Fitted coefficients :math:`C`. Shape: (nterms, noutputs)
    """

    def __init__(
        self,
        surrogate: OperatorSurrogate[Array],
        params: Array,
    ) -> None:
        self._surrogate = surrogate
        self._params = params

    def surrogate(self) -> OperatorSurrogate[Array]:
        """Return the fitted surrogate."""
        return self._surrogate

    def params(self) -> Array:
        """Return the fitted coefficients. Shape: (nterms, noutputs)."""
        return self._params

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._surrogate.bkd()

    def _require_affine(self) -> Array:
        """Return the per-index total degrees of an affine surrogate.

        Raises
        ------
        TypeError
            If the latent map has no multi-index basis, so affineness
            is not a question it can answer.
        ValueError
            If it has one and some index exceeds first order.
        """
        latent_map = self._surrogate.latent_map()
        if not is_multi_index(latent_map):
            raise TypeError(
                f"operator_matrix and intercept describe an affine map "
                f"through its index set, which "
                f"{type(latent_map).__name__} does not have. Only a "
                f"latent map satisfying MultiIndexLatentMapProtocol can "
                f"be asked."
            )
        if not self._surrogate.is_affine():
            raise ValueError(
                "operator_matrix and intercept are defined only when no "
                "index exceeds first order, where the surrogate is "
                "affine in its input coefficients. This index set "
                "contains higher-order terms, so the surrogate is "
                "nonlinear and no matrix represents it."
            )
        bkd = self.bkd()
        return bkd.sum(self._surrogate.indices(), axis=0)

    def operator_matrix(self) -> Array:
        r"""Return the first-order coefficients as an operator matrix.

        For an affine surrogate each first-order basis function is
        degree one in a single coordinate, so for a Legendre basis
        :math:`\Phi(\hat f) = \sqrt{3}\,\hat f` on those terms and the
        surrogate reduces to

        .. math:: \hat g = \sqrt{3}\, A^T \hat f + b

        with :math:`A` the rows returned here and :math:`b` the
        constant from :meth:`intercept`. Row :math:`j` is the response
        to input coefficient :math:`j`, so an operator diagonal in the
        chosen bases — an inverse Laplacian in a sine basis, say —
        appears as a diagonal matrix of its eigenvalues.

        The constant term is excluded rather than returned as a row,
        since a row of :math:`A` and an intercept mean different things
        and conflating them would misreport the operator.

        Returns
        -------
        Array
            First-order coefficients. Shape: (nfirstorder, noutputs)

        Raises
        ------
        ValueError
            If any index exceeds first order.
        """
        # Affine means degrees are 0 or 1, so >= 1 selects the
        # first-order rows; __eq__ is Union-typed and cannot narrow.
        degrees = self._require_affine()
        return self._params[degrees >= 1]

    def intercept(self) -> Optional[Array]:
        r"""Return the constant term :math:`b`, if the index set has one.

        Returns
        -------
        Array or None
            The constant coefficients, shape (noutputs,), or None when
            the index set omits the zero index and the surrogate is
            strictly linear.

        Raises
        ------
        ValueError
            If any index exceeds first order.
        """
        degrees = self._require_affine()
        bkd = self.bkd()
        constant = self._params[degrees <= 0]
        if constant.shape[0] == 0:
            return None
        return bkd.flatten(constant)


class OptimizerStageResult:
    """How one stage of an iterative fit ended.

    Backend-agnostic on purpose: an iteration count, an objective value
    and a budget describe any iterative optimizer, whether it is driven
    by torch, scipy or an alternating least-squares sweep. Only the code
    that *reads* these numbers off a particular optimizer is specific to
    it, and that belongs with the fitter.

    Parameters
    ----------
    name : str
        The optimizer's name, for the message.
    loss : float
        The objective after this stage.
    niterations : int
        Iterations actually taken.
    maxiterations : int
        The budget it was given.
    """

    def __init__(
        self,
        name: str,
        loss: float,
        niterations: int,
        maxiterations: int,
    ) -> None:
        self._name = name
        self._loss = loss
        self._niterations = niterations
        self._maxiterations = maxiterations

    def name(self) -> str:
        """Return the optimizer's name."""
        return self._name

    def fun(self) -> float:
        """Return the objective value after this stage."""
        return self._loss

    def niterations(self) -> int:
        """Return the iterations taken."""
        return self._niterations

    def maxiterations(self) -> int:
        """Return the iteration budget."""
        return self._maxiterations

    def exhausted_budget(self) -> bool:
        """Whether the stage stopped because it ran out of iterations.

        The distinction that decides what to do next, and the reason
        this class exists. A stage that converged has nothing more to
        give, so a large remaining error points at the model or the
        data; one that used its whole budget points at the budget, and
        raising it is the cheap thing to try first. Without this a
        caller cannot tell those apart and is left tuning blind.

        A method with no convergence test reports True whenever it runs
        to its limit, which is honest: it stopped because it was told
        to, not because it was finished.
        """
        return self._niterations >= self._maxiterations

    def message(self) -> str:
        """Return a human-readable termination reason."""
        if self.exhausted_budget():
            return (
                f"{self._name} used its full budget of "
                f"{self._maxiterations} iterations, so the fit may "
                f"improve with more"
            )
        return (
            f"{self._name} converged after {self._niterations} of "
            f"{self._maxiterations} iterations"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._name}, "
            f"fun={self._loss:.3e}, niterations={self._niterations})"
        )


class IterativeOperatorFitResult(Generic[Array]):
    """A fitted surrogate and the record of how an iterative fit reached it.

    The counterpart to :class:`OperatorFitResult` for fits that descend
    rather than solve. That one carries ``params``, which is meaningful
    because a closed-form solve produces one coefficient array; this one
    carries per-stage termination records instead, because what an
    iterative fit leaves a caller needing to know is whether it
    finished.

    Mirrors the ``fun``/``success``/``message`` shape of
    :class:`~pyapprox.optimization.minimize.result_protocol.OptimizerResultProtocol`.
    ``optima`` is deliberately absent: it returns a solution as one
    array, and the parameters here may be many arrays of different
    shapes. The fitted map on the surrogate is the answer to that
    question.

    Parameters
    ----------
    surrogate : OperatorSurrogate[Array]
        The fitted surrogate.
    stages : sequence of OptimizerStageResult
        One per stage, in the order they ran.
    """

    def __init__(
        self,
        surrogate: OperatorSurrogate[Array],
        stages: Sequence[OptimizerStageResult],
    ) -> None:
        if not stages:
            raise ValueError("stages must not be empty")
        self._surrogate = surrogate
        self._stages = list(stages)

    def surrogate(self) -> OperatorSurrogate[Array]:
        """Return the fitted surrogate."""
        return self._surrogate

    def stages(self) -> List[OptimizerStageResult]:
        """Return the per-stage records, in order."""
        return list(self._stages)

    def fun(self) -> float:
        """Return the objective value at the end of the fit."""
        return self._stages[-1].fun()

    def success(self) -> bool:
        """Whether the final stage converged rather than ran out.

        Read off the last stage because that is what settled the answer.
        An earlier stage exhausting its budget is ordinary -- a
        first-order stage is usually there to get close, not to finish.
        """
        return not self._stages[-1].exhausted_budget()

    def message(self) -> str:
        """Return a termination reason covering every stage."""
        return "; ".join(stage.message() for stage in self._stages)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fun={self.fun():.3e}, "
            f"success={self.success()})"
        )


class WeightedLeastSquaresOperatorFitter(Generic[Array]):
    r"""Fit an operator surrogate by weighted least squares.

    Solves

    .. math:: \min_C \sum_i w_i \|C^T \Phi(\hat f^i) - \hat g^i\|_2^2

    as a single least squares problem with :math:`d_{\mathrm{out}}`
    right-hand sides, at cost :math:`O(MN^2 + MNd_{\mathrm{out}})`.

    Weights come from the sampler that drew the inputs, so they are
    passed to a fit call rather than fixed at construction: they are a
    property of the sample, not of the estimator. Pass :math:`w`
    itself, not :math:`\sqrt{w}` — the solver applies the square root.

    Parameters
    ----------
    input_encoder : FunctionEncoderProtocol[Array]
        Maps input fields to the coefficients the expansion takes as
        its variables. The weaker protocol: the isometry matters only
        on the output side, and this encoder's answer is never read.
    output_encoder : FieldEncoderProtocol[Array]
        Maps output fields to the coefficients it predicts. Must be an
        isometry unless ``allow_proxy``, since the residual this
        minimizes is a field error only then.
    bkd : Backend[Array]
        Computational backend.
    solver : LinearSystemSolverProtocol[Array], optional
        The linear solver. Defaults to unregularized least squares.
        Must satisfy ``WeightedSolverProtocol`` if weights are used.
    allow_proxy : bool
        Fit against a non-isometric output encoder anyway, accepting
        the coefficient residual as an approximation of the field
        error. Often reasonable -- over a polynomial manifold it is the
        cheap fit, and usually a good one -- but it is no longer a
        guarantee, so it is opted into rather than assumed.
    """

    def __init__(
        self,
        input_encoder: FunctionEncoderProtocol[Array],
        output_encoder: FieldEncoderProtocol[Array],
        bkd: Backend[Array],
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
        allow_proxy: bool = False,
    ) -> None:
        if not isinstance(input_encoder, FunctionEncoderProtocol):
            raise TypeError(
                f"input_encoder must satisfy FunctionEncoderProtocol, "
                f"got {type(input_encoder).__name__}"
            )
        if not isinstance(output_encoder, FieldEncoderProtocol):
            raise TypeError(
                f"output_encoder must satisfy FieldEncoderProtocol, got "
                f"{type(output_encoder).__name__}"
            )
        self._input_encoder = input_encoder
        self._output_encoder = output_encoder
        self._allow_proxy = allow_proxy
        self._bkd = bkd
        self._solver = (
            LeastSquaresSolver(bkd) if solver is None else solver
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def solver(self) -> LinearSystemSolverProtocol[Array]:
        """Return the linear solver."""
        return self._solver

    def fit(
        self,
        latent_map: LinearInParamsLatentMapProtocol[Array],
        input_fields: Array,
        output_fields: Array,
        weights: Optional[Array] = None,
    ) -> OperatorFitResult[Array]:
        """Fit from realizations of the input and output fields.

        Encodes both, then delegates to :meth:`fit_encoded`.

        Parameters
        ----------
        latent_map : LinearInParamsLatentMapProtocol[Array]
            The map to fit, carrying the basis it is linear in.
        input_fields : Array
            Realizations of the input field on its grid.
            Shape: (ngrid_in, nsamples)
        output_fields : Array
            The corresponding output realizations.
            Shape: (ngrid_out, nsamples)
        weights : Array, optional
            Least-squares weights from the sampler that drew the
            inputs. Shape: (nsamples,)

        Returns
        -------
        OperatorFitResult
            The fitted surrogate and its coefficients.
        """
        return self.fit_encoded(
            latent_map,
            self._input_encoder.encode(input_fields),
            self._output_encoder.encode(output_fields),
            weights=weights,
        )

    def fit_encoded(
        self,
        latent_map: LinearInParamsLatentMapProtocol[Array],
        coefs_in: Array,
        coefs_out: Array,
        weights: Optional[Array] = None,
    ) -> OperatorFitResult[Array]:
        r"""Fit from coefficients that are already encoded.

        The primary entry point. Adaptive refinement refits repeatedly
        on one dataset, so re-encoding per iteration would be wasted
        work and would put the encoder inside a loop that is otherwise
        pure linear algebra.

        Parameters
        ----------
        latent_map : LinearInParamsLatentMapProtocol[Array]
            The map to fit, carrying the basis it is linear in.
        coefs_in : Array
            Encoded input realizations, one column per realization.
            Shape: (ncodes_in, nsamples)
        coefs_out : Array
            Encoded output realizations. Shape: (ncodes_out, nsamples)
        weights : Array, optional
            Least-squares weights :math:`w_i`, as returned by a sampler
            alongside its samples. Shape: (nsamples,). None means unit
            weights, correct only when the realizations were drawn from
            the reference measure itself — never a request for optimal
            weights.

        Returns
        -------
        OperatorFitResult
            The fitted surrogate and its coefficients.
        """
        require_coefficient_error_is_field_error(
            self._output_encoder,
            f"{type(self).__name__} minimizes a coefficient residual, "
            f"which",
            self._allow_proxy,
        )
        if not is_linear_in_params(latent_map):
            raise TypeError(
                f"{type(self).__name__} solves one linear least squares "
                f"problem, which needs a latent map linear in its "
                f"parameters -- satisfying "
                f"LinearInParamsLatentMapProtocol, so that basis_matrix "
                f"and with_params exist. {type(latent_map).__name__} is "
                f"not, so fit it with a gradient-based fitter instead."
            )
        if coefs_out.ndim != 2:
            raise ValueError(
                f"coefs_out must be 2D with shape (ncodes_out, nsamples), "
                f"got {coefs_out.shape}"
            )
        nsamples = coefs_in.shape[1]
        if coefs_out.shape[1] != nsamples:
            raise ValueError(
                f"coefs_out has {coefs_out.shape[1]} realizations but "
                f"coefs_in has {nsamples}"
            )
        if weights is not None and weights.shape != (nsamples,):
            raise ValueError(
                f"weights has wrong shape {weights.shape}, expected "
                f"({nsamples},)"
            )

        if weights is not None:
            if not isinstance(self._solver, WeightedSolverProtocol):
                raise TypeError(
                    f"weights were supplied but "
                    f"{type(self._solver).__name__} does not satisfy "
                    f"WeightedSolverProtocol, so they would be silently "
                    f"ignored"
                )
            self._solver.set_weights(weights)

        basis_matrix = latent_map.basis_matrix(coefs_in)
        params = self._solver.solve(basis_matrix, coefs_out.T)
        surrogate = OperatorSurrogate(
            self._input_encoder,
            self._output_encoder,
            latent_map.with_params(params),
            self._bkd,
        )
        return OperatorFitResult(surrogate, params)
