r"""Weighted least-squares fitting of an operator surrogate.

Once fields are encoded the fit is a vector-valued polynomial chaos
expansion: the input coefficients are its samples, the output
coefficients its quantities of interest, and :math:`C` its coefficient
array of shape ``(nterms, nqoi)``. So the fitter holds an expansion
rather than reimplementing one, and takes the same
``fit(expansion, samples, values)`` shape as its siblings in
``affine/expansions/fitters``.

The design matrix reaches the solver intact. ``LinearSystemSolver``
applies :math:`\sqrt{w}` to both it and the right-hand side and then
solves once with :math:`d_{\mathrm{out}}` columns, so the weighting is
a row scaling of a single least squares problem rather than something
folded into normal equations.
"""

from __future__ import annotations

from typing import Generic, Optional

from pyapprox.optimization.linear import LeastSquaresSolver
from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.protocols.solver import (
    LinearSystemSolverProtocol,
    WeightedSolverProtocol,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
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
        """Return the per-index total degrees of an affine surrogate."""
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
    input_encoder : FieldEncoderProtocol[Array]
        Maps input fields to the coefficients the expansion takes as
        its variables.
    output_encoder : FieldEncoderProtocol[Array]
        Maps output fields to the coefficients it predicts. Must be an
        isometry.
    bkd : Backend[Array]
        Computational backend.
    solver : LinearSystemSolverProtocol[Array], optional
        The linear solver. Defaults to unregularized least squares.
        Must satisfy ``WeightedSolverProtocol`` if weights are used.
    """

    def __init__(
        self,
        input_encoder: FieldEncoderProtocol[Array],
        output_encoder: FieldEncoderProtocol[Array],
        bkd: Backend[Array],
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
    ) -> None:
        for name, encoder in (
            ("input_encoder", input_encoder),
            ("output_encoder", output_encoder),
        ):
            if not isinstance(encoder, FieldEncoderProtocol):
                raise TypeError(
                    f"{name} must satisfy FieldEncoderProtocol, got "
                    f"{type(encoder).__name__}"
                )
        self._input_encoder = input_encoder
        self._output_encoder = output_encoder
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
        expansion: PolynomialChaosExpansion[Array],
        input_fields: Array,
        output_fields: Array,
        weights: Optional[Array] = None,
    ) -> OperatorFitResult[Array]:
        """Fit from realizations of the input and output fields.

        Encodes both, then delegates to :meth:`fit_encoded`.

        Parameters
        ----------
        expansion : PolynomialChaosExpansion[Array]
            The expansion to fit, carrying the basis and index set.
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
            expansion,
            self._input_encoder.encode(input_fields),
            self._output_encoder.encode(output_fields),
            weights=weights,
        )

    def fit_encoded(
        self,
        expansion: PolynomialChaosExpansion[Array],
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
        expansion : PolynomialChaosExpansion[Array]
            The expansion to fit, carrying the basis and index set.
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

        basis_matrix = expansion.basis_matrix(coefs_in)
        params = self._solver.solve(basis_matrix, coefs_out.T)
        surrogate = OperatorSurrogate(
            self._input_encoder,
            self._output_encoder,
            expansion.with_params(params),
            self._bkd,
        )
        return OperatorFitResult(surrogate, params)
