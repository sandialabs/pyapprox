"""Gradient-enhanced PCE fitter using constrained least squares."""

from typing import Generic

from pyapprox.optimization.linear.least_squares import (
    LinearlyConstrainedLstSqSolver,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class GradientEnhancedPCEFitter(Generic[Array]):
    """Fit PCE using gradient matching with exact function interpolation.

    This fitter solves:
        min_theta ||G - Phi_G theta||_2^2
        subject to: Phi theta = y

    where:
        - Phi: basis matrix for function values (nsamples, nterms)
        - Phi_G: gradient basis matrix (nsamples * nvars, nterms)
        - y: function values (nsamples,)
        - G: gradient values (nsamples * nvars,)

    The constraint ensures exact interpolation of function values.
    The objective minimizes gradient mismatch.

    Requires an expansion with:
        - basis_matrix() method returning (nsamples, nterms)
        - jacobian_batch() method returning (nsamples, nqoi, nvars) derivatives
        - with_params() method for immutable pattern

    Only supports nqoi=1 (single quantity of interest).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
        gradients: Array,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit using gradient matching with function value constraints.

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix(), jacobian_batch(), and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Function values. Shape: (1, nsamples) or (nsamples,).
            Only nqoi=1 supported.
        gradients : Array
            Gradient values. Shape: (nvars, nsamples).

        Returns
        -------
        DirectSolverResult
            Result containing fitted expansion.

        Raises
        ------
        ValueError
            If nqoi > 1, or if total rows (value + gradient) < nterms.
        TypeError
            If expansion lacks required jacobian_batch method.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        # Validate single QoI
        if values.shape[0] != 1:
            raise ValueError(
                f"GradientEnhancedPCEFitter only supports nqoi=1, got {values.shape[0]}"
            )

        # Check expansion has jacobian_batch
        if not hasattr(expansion, "jacobian_batch"):
            raise TypeError(
                f"Expansion must have jacobian_batch method, "
                f"got {type(expansion).__name__}"
            )

        nvars, nsamples = samples.shape
        nterms = expansion.nterms()

        # DGGLSE-style validation: n <= m + p
        # n = nterms, m = nsamples * nvars (gradient rows), p = nsamples (value rows)
        # Need enough total rows to determine all coefficients.
        total_rows = nsamples + nsamples * nvars
        if nterms > total_rows:
            raise ValueError(
                f"Underdetermined: {nterms} terms but only "
                f"{nsamples} value + {nsamples * nvars} gradient = "
                f"{total_rows} total rows"
            )

        # Build basis matrix for function values: Phi (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)

        # Build gradient basis matrix: Phi_G (nsamples * nvars, nterms)
        # expansion.jacobian_batch returns (nsamples, nqoi, nvars) for each basis term
        # We need the derivative of each basis function w.r.t. each input variable
        # This is obtained from the underlying basis's jacobian_batch
        basis = expansion._basis  # Access underlying basis for gradient computation

        # basis.jacobian_batch returns (nsamples, nterms, nvars)
        basis_jac = basis.jacobian_batch(samples)  # (nsamples, nterms, nvars)

        # Reshape to (nsamples * nvars, nterms)
        # For each sample i and variable j, row i*nvars + j contains d(phi_k)/dx_j at
        # sample i
        Phi_G = bkd.reshape(
            bkd.transpose(basis_jac, (0, 2, 1)),  # (nsamples, nvars, nterms)
            (nsamples * nvars, nterms),
        )

        # Flatten gradient values to match: (nsamples * nvars,)
        # gradients shape: (nvars, nsamples) -> transpose and flatten
        G = bkd.reshape(gradients.T, (-1,))  # (nsamples * nvars,)
        G = bkd.reshape(G, (-1, 1))  # (nsamples * nvars, 1)

        # Function values as constraint: y (nsamples, 1)
        y = values.T  # (nsamples, 1)

        # Solve constrained least squares:
        # min ||Phi_G theta - G||^2 s.t. Phi theta = y
        solver = LinearlyConstrainedLstSqSolver(
            bkd,
            constraint_matrix=Phi,
            constraint_vector=y,
        )
        params = solver.solve(Phi_G, G)  # (nterms, 1)

        # Create fitted expansion (immutable pattern)
        fitted_expansion = expansion.with_params(params)

        return DirectSolverResult(
            surrogate=fitted_expansion,
            params=params,
        )
