"""Tensor product subspace for sparse grids.

A subspace represents a single tensor product of univariate interpolations,
identified by a multi-index specifying the level in each dimension.

This module wraps TensorProductInterpolant with sparse-grid-specific
functionality: multi-index tracking, growth rules, and quadrature.
"""

from typing import Generic, List, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.cartesian import outer_product_weights
from pyapprox.typing.surrogates.affine.protocols import (
    Basis1DProtocol,
    IndexGrowthRuleProtocol,
)
from pyapprox.typing.surrogates.affine.univariate.lagrange import LagrangeBasis1D
from pyapprox.typing.surrogates.tensorproduct import TensorProductInterpolant
from pyapprox.typing.surrogates.sparsegrids.basis_setup import (
    compute_npts_from_growth_rule,
    get_quadrature_rule,
    create_lagrange_from_quadrature,
)


class TensorProductSubspace(Generic[Array]):
    """Single tensor product subspace in sparse grid.

    Wraps TensorProductInterpolant with sparse-grid-specific functionality:
    - Multi-index tracking (level in each dimension)
    - Growth rule (level -> number of points)
    - Quadrature weights and integration

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    index : Array
        Multi-index identifying this subspace, shape (nvars,)
    univariate_bases : List[Basis1DProtocol[Array]]
        Univariate bases for each dimension.
    growth_rule : IndexGrowthRuleProtocol
        Rule mapping level to number of points.

    Notes
    -----
    Values have shape (nqoi, nsamples) following CLAUDE.md conventions:
    - Output f(X) (batch): (nqoi, nsamples) - QoIs as rows, samples as columns

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule
    >>> bkd = NumpyBkd()
    >>> basis = LegendrePolynomial1D(bkd)
    >>> growth = LinearGrowthRule()
    >>> index = bkd.asarray([1, 2])
    >>> subspace = TensorProductSubspace(bkd, index, [basis, basis], growth)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        index: Array,
        univariate_bases: List[Basis1DProtocol[Array]],
        growth_rule: IndexGrowthRuleProtocol,
    ):
        self._bkd = bkd
        self._index = bkd.copy(index)
        self._univariate_bases = univariate_bases
        self._growth_rule = growth_rule

        # Compute number of points per dimension from growth rule
        self._npts_1d = compute_npts_from_growth_rule(index, growth_rule)

        # Create independent LagrangeBasis1D for each dimension
        # IMPORTANT: Each dimension needs its own basis instance because
        # different dimensions may have different numbers of points.
        # Even if the same basis object is passed for multiple dimensions,
        # we create separate LagrangeBasis1D wrappers for each.
        self._interp_bases_1d: List[LagrangeBasis1D[Array]] = []
        self._1d_weights: List[Array] = []

        for dim, basis in enumerate(univariate_bases):
            npts = self._npts_1d[dim]

            # Extract quadrature rule from the basis
            quad_rule = get_quadrature_rule(basis)

            # Create a new LagrangeBasis1D for this dimension
            # This ensures each dimension has independent nterms
            interp_basis: LagrangeBasis1D[Array] = create_lagrange_from_quadrature(
                bkd, quad_rule
            )
            _, weights_1d = quad_rule(npts)

            self._interp_bases_1d.append(interp_basis)
            self._1d_weights.append(weights_1d)

        # Create TensorProductInterpolant using Lagrange bases
        self._interpolant = TensorProductInterpolant(
            bkd, self._interp_bases_1d, self._npts_1d
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def get_index(self) -> Array:
        """Return the multi-index identifying this subspace."""
        return self._bkd.copy(self._index)

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._interpolant.nvars()

    def nsamples(self) -> int:
        """Return the number of samples in this subspace."""
        return self._interpolant.nsamples()

    def nqoi(self) -> int:
        """Return the number of quantities of interest, or 0 if not set."""
        return self._interpolant.nqoi()

    def get_samples(self) -> Array:
        """Return sample locations for this subspace."""
        return self._interpolant.get_samples()

    def get_samples_1d(self, dim: int) -> Array:
        """Return 1D interpolation nodes for a specific dimension.

        Parameters
        ----------
        dim : int
            Dimension index.

        Returns
        -------
        Array
            1D sample locations with shape (1, npts_1d[dim]).
        """
        return self._interpolant.get_samples_1d(dim)

    def get_values(self) -> Optional[Array]:
        """Return function values at samples, if set.

        Returns
        -------
        Optional[Array]
            Values with shape (nqoi, nsamples), or None if not set.
        """
        return self._interpolant.get_values()

    def set_values(self, values: Array) -> None:
        """Set function values at samples.

        Parameters
        ----------
        values : Array
            Values with shape (nqoi, nsamples).
        """
        self._interpolant.set_values(values)

    def __call__(self, samples: Array) -> Array:
        """Evaluate subspace interpolant at given samples.

        Parameters
        ----------
        samples : Array
            Evaluation points of shape (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values of shape (nqoi, npoints)
        """
        return self._interpolant(samples)

    def jacobian_supported(self) -> bool:
        """Return whether Jacobian computation is supported."""
        return self._interpolant.jacobian_supported()

    def hessian_supported(self) -> bool:
        """Return whether Hessian computation is supported."""
        return self._interpolant.hessian_supported()

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample point.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)

        Returns
        -------
        Array
            Jacobian matrix of shape (nqoi, nvars)
        """
        return self._interpolant.jacobian(sample)

    def hessian(self, sample: Array) -> Array:
        """Compute Hessian at a single sample point for scalar QoI.

        Only valid when nqoi == 1.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)

        Returns
        -------
        Array
            Hessian matrix of shape (nvars, nvars)
        """
        return self._interpolant.hessian(sample)

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product efficiently.

        Only valid when nqoi == 1.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)
        vec : Array
            Direction vector of shape (nvars, 1)

        Returns
        -------
        Array
            Hessian-vector product of shape (nvars, 1)
        """
        return self._interpolant.hvp(sample, vec)

    def whvp(
        self,
        sample: Array,
        vec: Array,
        weights: Array,
    ) -> Array:
        """Compute weighted Hessian-vector product efficiently.

        Computes sum_q weights[q] * H_q @ v where H_q is the Hessian for QoI q.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)
        vec : Array
            Direction vector of shape (nvars, 1)
        weights : Array
            Weights for each QoI. Shape: (nqoi, 1), (1, nqoi), or (nqoi,).

        Returns
        -------
        Array
            Weighted Hessian-vector product of shape (nvars, 1)
        """
        return self._interpolant.whvp(sample, vec, weights)

    def get_quadrature_weights(self) -> Array:
        """Return the tensor product quadrature weights.

        The weights are the raw tensor product of 1D quadrature weights
        without normalization. For Gauss-Legendre on [-1,1]^d, the sum
        of weights equals 2^d (Lebesgue measure).

        Returns
        -------
        Array
            Tensor product weights of shape (nsamples,)
        """
        return outer_product_weights(self._1d_weights, self._bkd)

    def integrate(self) -> Array:
        """Compute integral using tensor product quadrature.

        Computes the weighted sum:
            integral f(x) w(x) dx ≈ sum_i weights[i] * f(x_i)

        where w(x) is the weight function associated with the polynomial
        basis. For orthonormal polynomials with probability=True, the
        weights sum to 1 and this directly computes the expectation E[f].

        Returns
        -------
        Array
            Integral values of shape (nqoi,)

        Notes
        -----
        For exactly interpolated functions (polynomials up to the
        quadrature degree), this gives the exact integral.
        """
        values = self._interpolant.get_values()
        if values is None:
            raise ValueError("Values not set. Call set_values() first.")

        weights = self.get_quadrature_weights()
        # values is (nqoi, nsamples), weights is (nsamples,)
        # Result should be (nqoi,)
        return values @ weights

    def __repr__(self) -> str:
        index_str = ",".join(str(int(i)) for i in self._index)
        return (
            f"TensorProductSubspace(index=[{index_str}], "
            f"nsamples={self.nsamples()})"
        )
