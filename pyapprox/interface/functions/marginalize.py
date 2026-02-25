"""Dimension reduction for multivariate functions.

Provides tools for reducing the dimensionality of multivariate functions
via integration (marginalization) or cross-sections (fixing variables at
nominal values).  Works with any FunctionProtocol.

Two concrete reducers are provided:

* ``FunctionMarginalizer`` – integrates out variables using a
  user-supplied quadrature strategy (``QuadratureFactoryProtocol``).
* ``CrossSectionReducer`` – fixes non-kept variables at user-defined
  nominal values (no quadrature required).

Both satisfy ``DimensionReducerProtocol``, so higher-level tools such as
``PairPlotter`` can accept either interchangeably.
"""

from typing import Callable, Generic, List, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.quadrature.protocols import (
    MultivariateQuadratureRuleProtocol,
)


# ------------------------------------------------------------------ #
# Protocols                                                          #
# ------------------------------------------------------------------ #

@runtime_checkable
class QuadratureFactoryProtocol(Protocol, Generic[Array]):
    """Factory that builds quadrature rules for integrating out variables.

    Given a list of variable indices to integrate out, returns a
    MultivariateQuadratureRuleProtocol with samples on the correct
    domain and appropriate weights. The measure (Lebesgue, probability,
    etc.) depends on the factory implementation.

    This protocol enables extensibility: users can implement factories
    for tensor product quadrature, Monte Carlo, Sobol sequences, sparse
    grids, or any other integration strategy.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def __call__(
        self, integrate_indices: List[int]
    ) -> MultivariateQuadratureRuleProtocol[Array]:
        """Build a quadrature rule for the specified variables.

        Parameters
        ----------
        integrate_indices : List[int]
            Indices of variables to integrate out.

        Returns
        -------
        MultivariateQuadratureRuleProtocol[Array]
            Quadrature rule with samples on the domain of the
            specified variables.
        """
        ...


@runtime_checkable
class DimensionReducerProtocol(Protocol, Generic[Array]):
    """Reduces a d-dimensional function to a lower-dimensional one.

    Implementations include integration (marginalization) and
    cross-sections (fixing variables at nominal values).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables in the original function."""
        ...

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        ...

    def reduce(
        self, keep_indices: List[int]
    ) -> "ReducedFunction[Array]":
        """Reduce to the specified variables.

        Parameters
        ----------
        keep_indices : List[int]
            Indices of variables to keep (0-based).

        Returns
        -------
        ReducedFunction[Array]
            A function satisfying FunctionProtocol with
            ``nvars = len(keep_indices)`` and the same ``nqoi``.
        """
        ...


# ------------------------------------------------------------------ #
# Reduced function wrapper                                           #
# ------------------------------------------------------------------ #

class ReducedFunction(Generic[Array]):
    """Function produced by dimension reduction.

    Satisfies FunctionProtocol: has ``bkd()``, ``nvars()``, ``nqoi()``,
    ``__call__()``.

    Parameters
    ----------
    nvars : int
        Number of remaining (kept) variables.
    nqoi : int
        Number of quantities of interest.
    eval_fn : Callable[[Array], Array]
        Function mapping ``(nvars, nsamples) -> (nqoi, nsamples)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        nvars: int,
        nqoi: int,
        eval_fn: Callable[[Array], Array],
        bkd: Backend[Array],
    ):
        self._nvars = nvars
        self._nqoi = nqoi
        self._eval_fn = eval_fn
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        """Evaluate the reduced function.

        Parameters
        ----------
        samples : Array
            Input samples of shape ``(nvars, nsamples)``.

        Returns
        -------
        Array
            Output values of shape ``(nqoi, nsamples)``.
        """
        return self._eval_fn(samples)


# Backward-compatible alias
MarginalizedFunction = ReducedFunction


# ------------------------------------------------------------------ #
# Concrete reducers                                                  #
# ------------------------------------------------------------------ #

class FunctionMarginalizer(Generic[Array]):
    """Integrates out variables from any function using quadrature.

    Given a d-dimensional function and a quadrature factory, produces
    lower-dimensional ReducedFunction objects by integrating out
    specified variables.

    Satisfies ``DimensionReducerProtocol``.

    Parameters
    ----------
    function : object
        A d-dimensional function satisfying FunctionProtocol
        (must have ``bkd()``, ``nvars()``, ``nqoi()``, ``__call__()``).
    quad_factory : QuadratureFactoryProtocol[Array]
        Factory that builds quadrature rules for any subset of variables.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        function: object,
        quad_factory: QuadratureFactoryProtocol[Array],
        bkd: Backend[Array],
    ):
        self._function = function
        self._quad_factory = quad_factory
        self._bkd = bkd
        self._nvars = function.nvars()
        self._nqoi = function.nqoi()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables in the original function."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._nqoi

    def reduce(
        self, keep_indices: List[int]
    ) -> ReducedFunction[Array]:
        """Integrate out all variables except *keep_indices*.

        Alias for :meth:`marginalize`.
        """
        return self.marginalize(keep_indices)

    def marginalize(
        self, keep_indices: List[int]
    ) -> ReducedFunction[Array]:
        """Integrate out all variables except keep_indices.

        Parameters
        ----------
        keep_indices : List[int]
            Indices of variables to keep (0-based). The returned
            function has nvars = len(keep_indices).

        Returns
        -------
        ReducedFunction[Array]
            A function satisfying FunctionProtocol with
            nvars = len(keep_indices) and the same nqoi.
        """
        integrate_indices = sorted(
            set(range(self._nvars)) - set(keep_indices)
        )

        if not integrate_indices:
            return ReducedFunction(
                self._nvars, self._nqoi, self._function.__call__, self._bkd
            )

        quad_rule = self._quad_factory(integrate_indices)
        quad_samples, quad_weights = quad_rule()
        # quad_samples: (n_integrate, nquad), quad_weights: (nquad,)

        bkd = self._bkd
        function = self._function
        nvars = self._nvars
        nqoi = self._nqoi

        def eval_fn(samples_keep: Array) -> Array:
            # samples_keep: (n_keep, nsamples)
            nsamples = samples_keep.shape[1]
            nquad = quad_samples.shape[1]

            # Repeat each keep-sample nquad times contiguously
            samples_rep = bkd.repeat(samples_keep, nquad, axis=1)
            # Tile quad points nsamples times
            quad_tiled = bkd.tile(quad_samples, (1, nsamples))

            # Assemble full d-dimensional sample array
            full = bkd.zeros((nvars, nsamples * nquad))
            for kk, idx in enumerate(keep_indices):
                full[idx] = samples_rep[kk]
            for kk, idx in enumerate(integrate_indices):
                full[idx] = quad_tiled[kk]

            # Evaluate function: (nqoi, nsamples * nquad)
            vals = function(full)

            # Reshape to (nqoi, nsamples, nquad), apply weights, sum
            vals_3d = bkd.reshape(vals, (nqoi, nsamples, nquad))
            result = bkd.sum(
                vals_3d * quad_weights[None, None, :], axis=2
            )
            return result  # (nqoi, nsamples)

        return ReducedFunction(
            len(keep_indices), nqoi, eval_fn, bkd
        )


class CrossSectionReducer(Generic[Array]):
    """Fixes non-kept variables at nominal values.

    Produces lower-dimensional ReducedFunction objects by substituting
    fixed nominal values for all variables not in *keep_indices*.

    Satisfies ``DimensionReducerProtocol``.

    Parameters
    ----------
    function : object
        A d-dimensional function satisfying FunctionProtocol.
    nominal_values : Array
        Shape ``(nvars,)`` giving the nominal value for each variable.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        function: object,
        nominal_values: Array,
        bkd: Backend[Array],
    ):
        self._function = function
        self._nominal_values = nominal_values
        self._bkd = bkd
        self._nvars = function.nvars()
        self._nqoi = function.nqoi()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables in the original function."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._nqoi

    def reduce(
        self, keep_indices: List[int]
    ) -> ReducedFunction[Array]:
        """Fix all variables except *keep_indices* at nominal values.

        Parameters
        ----------
        keep_indices : List[int]
            Indices of variables to keep (0-based).

        Returns
        -------
        ReducedFunction[Array]
            A function satisfying FunctionProtocol with
            ``nvars = len(keep_indices)`` and the same ``nqoi``.
        """
        fix_indices = sorted(
            set(range(self._nvars)) - set(keep_indices)
        )

        if not fix_indices:
            return ReducedFunction(
                self._nvars, self._nqoi, self._function.__call__, self._bkd
            )

        bkd = self._bkd
        function = self._function
        nvars = self._nvars
        nqoi = self._nqoi
        nominal = self._nominal_values

        def eval_fn(samples_keep: Array) -> Array:
            # samples_keep: (n_keep, nsamples)
            nsamples = samples_keep.shape[1]
            full = bkd.zeros((nvars, nsamples))
            for kk, idx in enumerate(keep_indices):
                full[idx] = samples_keep[kk]
            for idx in fix_indices:
                full[idx] = nominal[idx]
            return function(full)

        return ReducedFunction(
            len(keep_indices), nqoi, eval_fn, bkd
        )


class ActiveSetFunction(Generic[Array]):
    """Fix a subset of variables at nominal values, expose only kept variables.

    Unlike CrossSectionReducer which returns ReducedFunction (evaluation only),
    ActiveSetFunction dynamically propagates jacobian/hvp/whvp from the
    wrapped model, making it suitable for optimization.

    TODO: Consider consolidating with CrossSectionReducer by adding dynamic
    derivative binding to ReducedFunction or the reducer itself.

    Parameters
    ----------
    function : object
        A function satisfying FunctionProtocol. May also have
        jacobian(), hvp(), whvp() methods.
    nominal_values : Array
        Shape (nvars,). Nominal values for ALL variables.
    keep_indices : List[int]
        Indices of variables to keep (expose to the optimizer).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        function: object,
        nominal_values: Array,
        keep_indices: List[int],
        bkd: Backend[Array],
    ) -> None:
        self._function = function
        self._nominal_values = nominal_values
        self._keep_indices = keep_indices
        self._bkd = bkd
        self._nvars_full: int = function.nvars()  # type: ignore[union-attr]
        self._nqoi: int = function.nqoi()  # type: ignore[union-attr]

        # Dynamic binding of derivative methods
        if hasattr(function, "jacobian"):
            self.jacobian = self._jacobian
        if hasattr(function, "hvp"):
            self.hvp = self._hvp
        if hasattr(function, "whvp"):
            self.whvp = self._whvp

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return len(self._keep_indices)

    def nqoi(self) -> int:
        return self._nqoi

    def _assemble(self, samples: Array) -> Array:
        """Assemble full-dimensional samples from kept variables.

        Parameters
        ----------
        samples : Array
            Shape (n_keep, nsamples).

        Returns
        -------
        Array
            Shape (nvars_full, nsamples).
        """
        bkd = self._bkd
        nsamples = samples.shape[1]
        full = bkd.repeat(
            self._nominal_values[:, None], nsamples, axis=1
        )
        for kk, idx in enumerate(self._keep_indices):
            full[idx] = samples[kk]
        return full

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function at kept variables with fixed nominal values.

        Parameters
        ----------
        samples : Array
            Shape (n_keep, nsamples).

        Returns
        -------
        Array
            Shape (nqoi, nsamples).
        """
        full = self._assemble(samples)
        return self._function(full)  # type: ignore[operator]

    def _jacobian(self, sample: Array) -> Array:
        """Jacobian w.r.t. kept variables only.

        Parameters
        ----------
        sample : Array
            Shape (n_keep, 1).

        Returns
        -------
        Array
            Shape (nqoi, n_keep).
        """
        full = self._assemble(sample)
        jac_full = self._function.jacobian(full)  # type: ignore[union-attr]
        return jac_full[:, self._keep_indices]

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """Hessian-vector product w.r.t. kept variables only.

        Parameters
        ----------
        sample : Array
            Shape (n_keep, 1).
        vec : Array
            Shape (n_keep, 1).

        Returns
        -------
        Array
            Shape (n_keep, 1).
        """
        bkd = self._bkd
        full_sample = self._assemble(sample)
        full_vec = bkd.zeros((self._nvars_full, 1))
        for kk, idx in enumerate(self._keep_indices):
            full_vec[idx] = vec[kk]
        result_full = self._function.hvp(full_sample, full_vec)  # type: ignore[union-attr]
        return result_full[self._keep_indices, :]

    def _whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Weighted Hessian-vector product w.r.t. kept variables only.

        Parameters
        ----------
        sample : Array
            Shape (n_keep, 1).
        vec : Array
            Shape (n_keep, 1).
        weights : Array
            Shape (nqoi, 1).

        Returns
        -------
        Array
            Shape (n_keep, 1).
        """
        bkd = self._bkd
        full_sample = self._assemble(sample)
        full_vec = bkd.zeros((self._nvars_full, 1))
        for kk, idx in enumerate(self._keep_indices):
            full_vec[idx] = vec[kk]
        result_full = self._function.whvp(full_sample, full_vec, weights)  # type: ignore[union-attr]
        return result_full[self._keep_indices, :]
