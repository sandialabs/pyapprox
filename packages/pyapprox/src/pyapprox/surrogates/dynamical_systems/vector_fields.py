"""Parametric vector field implementations for dynamical systems learning.

Provides BasisExpansionVectorField (wraps BasisExpansion) and
AdditiveVectorField (composes multiple vector fields).
"""

from typing import Generic, List, Sequence

from pyapprox.surrogates.affine.expansions.base import BasisExpansion
from pyapprox.surrogates.affine.protocols import BasisHasJacobianProtocol
from pyapprox.surrogates.dynamical_systems.protocols import (
    ParametricVectorFieldProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameter, HyperParameterList


class BasisExpansionVectorField(Generic[Array]):
    """Vector field backed by a BasisExpansion.

    Wraps a BasisExpansion with nqoi == nstates to model
    F_eta(x) = expansion(x), where eta are the expansion coefficients.

    Parameters
    ----------
    expansion : BasisExpansion[Array]
        Expansion with nvars() == nqoi() (square system).
    """

    def __init__(self, expansion: BasisExpansion[Array]):
        if expansion.nvars() != expansion.nqoi():
            raise ValueError(
                f"BasisExpansion must have nvars == nqoi for a vector field, "
                f"got nvars={expansion.nvars()}, nqoi={expansion.nqoi()}"
            )
        self._expansion = expansion
        self._bkd = expansion.bkd()
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        if isinstance(self._expansion.get_basis(), BasisHasJacobianProtocol):
            self.state_jacobian = self._state_jacobian
        self.param_jacobian = self._param_jacobian

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return self._expansion.nvars()

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._expansion.hyp_list()

    def expansion(self) -> BasisExpansion[Array]:
        return self._expansion

    def with_params(self, params: Array) -> "BasisExpansionVectorField[Array]":
        """Return NEW instance with coefficients set. Original unchanged.

        Parameters
        ----------
        params : Array
            Coefficient values. Shape: (nterms, nqoi)

        Returns
        -------
        BasisExpansionVectorField[Array]
            New vector field with coefficients set.
        """
        new_expansion = self._expansion.with_params(params)
        return BasisExpansionVectorField(new_expansion)

    def __call__(self, states: Array) -> Array:
        """Evaluate F_eta(x).

        Parameters
        ----------
        states : Array
            Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Shape: (nstates, nsamples)
        """
        self._expansion.sync_params()
        return self._expansion(states)

    def _state_jacobian(self, states: Array) -> Array:
        """Compute dF/dx at each sample.

        Parameters
        ----------
        states : Array
            Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, nstates, nstates)
        """
        self._expansion.sync_params()
        return self._expansion.jacobian_batch(states)

    def _param_jacobian(self, states: Array) -> Array:
        """Compute dF/d_eta (active params only) at each sample.

        Parameters
        ----------
        states : Array
            Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, nstates, nactive_params)
        """
        return self._expansion.jacobian_wrt_params(states)


class AdditiveVectorField(Generic[Array]):
    """Additive composition of vector fields: F(x) = F1(x) + F2(x) + ...

    The combined parameter vector is the concatenation of all component
    parameter vectors. Derivative methods are bound only if all
    components support them.

    Parameters
    ----------
    components : sequence of ParametricVectorFieldProtocol
        Each must have the same nstates().
    """

    def __init__(
        self,
        components: Sequence[ParametricVectorFieldProtocol[Array]],
    ):
        if not components:
            raise ValueError("At least one component required")
        nstates = components[0].nstates()
        for i, comp in enumerate(components):
            if comp.nstates() != nstates:
                raise ValueError(
                    f"Component {i} has nstates={comp.nstates()}, "
                    f"expected {nstates}"
                )
        self._components = components
        self._bkd = components[0].bkd()
        self._nstates = nstates
        self._hyp_list = self._build_hyp_list()
        self._param_offsets = self._compute_offsets()
        self._setup_derivative_methods()

    def _build_hyp_list(self) -> HyperParameterList[Array]:
        all_hyps: List[HyperParameter[Array]] = []
        for comp in self._components:
            all_hyps.extend(comp.hyp_list().hyperparameters())
        return HyperParameterList(all_hyps, self._bkd)

    def _compute_offsets(self) -> List[int]:
        offsets = [0]
        for comp in self._components:
            offsets.append(offsets[-1] + comp.hyp_list().nactive_params())
        return offsets

    def _setup_derivative_methods(self) -> None:
        if all(hasattr(c, "state_jacobian") for c in self._components):
            self.state_jacobian = self._state_jacobian
        if all(hasattr(c, "param_jacobian") for c in self._components):
            self.param_jacobian = self._param_jacobian

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return self._nstates

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def __call__(self, states: Array) -> Array:
        """Evaluate F(x) = sum_i F_i(x).

        Parameters
        ----------
        states : Array
            Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Shape: (nstates, nsamples)
        """
        result = self._components[0](states)
        for comp in self._components[1:]:
            result = result + comp(states)
        return result

    def _state_jacobian(self, states: Array) -> Array:
        """Compute dF/dx = sum_i dF_i/dx.

        Parameters
        ----------
        states : Array
            Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, nstates, nstates)
        """
        result = self._components[0].state_jacobian(states)
        for comp in self._components[1:]:
            result = result + comp.state_jacobian(states)
        return result

    def _param_jacobian(self, states: Array) -> Array:
        """Compute dF/d_eta with block-concatenated active parameters.

        Parameters
        ----------
        states : Array
            Shape: (nstates, nsamples)

        Returns
        -------
        Array
            Shape: (nsamples, nstates, total_nactive_params)
        """
        jacs = [comp.param_jacobian(states) for comp in self._components]
        return self._bkd.concatenate(jacs, axis=2)
