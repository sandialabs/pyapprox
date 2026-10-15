r"""Assemble a Galerkin physics from its residual terms, once for all uses.

A physics states its interior residual as terms, skfem forms built for
the basis they will be assembled on:

.. math:: F(u) = \sum \text{loads} + \sum \text{state loads}(u)
                 - \Big(\sum \text{stiffness}\Big) u,

with the state loads' Jacobians alongside (:class:`GalerkinResidualTerms`).
Everything else lives here, written once: how terms combine into a
residual and a Jacobian, which assembled pieces may be cached and on
what key, and when the physics' validity check runs. Two assemblies
share that composition:

- :class:`SkfemTermAssembler`, over the whole mesh, summed into the
  global residual and (sparse) Jacobian the physics returns;
- :class:`SkfemRestrictedOperator`, over a subset of elements, unsummed,
  per element and from element-local states
  (``RestrictedElementOperatorProtocol``), which hyper-reduction and
  partitioned assembly consume.

So a physics supplies its forms and nothing else, and its global and
element-wise evaluations cannot drift apart. This module is the skfem
implementation of the element-wise protocols in
:mod:`pyapprox.pde.elementwise`; another assembly library implements
them without touching the physics' mathematics.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Hashable,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix

from pyapprox.pde.elementwise import ElementSet
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.optional_deps import import_optional_dependency

_Dense = NDArray[np.floating[Any]]
# What the element-wise protocols carry on the NumPy backend: float
# states and residuals, integer DOF indices.
_NumpyArray = NDArray[Any]
# Skfem forms and bases are untyped; they cross this module as Any, the
# convention GalerkinBasisProtocol.skfem_basis already sets.
_Form = Any
_SkfemBasis = Any
_StiffnessCheck = Callable[[_SkfemBasis, float], None]


@dataclass(frozen=True)
class GalerkinResidualTerms:
    r"""An interior residual as forms built for one basis.

    Parameters
    ----------
    stiffness : tuple of skfem BilinearForm
        Their sum :math:`K` enters as :math:`-K u`.
    loads : tuple of skfem LinearForm
        State-independent, entering with a plus sign.
    state_loads : tuple of (LinearForm, BilinearForm) pairs
        A state-dependent load and its Jacobian, both assembled with the
        state at the quadrature points as the ``u_prev`` form parameter.
    stiffness_key : hashable, optional
        Cache key of the assembled stiffness: equal keys promise an
        identical stiffness on the same basis. ``None`` disables caching.
    load_key : hashable, optional
        The same for the summed loads.
    stiffness_check : callable, optional
        ``(skfem_basis, time) -> None``, raising if the coefficients make
        the operator invalid on that basis. Runs whenever the stiffness
        is assembled, not when it is served from the cache.
    """

    stiffness: Tuple[_Form, ...] = ()
    loads: Tuple[_Form, ...] = ()
    state_loads: Tuple[Tuple[_Form, _Form], ...] = ()
    stiffness_key: Optional[Hashable] = None
    load_key: Optional[Hashable] = None
    stiffness_check: Optional[_StiffnessCheck] = None


@runtime_checkable
class GalerkinResidualTermsProtocol(Protocol):
    """A physics that states its interior residual as terms."""

    def residual_terms(
        self, skfem_basis: _SkfemBasis, time: float
    ) -> GalerkinResidualTerms:
        """Return the terms, forms built for ``skfem_basis``."""
        ...


_Matrix = Union[spmatrix, _Dense]


class _Assembly(Protocol):
    """How forms become arrays on one basis: globally or per element."""

    def basis(self) -> _SkfemBasis: ...

    def bilinear(self, form: _Form, **params: Any) -> _Matrix: ...

    def linear(self, form: _Form, **params: Any) -> _Dense: ...

    def zero_matrix(self) -> _Matrix: ...

    def zero_vector(self) -> _Dense: ...

    def apply(self, matrix: _Matrix, state: _Dense) -> _Dense: ...

    def at_quadrature(self, state: _Dense) -> Any: ...


class _KeyedCache:
    """One cached value and the key it was built for."""

    def __init__(self) -> None:
        self._key: Optional[Hashable] = None
        self._value: Optional[Any] = None

    def value(self) -> Optional[Any]:
        return self._value

    def get(self, key: Optional[Hashable], build: Callable[[], Any]) -> Any:
        if key is not None and self._value is not None and self._key == key:
            return self._value
        value = build()
        if key is not None:
            self._key, self._value = key, value
        return value


def _stiffness(
    terms: GalerkinResidualTerms,
    assembly: _Assembly,
    time: float,
    cache: _KeyedCache,
) -> _Matrix:
    def build() -> _Matrix:
        if terms.stiffness_check is not None:
            terms.stiffness_check(assembly.basis(), time)
        total = assembly.zero_matrix()
        for form in terms.stiffness:
            total = total + assembly.bilinear(form)
        return total

    result: _Matrix = cache.get(terms.stiffness_key, build)
    return result


def _loads(
    terms: GalerkinResidualTerms, assembly: _Assembly, cache: _KeyedCache
) -> _Dense:
    def build() -> _Dense:
        total = assembly.zero_vector()
        for form in terms.loads:
            total = total + assembly.linear(form)
        return total

    result: _Dense = cache.get(terms.load_key, build)
    return result


def _residual(
    terms: GalerkinResidualTerms,
    assembly: _Assembly,
    state: _Dense,
    stiffness: _Matrix,
    loads: _Dense,
) -> _Dense:
    """loads + state loads(u) - K u: the one place terms are combined."""
    residual = loads - assembly.apply(stiffness, state)
    if terms.state_loads:
        at_quadrature = assembly.at_quadrature(state)
        for load_form, _ in terms.state_loads:
            residual = residual + assembly.linear(load_form, u_prev=at_quadrature)
    return residual


def _jacobian(
    terms: GalerkinResidualTerms,
    assembly: _Assembly,
    state: _Dense,
    stiffness: _Matrix,
) -> _Matrix:
    """-K + d(state loads)/du, matching :func:`_residual`."""
    jacobian = -stiffness
    if terms.state_loads:
        at_quadrature = assembly.at_quadrature(state)
        for _, jacobian_form in terms.state_loads:
            jacobian = jacobian + assembly.bilinear(jacobian_form, u_prev=at_quadrature)
    return jacobian


def _skfem() -> Any:
    return import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class _GlobalAssembly:
    """Forms assembled and summed over the whole basis."""

    def __init__(self, skfem_basis: _SkfemBasis) -> None:
        self._skfem_basis = skfem_basis
        self._ndofs = int(skfem_basis.N)

    def basis(self) -> _SkfemBasis:
        return self._skfem_basis

    def bilinear(self, form: _Form, **params: Any) -> _Matrix:
        matrix: _Matrix = _skfem().asm(form, self._skfem_basis, **params)
        return matrix

    def linear(self, form: _Form, **params: Any) -> _Dense:
        return np.asarray(_skfem().asm(form, self._skfem_basis, **params))

    def zero_matrix(self) -> _Matrix:
        return csr_matrix((self._ndofs, self._ndofs))

    def zero_vector(self) -> _Dense:
        return np.zeros(self._ndofs)

    def apply(self, matrix: _Matrix, state: _Dense) -> _Dense:
        return np.asarray(matrix @ state)

    def at_quadrature(self, state: _Dense) -> Any:
        return self._skfem_basis.interpolate(state)


class _ElementAssembly:
    """Forms assembled per element of a restricted basis, not summed.

    Matrices are (nselected, nper, nper) with ``[e, i, j]`` the test-``i``,
    trial-``j`` entry; vectors (nselected, nper). Skfem's ``coo_data``
    lists a bilinear form's entries trial index first, hence the
    transpose.
    """

    def __init__(self, restricted_basis: _SkfemBasis) -> None:
        self._skfem_basis = restricted_basis
        self._nper = int(restricted_basis.Nbfun)
        self._nselected = int(restricted_basis.nelems)
        # Basis values at the quadrature points, (nper, nselected, nquad):
        # the local states reach the quadrature points without any
        # global vector.
        self._phi = np.stack(
            [
                np.asarray(restricted_basis.basis[ii][0].value)
                for ii in range(self._nper)
            ]
        )

    def basis(self) -> _SkfemBasis:
        return self._skfem_basis

    def bilinear(self, form: _Form, **params: Any) -> _Matrix:
        data = np.asarray(form.coo_data(self._skfem_basis, **params).data)
        blocks = data.reshape(self._nper, self._nper, self._nselected)
        return np.transpose(blocks, (2, 1, 0))

    def linear(self, form: _Form, **params: Any) -> _Dense:
        data = np.asarray(form.coo_data(self._skfem_basis, **params).data)
        return data.reshape(self._nper, self._nselected).T

    def zero_matrix(self) -> _Matrix:
        return np.zeros((self._nselected, self._nper, self._nper))

    def zero_vector(self) -> _Dense:
        return np.zeros((self._nselected, self._nper))

    def apply(self, matrix: _Matrix, state: _Dense) -> _Dense:
        applied: _Dense = np.einsum("eij,ej->ei", np.asarray(matrix), state)
        return applied

    def at_quadrature(self, state: _Dense) -> Any:
        # (nselected, nquad): the raw array form the state-load kernels
        # accept for u_prev.
        return np.einsum("ieq,ei->eq", self._phi, state)


class SkfemTermAssembler:
    """Global assembly of a physics' terms, with keyed caches.

    Owned by one physics instance. Caches the summed stiffness and loads
    on the keys the terms carry, so Newton iterations and time steps
    reuse them and a coefficient rebind (which changes the key)
    invalidates them.
    """

    def __init__(self) -> None:
        self._stiffness_cache = _KeyedCache()
        self._load_cache = _KeyedCache()

    def stiffness(
        self, terms: GalerkinResidualTerms, skfem_basis: _SkfemBasis, time: float
    ) -> spmatrix:
        """Return the summed stiffness :math:`K`, a sparse matrix."""
        stiffness: spmatrix = _stiffness(
            terms, _GlobalAssembly(skfem_basis), time, self._stiffness_cache
        )
        return stiffness

    def loads(self, terms: GalerkinResidualTerms, skfem_basis: _SkfemBasis) -> _Dense:
        """Return the summed state-independent loads."""
        return _loads(terms, _GlobalAssembly(skfem_basis), self._load_cache)

    def residual(
        self,
        terms: GalerkinResidualTerms,
        skfem_basis: _SkfemBasis,
        state: _Dense,
        time: float,
    ) -> _Dense:
        """Return the assembled interior residual. Shape: (ndofs,)."""
        assembly = _GlobalAssembly(skfem_basis)
        return _residual(
            terms,
            assembly,
            state,
            _stiffness(terms, assembly, time, self._stiffness_cache),
            _loads(terms, assembly, self._load_cache),
        )

    def jacobian(
        self,
        terms: GalerkinResidualTerms,
        skfem_basis: _SkfemBasis,
        state: _Dense,
        time: float,
    ) -> spmatrix:
        """Return the assembled interior Jacobian, a sparse matrix."""
        assembly = _GlobalAssembly(skfem_basis)
        jacobian: spmatrix = _jacobian(
            terms,
            assembly,
            state,
            _stiffness(terms, assembly, time, self._stiffness_cache),
        )
        return jacobian

    def cached_stiffness(self) -> Optional[spmatrix]:
        """The cached stiffness, or None; for tests of the caching."""
        value: Optional[spmatrix] = self._stiffness_cache.value()
        return value

    def cached_loads(self) -> Optional[_Dense]:
        """The cached loads, or None; for tests of the caching."""
        value: Optional[_Dense] = self._load_cache.value()
        return value


class SkfemRestrictedOperator:
    """A physics restricted to a set of elements, evaluated per element.

    Satisfies ``RestrictedElementOperatorProtocol[NDArray]``: skfem
    assembles in NumPy, so this implementation is NumPy-typed rather
    than generic, and pairing it with a non-NumPy consumer is a type
    error instead of a silent conversion.

    Built once per element set: the restricted basis, the local DOF
    numbering and the basis values at the quadrature points are
    precomputed. Each evaluation asks the physics for its terms on the
    restricted basis, so the integrands are the physics' own; this class
    only assembles them. Rows and states coincide (the element's nodes).

    Coefficients are read from the physics at evaluation time, which
    holds the whole field; supplying only the elements' coefficient
    values is a later capability. Basis values are stored per selected
    element, as skfem lays them out; an assembler that shares geometry
    between congruent elements would store them once per class.

    Parameters
    ----------
    physics : GalerkinResidualTermsProtocol
        Supplies the terms.
    skfem_basis : skfem basis
        The physics' full basis.
    elements : ElementSet
        The elements to restrict to, on the NumPy backend.
    """

    def __init__(
        self,
        physics: GalerkinResidualTermsProtocol,
        skfem_basis: _SkfemBasis,
        elements: ElementSet[_NumpyArray],
    ) -> None:
        if not isinstance(physics, GalerkinResidualTermsProtocol):
            raise TypeError(
                "physics must satisfy GalerkinResidualTermsProtocol, got "
                f"{type(physics).__name__}"
            )
        indices = np.asarray(elements.indices(), dtype=np.int64)
        nelems = int(skfem_basis.nelems)
        if indices.size and (indices.min() < 0 or indices.max() >= nelems):
            raise ValueError(f"element indices must lie in [0, {nelems})")
        self._physics = physics
        self._elements = elements
        self._restricted = skfem_basis.with_elements(indices)
        element_global = np.asarray(self._restricted.element_dofs).T
        self._global_dofs: _NumpyArray = np.unique(element_global)
        self._element_local: _NumpyArray = np.searchsorted(
            self._global_dofs, element_global
        )
        self._assembly = _ElementAssembly(self._restricted)
        self._stiffness_cache = _KeyedCache()
        self._load_cache = _KeyedCache()

    def bkd(self) -> Backend[_NumpyArray]:
        """Return the element set's backend, which is NumPy."""
        return self._elements.bkd()

    def elements(self) -> ElementSet[_NumpyArray]:
        """Return the element set."""
        return self._elements

    def global_dofs(self) -> _NumpyArray:
        """Return the global DOF of each local DOF. Shape: (nlocal_dofs,)."""
        return self._global_dofs

    def element_row_dofs(self) -> _NumpyArray:
        """Return each element's local DOFs. Shape: (nselected, nper)."""
        return self._element_local

    def element_state_dofs(self) -> _NumpyArray:
        """The same as the rows: an element reads its own nodes."""
        return self._element_local

    def element_residuals(self, local_states: _NumpyArray, time: float) -> _NumpyArray:
        """Return the element residuals. Shape: (nselected, nper)."""
        terms = self._physics.residual_terms(self._restricted, time)
        return _residual(
            terms,
            self._assembly,
            local_states,
            _stiffness(terms, self._assembly, time, self._stiffness_cache),
            _loads(terms, self._assembly, self._load_cache),
        )

    def element_jacobians(self, local_states: _NumpyArray, time: float) -> _NumpyArray:
        """Return the element Jacobians. Shape: (nselected, nper, nper)."""
        terms = self._physics.residual_terms(self._restricted, time)
        jacobians: _NumpyArray = np.asarray(
            _jacobian(
                terms,
                self._assembly,
                local_states,
                _stiffness(terms, self._assembly, time, self._stiffness_cache),
            )
        )
        return jacobians

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nselected={self._elements.nselected()}, "
            f"physics={type(self._physics).__name__})"
        )
