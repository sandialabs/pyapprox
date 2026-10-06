"""Runtime protocol validation helpers for sparsegrids module.

This module provides validation functions that check if objects satisfy
the required protocols at runtime. These checks provide clear error messages
when incorrect types are passed to constructors.

Functions are in a separate module to avoid circular imports.
"""

from typing import Sequence, Union

from pyapprox.surrogates.affine.protocols import (
    AdmissibilityCriteriaProtocol,
    Basis1DProtocol,
    IndexGrowthRuleProtocol,
)
from pyapprox.surrogates.sparsegrids.basis_factory import BasisFactoryProtocol
from pyapprox.util.backends.protocols import Backend


def validate_backend(bkd: object, param_name: str = "bkd") -> None:
    """Validate that bkd satisfies Backend protocol."""
    if not isinstance(bkd, Backend):
        raise TypeError(
            f"{param_name} must satisfy Backend protocol, got {type(bkd).__name__}"
        )


def validate_basis_factories(
    factories: Sequence[object], param_name: str = "basis_factories"
) -> None:
    """Validate that all factories satisfy BasisFactoryProtocol."""
    for i, factory in enumerate(factories):
        if not isinstance(factory, BasisFactoryProtocol):
            raise TypeError(
                f"{param_name}[{i}] must satisfy BasisFactoryProtocol, "
                f"got {type(factory).__name__}"
            )


def validate_growth_rules(
    rules: Union[object, Sequence[object]], param_name: str = "growth_rules"
) -> None:
    """Validate that growth_rules satisfy IndexGrowthRuleProtocol."""
    if isinstance(rules, list):
        for i, rule in enumerate(rules):
            if not isinstance(rule, IndexGrowthRuleProtocol):
                raise TypeError(
                    f"{param_name}[{i}] must satisfy IndexGrowthRuleProtocol, "
                    f"got {type(rule).__name__}"
                )
    else:
        if not isinstance(rules, IndexGrowthRuleProtocol):
            raise TypeError(
                f"{param_name} must satisfy IndexGrowthRuleProtocol, "
                f"got {type(rules).__name__}"
            )


def validate_admissibility(
    admissibility: object, param_name: str = "admissibility"
) -> None:
    """Validate that admissibility satisfies AdmissibilityCriteriaProtocol."""
    if not isinstance(admissibility, AdmissibilityCriteriaProtocol):
        raise TypeError(
            f"{param_name} must satisfy AdmissibilityCriteriaProtocol, "
            f"got {type(admissibility).__name__}"
        )


def validate_basis1d(basis: object, param_name: str = "univariate_basis") -> None:
    """Validate that basis satisfies Basis1DProtocol."""
    if not isinstance(basis, Basis1DProtocol):
        raise TypeError(
            f"{param_name} must satisfy Basis1DProtocol, got {type(basis).__name__}"
        )


def validate_piecewise_growth_compatibility(
    factories: Sequence[object],
    growth_rules: Union[object, Sequence[object]],
    max_level: int = 5,
) -> None:
    """Validate that growth rules give node counts each basis accepts.

    Each factory's basis is built at the node count of every level up to
    ``max_level``, and the basis itself rejects counts it cannot use, so
    the check needs no knowledge of basis types and covers any injected
    basis. For example a piecewise quadratic basis needs an odd count
    (``ClenshawCurtisGrowthRule`` gives 1, 3, 5, 9, 17, ...) and a piecewise
    cubic one ``3k + 1`` (``CubicNestedGrowthRule`` gives 1, 4, 7, 13, 25,
    ...); a piecewise linear basis accepts any count.

    Parameters
    ----------
    factories : Sequence[BasisFactoryProtocol]
        List of basis factories.
    growth_rules : IndexGrowthRuleProtocol or Sequence[IndexGrowthRuleProtocol]
        Growth rule(s) to validate.
    max_level : int, optional
        Maximum level to check. Default: 5.

    Raises
    ------
    ValueError
        If a growth rule gives a node count a factory's basis rejects.
    """
    validate_basis_factories(factories)
    validate_growth_rules(growth_rules)
    if isinstance(growth_rules, list):
        rules_list = growth_rules
    else:
        rules_list = [growth_rules] * len(factories)

    for dim, (factory, rule) in enumerate(zip(factories, rules_list)):
        if not isinstance(factory, BasisFactoryProtocol) or not isinstance(
            rule, IndexGrowthRuleProtocol
        ):
            continue  # unreachable: both were validated above
        basis = factory.create_basis()
        for level in range(1, max_level + 1):
            npts = rule(level)
            try:
                basis.set_nterms(npts)
            except ValueError as error:
                raise ValueError(
                    f"the basis of dimension {dim} rejects growth_rule({level}) "
                    f"= {npts} nodes from {rule!r}: {error}. Use a growth rule "
                    "whose node counts this basis accepts."
                ) from error
