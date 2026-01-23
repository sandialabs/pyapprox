"""Runtime protocol validation helpers for sparsegrids module.

This module provides validation functions that check if objects satisfy
the required protocols at runtime. These checks provide clear error messages
when incorrect types are passed to constructors.

Functions are in a separate module to avoid circular imports.
"""

from typing import Sequence, Union

from pyapprox.typing.util.backends.protocols import Backend
from pyapprox.typing.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
    AdmissibilityCriteriaProtocol,
    Basis1DProtocol,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import BasisFactoryProtocol


def validate_backend(bkd: object, param_name: str = "bkd") -> None:
    """Validate that bkd satisfies Backend protocol."""
    if not isinstance(bkd, Backend):
        raise TypeError(
            f"{param_name} must satisfy Backend protocol, "
            f"got {type(bkd).__name__}"
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
            f"{param_name} must satisfy Basis1DProtocol, "
            f"got {type(basis).__name__}"
        )
