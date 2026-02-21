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


def validate_piecewise_growth_compatibility(
    factories: Sequence[object],
    growth_rules: Union[object, Sequence[object]],
    max_level: int = 5,
) -> None:
    """Validate that growth rules are compatible with piecewise basis factories.

    Piecewise polynomial bases have specific node count requirements:
    - piecewise_quadratic: Requires odd number of nodes
    - piecewise_cubic: Requires (n - 4) % 3 == 0

    This function checks the first few levels to catch incompatibilities early.

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
        If a growth rule produces incompatible node counts for a piecewise basis.

    Notes
    -----
    Growth rule requirements by basis type:

    - piecewise_linear: Any growth rule works
    - piecewise_quadratic: Use ClenshawCurtisGrowthRule() (produces 1, 3, 5, 9, 17, ...)
    - piecewise_cubic: Use CubicNestedGrowthRule() (produces 1, 4, 7, 13, 25, ...)
    - gauss, leja, clenshaw_curtis: LinearGrowthRule or ClenshawCurtisGrowthRule
    """
    # Import here to avoid circular imports
    from pyapprox.typing.surrogates.sparsegrids.basis_factory import PiecewiseFactory

    # Normalize growth_rules to a list
    if isinstance(growth_rules, list):
        rules_list = growth_rules
    else:
        rules_list = [growth_rules] * len(factories)

    for dim, (factory, rule) in enumerate(zip(factories, rules_list)):
        # Only check PiecewiseFactory instances
        if not isinstance(factory, PiecewiseFactory):
            continue

        poly_type = getattr(factory, "_poly_type", None)
        if poly_type is None:
            continue

        # Check node counts for first few levels
        for level in range(1, max_level + 1):
            npts = rule(level)

            if poly_type == "quadratic" and npts > 1 and npts % 2 == 0:
                raise ValueError(
                    f"piecewise_quadratic (dimension {dim}) requires odd number "
                    f"of nodes, but growth_rule({level}) = {npts}. "
                    f"Use ClenshawCurtisGrowthRule() instead of {rule!r}."
                )

            if poly_type == "cubic" and npts > 1 and (npts - 4) % 3 != 0:
                raise ValueError(
                    f"piecewise_cubic (dimension {dim}) requires (n - 4) % 3 == 0, "
                    f"but growth_rule({level}) = {npts}. "
                    f"Use CubicNestedGrowthRule() instead of {rule!r}."
                )
