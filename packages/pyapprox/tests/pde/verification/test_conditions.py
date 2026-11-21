"""The condition visitor: each kind calls its own visitor method, and a
visitor that lacks one does not satisfy the protocol."""

from pyapprox.pde.expressions.symbols import pde_names, standard
from pyapprox.pde.verification.conditions import (
    ConditionVisitorProtocol,
    Dirichlet,
    Neumann,
    Periodic,
    Robin,
    Unprescribed,
)


class _Names:
    def dirichlet(self, condition: Dirichlet) -> str:
        return "dirichlet"

    def neumann(self, condition: Neumann) -> str:
        return "neumann"

    def robin(self, condition: Robin) -> str:
        return f"robin {condition.alpha(pde_names(standard(1)))}"

    def periodic(self, condition: Periodic) -> str:
        return "periodic"

    def unprescribed(self, condition: Unprescribed) -> str:
        return "unprescribed"


class _NoRobin:
    def dirichlet(self, condition: Dirichlet) -> str:
        return "dirichlet"

    def neumann(self, condition: Neumann) -> str:
        return "neumann"


def test_each_kind_calls_its_method() -> None:
    visitor = _Names()
    assert isinstance(visitor, ConditionVisitorProtocol)
    assert Dirichlet().accept(visitor) == "dirichlet"
    assert Neumann().accept(visitor) == "neumann"
    assert Robin("1 + x").accept(visitor) == "robin x + 1"
    assert Periodic().accept(visitor) == "periodic"
    assert Unprescribed().accept(visitor) == "unprescribed"


def test_unprescribed_has_no_data() -> None:
    """Nothing to parse and no right-hand side, so the problem leaves the
    row out of the part's data."""
    condition = Unprescribed()
    names = pde_names(standard(1))
    assert condition.expressions(names) == ()
    assert repr(condition) == "Unprescribed()"


def test_incomplete_visitor_is_not_one() -> None:
    assert not isinstance(_NoRobin(), ConditionVisitorProtocol)
