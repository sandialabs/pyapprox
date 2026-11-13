"""The condition visitor: each kind calls its own visitor method, and a
visitor that lacks one does not satisfy the protocol."""

from pyapprox.pde.verification.conditions import (
    ConditionVisitorProtocol,
    Dirichlet,
    Neumann,
    Robin,
)


class _Names:
    def dirichlet(self, condition: Dirichlet) -> str:
        return "dirichlet"

    def neumann(self, condition: Neumann) -> str:
        return "neumann"

    def robin(self, condition: Robin) -> str:
        return f"robin {condition.alpha()}"


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


def test_incomplete_visitor_is_not_one() -> None:
    assert not isinstance(_NoRobin(), ConditionVisitorProtocol)
