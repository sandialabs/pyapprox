from typing import Protocol, Generic

from pyapprox.typing.util.backend import Array, Backend


class ExplicitODEResidualProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def __call__(self, iterate: Array) -> Array: ...

    def set_time(self, time: float) -> None: ...
