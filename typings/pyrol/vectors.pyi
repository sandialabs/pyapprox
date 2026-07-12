from typing import Any

def __getattr__(name: str) -> Any: ...

# Any (not a typed class): pyapprox monkey-patches NumPyVector.__init__ /
# axpy for performance, which a typed class would reject
NumPyVector: Any
