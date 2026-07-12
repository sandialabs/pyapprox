"""Minimal local type stubs for the pyrol (Trilinos ROL) bindings.

Only the surface pyapprox touches is declared; everything else falls back
to Any via module-level __getattr__. Method slots on the base classes are
declared as Any attributes (not methods) because pyapprox attaches
implementations conditionally onto adapter subclasses. Signatures on the
pyrol side are deliberately NOT modelled: they would have to mirror
pybind11 overloads for a manually-built optional extra and would drift.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...

class Vector:
    array: Any
    def __init__(self, *args: Any) -> None: ...
    def __getitem__(self, index: Any) -> Any: ...
    def __setitem__(self, index: Any, value: Any) -> None: ...

class Objective:
    value: Any
    gradient: Any
    hessVec: Any
    def __init__(self) -> None: ...

class Constraint:
    value: Any
    applyJacobian: Any
    applyAdjointJacobian: Any
    applyAdjointHessian: Any
    def __init__(self) -> None: ...

class LinearOperator:
    apply: Any
    applyAdjoint: Any
    def __init__(self) -> None: ...

class LinearConstraint(Constraint):
    def __init__(self, op: Any, b: Any) -> None: ...

class Bounds:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Problem:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def addBoundConstraint(self, *args: Any) -> None: ...
    def addLinearConstraint(self, *args: Any) -> None: ...
    def addConstraint(self, *args: Any) -> None: ...

class Solver:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def solve(self, *args: Any) -> Any: ...
