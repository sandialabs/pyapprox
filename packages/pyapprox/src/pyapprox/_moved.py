"""Old import paths of moved modules.

When a directory moves, :data:`MOVED` maps its old dotted prefix to the new
one. :func:`install` puts an :class:`AliasFinder` first on
``sys.meta_path``, so ``import old.mod`` returns the module object of
``new.mod`` itself: class identity, ``isinstance``, ``mock.patch`` on old
target strings and unpickling of objects pickled under the old path all
keep working. Each old prefix warns once per process.

The finder must come first. A moved directory keeps its old location for
type stubs, without an ``__init__.py``; a finder after the path finder
would let Python import that location as an empty namespace package, and
old submodule imports would then fail.

Only the standard library is imported here: ``pyapprox`` loads this
module on every import.
"""

import importlib
import importlib._abc
import sys
import warnings
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Mapping, Optional, Sequence

MOVED: dict[str, str] = {}
"""Old prefix -> new prefix, one entry per moved directory."""


class _AliasLoader(importlib._abc.Loader):
    """Loads an old name as the module already at its new name.

    The base is ``importlib.abc.Loader`` itself, taken from the module
    that defines it and that interpreter startup already loads:
    ``importlib.abc`` pulls in ``importlib.resources`` (tens of
    milliseconds) on every ``import pyapprox``. Finders are read
    structurally, so :class:`AliasFinder` needs no base.
    """

    def __init__(self, new_name: str) -> None:
        self._new_name = new_name

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return importlib.import_module(self._new_name)

    def exec_module(self, module: ModuleType) -> None:
        # The module ran under its new name; running it again would
        # create a second copy of everything it defines.
        return None


class AliasFinder:
    """Resolves names under an old prefix to the same names under the new
    one.

    Parameters
    ----------
    table
        Old prefix -> new prefix. The longest matching prefix wins.
    """

    def __init__(self, table: Mapping[str, str]) -> None:
        self._table = table
        self._warned: set[str] = set()

    def _match(self, fullname: str) -> Optional[str]:
        matches = [
            old
            for old in self._table
            if fullname == old or fullname.startswith(old + ".")
        ]
        return max(matches, key=len) if matches else None

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        old = self._match(fullname)
        if old is None:
            return None
        new_name = self._table[old] + fullname[len(old) :]
        if old not in self._warned:
            self._warned.add(old)
            warnings.warn(
                f"{old} moved to {self._table[old]}; import {new_name} instead "
                f"of {fullname}",
                DeprecationWarning,
                stacklevel=2,
            )
        return ModuleSpec(fullname, _AliasLoader(new_name))


def install(table: Mapping[str, str] = MOVED) -> AliasFinder:
    """Put a finder for ``table`` first on ``sys.meta_path``, once per
    table, and return it."""
    for finder in sys.meta_path:
        if isinstance(finder, AliasFinder) and finder._table is table:
            return finder
    finder = AliasFinder(table)
    sys.meta_path.insert(0, finder)
    return finder
