# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Marks every test that depends on an optional dependency, so ``make testoptional`` can select it.

The suite guards optional dependencies in two ways, and neither is selectable on its own:

- ``pytest.importorskip(...)``, in a test module or inside a single test. This leaves no marker
  behind, and at module scope it raises during *collection* when the dependency is missing, so the
  module's tests never become items that a marker could be attached to.
- a Sybil ``.. skip: start if(not HAS_...)`` region around a docstring doctest. The resulting item is
  a ``SybilItem`` with no marker and no module namespace.

Both are detected here rather than restated as a list of files or of dependency names, so that
guarding a new test is all it takes to have it selected. The two hooks below ask the guard mechanisms
themselves what they cover:

- a call to ``pytest.importorskip`` is found in the source itself, so the guard declares its own
  test. A module-scope call gates the whole file, a call inside one test function gates only that
  test, and neither is tied to *which* dependency was requested.
- Sybil exposes each doctest as an ``Example`` whose region carries the evaluator that produced it.
  A skip directive's evaluator is a ``Skipper`` and its ``parsed`` payload gives the action and the
  condition, so the ``start``/``end`` line numbers delimit exactly which doctests a region gates.

The asymmetry with collection is deliberate and harmless: the marker is only ever *needed* when the
dependency is present, because that is when there is something to select. When it is absent the
module skips or fails to collect exactly as it would have anyway.
"""

import ast
import functools
import pathlib

import pytest
from sybil.evaluators.skip import Skipper


@functools.cache
def _importorskip_guards(path: str) -> tuple[bool, frozenset[str]]:
    """Returns whether ``path`` guards at module scope, and which of its functions guard themselves.

    Asking the source rather than a list of dependency names is what keeps this from needing
    maintenance: a call to ``importorskip`` *is* the declaration that a test needs something optional,
    whatever it happens to be importing. A module-scope call gates the whole file; a call inside a
    function gates only that function, which is the distinction the AST makes and a plain substring
    search could not.
    """
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - collected files parse
        return False, frozenset()

    def is_importorskip_call(node: ast.AST) -> bool:
        # Matches both `pytest.importorskip(...)` and a bare `importorskip(...)`.
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr == "importorskip"
        return isinstance(func, ast.Name) and func.id == "importorskip"

    def calls_importorskip(node: ast.AST) -> bool:
        return any(is_importorskip_call(sub) for sub in ast.walk(node))

    module_scope = False
    guarded_functions = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if calls_importorskip(node):
                guarded_functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and calls_importorskip(
                    sub
                ):
                    guarded_functions.add(sub.name)
        elif calls_importorskip(node):
            module_scope = True

    return module_scope, frozenset(guarded_functions)


def _skip_gated_lines(items: list[pytest.Item]) -> dict[str, set[int]]:
    """Returns, per file, the line numbers of the doctests that a Sybil skip region gates.

    A ``.. skip: start`` region runs until the matching ``.. skip: end``, and Sybil hands us both as
    items of their own, in document order, with the condition attached to the ``start``. Tracking the
    open region while walking a file therefore identifies exactly the gated doctests, rather than
    treating every doctest in a file that happens to contain a region as gated.
    """
    per_file: dict[str, list[tuple[int, object]]] = {}
    for item in items:
        example = getattr(item, "example", None)
        if example is None:
            continue
        per_file.setdefault(str(item.path), []).append((example.line, example))

    gated: dict[str, set[int]] = {}
    for path, examples in per_file.items():
        open_from: int | None = None
        for line, example in sorted(examples, key=lambda pair: pair[0]):
            if isinstance(getattr(example.region, "evaluator", None), Skipper):
                action, condition = example.parsed
                if action in ("start", "next") and condition:
                    open_from = line
                    # The directive itself has to be selected along with what it gates: Sybil's skip
                    # state is per-document and installed as the `start` item is evaluated, so running
                    # the gated doctests without it would leave the region never opened and the
                    # doctests would execute rather than skip.
                    gated.setdefault(path, set()).add(line)
                elif action == "end":
                    open_from = None
                    if path in gated:
                        gated[path].add(line)
                continue
            if open_from is not None:
                gated.setdefault(path, set()).add(line)
    return gated


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Adds the ``optionaldep`` marker to every collected test guarded by an optional dependency."""
    gated_doctests = _skip_gated_lines(items)

    for item in items:
        example = getattr(item, "example", None)
        if example is not None:
            if example.line in gated_doctests.get(str(item.path), ()):
                item.add_marker(pytest.mark.optionaldep)
            continue
        module_scope, guarded_functions = _importorskip_guards(str(item.path))
        if module_scope or item.originalname in guarded_functions:
            item.add_marker(pytest.mark.optionaldep)
