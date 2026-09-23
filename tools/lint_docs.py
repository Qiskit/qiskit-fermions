#!/usr/bin/env python3
# This code is a Qiskit project.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Lint documentation markup that builds cleanly but is still wrong.

``make docs`` already runs ``sphinx-build -W``, so anything Sphinx can diagnose is a hard error
already.  The checks here exist precisely because Sphinx *cannot* see their subject: each one
guards a defect that produces a clean build and a broken page, and each one has already shipped
at least once.

*   :func:`check_embedded_uri` -- an *embedded URI* used where a cross-reference was meant.
    ``docutils`` treats the target of ```text <target>`_`` as a URI, so a target that is really a
    label becomes a relative link to a page that does not exist.  It is valid reStructuredText,
    diagnosed at no report level, and therefore invisible to ``-W``: Sphinx is never handed a
    reference to resolve.  Shipped twice -- fixed by hand in ``c3101c4c`` and again in
    ``e61db631`` (https://github.com/Qiskit/qiskit-fermions/issues/355).

*   :func:`check_alt_text` -- an image without a text alternative.  Nothing in a Sphinx build
    requires one, so the omission surfaces only downstream, in the IBM Quantum Platform's own
    documentation linter.  That is what happened in ``2c298333``.

Both are *form* checks.  Whether a reference resolves is Sphinx's job and it does it better than
this script could: labels may be defined in a Python or Rust docstring rather than in ``.rst``
(``docs/guides/transpilation.rst`` refers to one defined in
``python/qiskit_fermions/transpiler/passes/__init__.py``), so a target-existence check here would
report false positives on correct markup.  Do not add one.

Directives and citations live in reStructuredText, in Rust doc comments and in the generated type
stubs alike, so every check accepts the ``///``, ``//!`` and ``#`` comment prefixes and arbitrary
indentation.

Usage:
    python tools/lint_docs.py crates docs python --ignore crates/qiskit-pyo3-ffi
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import NamedTuple

#: Extensions that can carry documentation markup.  ``.rst`` is the guides and the API-reference
#: scaffolding; ``.rs`` is the ``pyext``/``cext`` docstrings that become the API reference; ``.py``
#: and ``.pyi`` are the pure-Python modules and the generated stubs.
EXTENSIONS = ("rst", "rs", "py", "pyi")

#: Strips a comment prefix so a Rust doc comment reads like the reStructuredText it contains.  The
#: single optional space matches how the text is written, not how much indentation follows it.
COMMENT_PREFIX = re.compile(r"^\s*(?://[/!]|#)?[ \t]?")

#: An embedded URI: ```text <target>`_`` or its anonymous ```text <target>`__`` form.
#:
#: Three details are load-bearing:
#:
#: *   ``re.DOTALL``, applied by the caller, and ``[^`<]*?`` for the text.  The target may be
#:     wrapped across lines (the real defect in ``e61db631`` was) and ``docutils`` joins the halves
#:     with nothing at all, so a line-oriented scan misses the very case that motivated this check.
#: *   The trailing ``_``.  It is what distinguishes an embedded URI from a cross-reference role:
#:     ``:ref:`text <label>``` and every ``:c:func:``/``:c:struct:``/``:class:`` sibling carry the
#:     same angle-bracket payload and are all correct, so without it every one of them is a false
#:     positive.
#: *   The lookbehind and lookahead, which reject a role whose own name ends in ``_`` and a
#:     trailing word character (```a <b>`_foo`` is not a reference).
EMBEDDED_URI = re.compile(
    r"(?<![:`\w])`(?P<text>[^`<]*?)<(?P<target>[^<>`]+)>`(?P<suffix>__?)(?!\w)",
    re.DOTALL,
)

#: A URI scheme, a protocol-relative URL, or a fragment -- i.e. a target that really is a URI.
#: Anything else in an embedded URI is a cross-reference wearing the wrong syntax.
URI_SCHEME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//|#)")

#: The directives that place an image on the page.  ``figure`` is deliberately included: it is the
#: form ``2c298333`` moved to, and the tool the documentation repository ships for this check omits
#: it (https://github.com/Qiskit/documentation/blob/main/scripts/image-tester).
#: Anchored, and matched against a line whose indentation and comment prefix have already been
#: stripped: a directive opens a line, so prose that merely names one ("the ``.. image::``
#: directive") is not one.
IMAGE_DIRECTIVE = re.compile(r"\.\.[ \t]+(?P<kind>image|figure|plot)::")

#: An option line inside a directive's option block, e.g. ``:alt:`` or ``:nofigs:``.
DIRECTIVE_OPTION = re.compile(r"^:(?P<name>[^:]+):")


class Finding(NamedTuple):
    """One problem, at one place."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        """Render as ``path:line: message``, so an editor can jump straight to it."""
        return f"{self.path}:{self.line}: {self.message}"


def _strip_prefix(line: str) -> str:
    """Return ``line`` with any comment prefix and leading indentation removed."""
    return COMMENT_PREFIX.sub("", line).strip()


def check_embedded_uri(text: str, path: str) -> list[Finding]:
    """Find embedded URIs whose target is not a URI, and so is really a cross-reference."""
    findings = []
    for match in EMBEDDED_URI.finditer(text):
        # `docutils` discards the whitespace inside a wrapped target rather than folding it to a
        # space, so normalising the same way is what makes the scheme test meaningful.
        target = "".join(match.group("target").split())
        if URI_SCHEME.match(target):
            continue
        label = " ".join(match.group("text").split())
        findings.append(
            Finding(
                path,
                text.count("\n", 0, match.start()) + 1,
                f"`{label} <{target}>`{match.group('suffix')} is an embedded URI, so "
                f"'{target}' is rendered as a relative link rather than resolved. Use "
                f":ref:`{label} <{target}>` for a label, or :doc: for a document.",
            )
        )
    return findings


def check_alt_text(text: str, path: str) -> list[Finding]:
    """Find image directives that offer no text alternative.

    ``:alt:`` satisfies every directive.  ``.. plot::`` may instead carry ``:nofigs:``, which
    means it renders code and no image, and so has nothing to describe.
    """
    findings = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = IMAGE_DIRECTIVE.match(_strip_prefix(lines[index]))
        if match is None:
            index += 1
            continue

        # Consume the option block.  It ends at the first line that is blank or is not an option,
        # which is also how `docutils` delimits it -- including at end of file, where the tool the
        # documentation repository ships stops checking and silently passes the directive.
        options = set()
        cursor = index + 1
        while cursor < len(lines):
            stripped = _strip_prefix(lines[cursor])
            option = DIRECTIVE_OPTION.match(stripped)
            if option is None:
                break
            options.add(option.group("name"))
            cursor += 1

        kind = match.group("kind")
        permitted = {"alt", "nofigs"} if kind == "plot" else {"alt"}
        if not options & permitted:
            expected = (
                "':alt:' (or ':nofigs:' if it renders no image)" if kind == "plot" else "':alt:'"
            )
            findings.append(
                Finding(
                    path,
                    index + 1,
                    f".. {kind}:: has no text alternative; add {expected}.",
                )
            )
        index = max(cursor, index + 1)
    return findings


#: Every check, in reporting order.  A check takes the full text of one file and its path, and
#: returns the problems it found.
CHECKS = (check_embedded_uri, check_alt_text)


def check_text(text: str, path: str) -> list[Finding]:
    """Run every check over the contents of one file."""
    return [finding for check in CHECKS for finding in check(text, path)]


def discover_files(paths: list[str], ignore_paths: list[str] | None) -> list[pathlib.Path]:
    """Find every file that can carry documentation markup, in a list of trees.

    A path may name a file rather than a tree, which is what makes it possible to check a single
    guide.  Accepting that explicitly matters: globbing a file path yields nothing, so without
    this a mistyped or misused argument would report success having read no files at all.
    """
    ignored = [pathlib.Path(path) for path in ignore_paths or ()]
    found = set()
    for path in map(pathlib.Path, paths):
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(
                file for extension in EXTENSIONS for file in path.glob(f"**/*.{extension}")
            )
        else:
            # Silence is this script's success condition, so a path that does not exist has to be
            # loud: it would otherwise be indistinguishable from a tree with nothing wrong in it.
            raise FileNotFoundError(path)
    return sorted(
        file for file in found if not any(file.is_relative_to(ignore) for ignore in ignored)
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Check documentation markup.")
    parser.add_argument("paths", type=str, nargs="+", help="Paths to scan")
    parser.add_argument("--ignore", type=str, nargs="+", help="Paths to ignore")
    args = parser.parse_args()

    findings = [
        finding
        for file in discover_files(args.paths, ignore_paths=args.ignore)
        for finding in check_text(file.read_text(encoding="utf-8"), str(file))
    ]
    if findings:
        for finding in findings:
            sys.stderr.write(f"{finding}\n")
        sys.stderr.write(f"\n{len(findings)} problem(s) found in the documentation markup.\n")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    _main()
