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

"""Rewrite absolute IBM Quantum Platform API-reference links to local targets.

The hosted documentation lives on the IBM Quantum Platform, so ``docs/index.rst`` points its
API-reference toctree entries at absolute ``quantum.cloud.ibm.com`` URLs.  That is correct for the
artifact the platform ingests, but it makes a *standalone* build of this documentation -- a branch or
pull-request preview on GitHub Pages, or a local ``make docs-local`` -- link away from itself: the
sidebar's "Python API reference" would navigate to the published docs rather than to the copy you are
reading.

This script rewrites exactly those links, in place, to the equivalent pages inside the same tree.  Run
it on a *copy* of the build when the pristine artifact still matters; ``make docs-local`` deliberately
mutates ``docs/_build/html`` because a local build is not an artifact.

Two properties make the rewrite safe, and both are load-bearing:

*   **Only complete URLs are replaced, never prefixes.**  The build also contains hundreds of
    legitimate intersphinx links to *other* projects on the same host (``docs/api/qiskit/...`` for the
    Qiskit SDK, ``docs/api/qiskit-c/...`` for its C API).  Those must stay absolute -- we do not host
    them -- so a hostname- or prefix-level substitution would break them.
*   **Nothing is rewritten silently-wrongly.**  Two independent traps exist, described on
    :func:`_rewrite_text`, and beyond them the script *fails* rather than guesses: an expected URL that
    no longer appears, or a new ``qiskit-fermions`` URL nobody has mapped, is an error.  There are no
    unit tests for this script; those two guards are what stands in for them, so please do not weaken
    them.

The rewrite is one-way and cannot be undone by substituting the targets back: ``release-notes.html`` is
both a target here *and* a link Sphinx already emits natively on every page, because ``docs/index.rst``
spells that toctree entry relatively.  Nothing in the output distinguishes "was absolute, got rewritten"
from "was always relative", so an inverse pass turns all ~139 of the native links into platform URLs.
To get a pristine tree back, rebuild it.

Usage:
    python tools/rewrite_docs_links.py docs/_build/html
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

#: Absolute URL -> path of the equivalent page, relative to the root of the built documentation.
#:
#: Keys must be complete URLs as they appear in the HTML.  When adding an entry, remember that the
#: target has to exist in the build: ``docs/pydoc/index.rst`` and ``docs/cdoc/index.rst`` are
#: ``:orphan:`` (built, but deliberately out of the toctree) precisely so these two do.
URL_MAP = {
    "https://quantum.cloud.ibm.com/docs/api/qiskit-fermions": "pydoc/index.html",
    "https://quantum.cloud.ibm.com/docs/api/qiskit-fermions-c": "cdoc/index.html",
    "https://quantum.cloud.ibm.com/docs/en/api/qiskit-fermions/release-notes": "release-notes.html",
}

#: Any link to this project's own documentation on the platform, however deep.  Used to catch URLs
#: that are *not* in `URL_MAP`; see `_find_unmapped`.  Scoped to ``qiskit-fermions`` so that the
#: neighbouring projects' intersphinx links, which legitimately stay absolute, do not match.
ANY_SELF_URL = re.compile(r"https://quantum\.cloud\.ibm\.com/[^\"'<>\s]*qiskit-fermions[^\"'<>\s]*")

#: Characters that may legally follow a URL in HTML.  A URL continuing past one of our keys with
#: anything else is a *different*, deeper URL that this script must not touch; see `_rewrite_text`.
TERMINATORS = "\"'<> \t\r\n)"


def _relative_prefix(html_file: pathlib.Path, root: pathlib.Path) -> str:
    """Return the ``../`` prefix that walks from a file's directory back to the build root.

    Args:
        html_file: The file being rewritten.
        root: Root of the built documentation.

    Returns:
        One ``../`` per directory level, so the empty string for a file at the root.
    """
    return "../" * len(html_file.relative_to(root).parent.parts)


def _rewrite_text(text: str, prefix: str) -> tuple[str, dict[str, int]]:
    """Replace every complete mapped URL in ``text`` with its local equivalent.

    Two distinct traps are avoided here, each of which produces plausible-looking but broken HTML:

    1.  **Key-prefix collision.**  ``.../api/qiskit-fermions`` is a strict prefix of
        ``.../api/qiskit-fermions-c``, so replacing in dictionary order turns the C link into
        ``cdoc/index.html`` only by luck -- get the order wrong and it becomes
        ``pydoc/index.html-c``.  Iterating longest-key-first makes the order explicit.
    2.  **URLs continuing past a key.**  Longest-first alone is not enough: a deeper URL such as
        ``.../api/qiskit-fermions/operators`` starts with a key, so a plain replacement would rewrite
        its head and leave the tail dangling as ``pydoc/index.html/operators``.  Requiring the next
        character to be a terminator (or end-of-string) means only whole URLs match.

    Args:
        text: File contents.
        prefix: ``../`` prefix for this file's depth, from `_relative_prefix`.

    Returns:
        The rewritten text, and a count of replacements per URL.
    """
    counts = {url: 0 for url in URL_MAP}
    # Longest first: see trap 1.
    for url in sorted(URL_MAP, key=len, reverse=True):
        target = prefix + URL_MAP[url]
        out = []
        position = 0
        while (found := text.find(url, position)) != -1:
            end = found + len(url)
            # See trap 2: only a whole URL is a match.
            if end == len(text) or text[end] in TERMINATORS:
                out.append(text[position:found])
                out.append(target)
                counts[url] += 1
            else:
                out.append(text[position:end])
            position = end
        out.append(text[position:])
        text = "".join(out)
    return text, counts


def _find_unmapped(text: str) -> set[str]:
    """Return links to this project's own platform docs that `URL_MAP` does not cover.

    This is the guard against a *new* link going unnoticed.  Because `_rewrite_text` only touches
    complete mapped URLs, a deep link such as ``.../api/qiskit-fermions/operators`` would otherwise be
    left pointing at the platform, and the "expected URL vanished" check would not notice because the
    known URLs still match.  A preview would then quietly navigate the reader away from itself.

    Args:
        text: File contents.

    Returns:
        The offending URLs, empty when everything found is mapped.
    """
    return {match.group(0) for match in ANY_SELF_URL.finditer(text)} - set(URL_MAP)


def rewrite_tree(root: pathlib.Path) -> int:
    """Rewrite every HTML file under ``root`` in place.

    Only ``*.html`` is touched.  In particular ``_sources/*.txt`` is left alone: that is Sphinx's
    verbatim copy of the reStructuredText, shown by "view page source", and it should keep matching
    the real source.

    Args:
        root: Root of the built documentation.

    Returns:
        A process exit status: 0 on success, 1 if any URL is unaccounted for.
    """
    html_files = sorted(root.rglob("*.html"))

    # Validate the whole tree before writing any of it.  Rewriting as we go would leave a
    # half-rewritten tree behind on failure, and that state is actively misleading: the URLs already
    # replaced are gone, so a re-run reports "expected URLs were never found" (the wrong error) instead
    # of the unmapped link that actually stopped us.  Two passes over 25 MB is a cheap price for
    # "either fully rewritten or untouched".
    unmapped: dict[str, set[str]] = {}
    for html_file in html_files:
        if found := _find_unmapped(html_file.read_text(encoding="utf-8")):
            unmapped[str(html_file.relative_to(root))] = found

    if unmapped:
        print(
            "error: found links to this project's documentation on the IBM Quantum Platform that are"
            " not in URL_MAP, so they would stay absolute and link out of a standalone build.\n"
            "Add each one to URL_MAP in this script, mapping it to the corresponding page in the"
            " build:",
            file=sys.stderr,
        )
        for path, urls in sorted(unmapped.items()):
            for url in sorted(urls):
                print(f"  {path}: {url}", file=sys.stderr)
        print("Nothing was modified.", file=sys.stderr)
        return 1

    totals = {url: 0 for url in URL_MAP}
    changed = 0
    for html_file in html_files:
        original = html_file.read_text(encoding="utf-8")
        rewritten, counts = _rewrite_text(original, _relative_prefix(html_file, root))
        for url, count in counts.items():
            totals[url] += count
        if rewritten != original:
            html_file.write_text(rewritten, encoding="utf-8")
            changed += 1

    if missing := [url for url, count in totals.items() if count == 0]:
        print(
            "error: expected URLs were never found, so this script did nothing for them.  Either"
            " docs/index.rst no longer links to them (remove them from URL_MAP) or the rewrite is"
            " broken:",
            file=sys.stderr,
        )
        for url in missing:
            print(f"  {url}", file=sys.stderr)
        return 1

    print(f"Rewrote {changed} file(s) under {root}:")
    for url, count in sorted(totals.items()):
        print(f"  {count:5d}  {url}  ->  {URL_MAP[url]}")
    return 0


def main() -> int:
    """Parse arguments and rewrite the requested tree.

    Returns:
        A process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        type=pathlib.Path,
        help="root of the built documentation, e.g. docs/_build/html",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"error: not a directory: {args.root}", file=sys.stderr)
        return 1
    return rewrite_tree(args.root)


if __name__ == "__main__":
    sys.exit(main())
