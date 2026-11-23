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

"""Set the project version consistently across Python and Rust.

The version is provided in :pep:`440` form (e.g. ``0.2.0``, ``0.2.0.dev0``, ``1.0.0rc1``).  It is
written verbatim to ``python/qiskit_fermions/VERSION.txt`` (read at import time by ``version.py``)
and to the ``[project]`` version in ``pyproject.toml`` (the package metadata used by the build
backend), and, converted to Cargo's SemVer pre-release syntax, to the ``[workspace.package]``
version in ``Cargo.toml``.

PEP 440 and SemVer spell pre-releases differently, so we translate the separator:

    PEP 440         SemVer (Cargo)
    0.2.0.dev0  ->  0.2.0-dev0
    1.0.0rc1    ->  1.0.0-rc1
    1.0.0a2     ->  1.0.0-a2
    0.2.0       ->  0.2.0        (final release, unchanged)

Run via ``make version-bump VERSION=<pep440-version>`` so that ``Cargo.lock`` is regenerated too.
"""

import argparse
import os
import re
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION_TXT = os.path.join(ROOT_DIR, "python", "qiskit_fermions", "VERSION.txt")
CARGO_TOML = os.path.join(ROOT_DIR, "Cargo.toml")
PYPROJECT_TOML = os.path.join(ROOT_DIR, "pyproject.toml")

# A deliberately strict subset of PEP 440: release segment with an optional pre-release
# (a/b/rc) and/or dev segment.  We do not support epochs, post-releases, or local versions here
# because a bump target should only ever set clean release / pre-release / dev numbers; the local
# ``+<sha>`` identifier is added automatically at import time by ``version.py``.
_PEP440 = re.compile(
    r"^(?P<release>\d+(?:\.\d+)*)"
    r"(?P<pre>(?:a|b|rc)\d+)?"
    r"(?:\.(?P<dev>dev\d+))?$"
)


def pep440_to_semver(version: str) -> str:
    """Convert a PEP 440 version string to Cargo's SemVer pre-release syntax.

    Args:
        version: A PEP 440 version, already validated by :func:`validate`.

    Returns:
        The equivalent SemVer string, with any pre-release/dev segment moved behind a ``-``.
    """
    match = _PEP440.match(version)
    assert match is not None  # guaranteed by validate()
    release = match.group("release")
    pre = match.group("pre")
    dev = match.group("dev")
    suffix = "".join(part for part in (pre, dev) if part)
    return f"{release}-{suffix}" if suffix else release


def validate(version: str) -> str:
    """Validate that ``version`` is an accepted PEP 440 string, returning it unchanged."""
    if _PEP440.match(version) is None:
        raise SystemExit(
            f"error: {version!r} is not an accepted PEP 440 version.\n"
            "       Expected e.g. '0.2.0', '0.2.0.dev0', '1.0.0rc1'."
        )
    return version


def write_version_txt(version: str) -> None:
    """Write the PEP 440 version to ``VERSION.txt`` (trailing newline, matching the committed file)."""
    with open(VERSION_TXT, "w") as handle:
        handle.write(version + "\n")


def write_toml_version(path: str, table: str, version: str) -> None:
    """Rewrite the single ``version`` line of the TOML ``table`` in the file at ``path``.

    Both manifests are edited line by line rather than round-tripped through a TOML library, so
    that their comments and formatting survive untouched.
    """
    with open(path) as handle:
        lines = handle.readlines()

    in_table = False
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            in_table = stripped == f"[{table}]"
            continue
        if in_table and re.match(r"version\s*=", stripped):
            lines[index] = f'version = "{version}"\n'
            replaced = True
            break

    if not replaced:
        raise SystemExit(f"error: could not find [{table}] version in {os.path.basename(path)}")

    with open(path, "w") as handle:
        handle.writelines(lines)


def main() -> int:
    """Entry point: parse the requested version and update all manifests."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the new PEP 440 version, e.g. 0.2.0.dev0")
    args = parser.parse_args()

    version = validate(args.version)
    semver = pep440_to_semver(version)

    write_version_txt(version)
    write_toml_version(PYPROJECT_TOML, "project", version)
    write_toml_version(CARGO_TOML, "workspace.package", semver)

    print(f"VERSION.txt    -> {version}")
    print(f"pyproject.toml -> {version}")
    print(f"Cargo.toml     -> {semver}")
    print("Remember to run the build (e.g. `make pyext-dev`) to refresh Cargo.lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
