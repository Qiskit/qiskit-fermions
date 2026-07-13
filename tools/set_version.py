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
written verbatim to ``python/qiskit_fermions/VERSION.txt`` (the single source of truth for the
Python package) and, converted to Cargo's SemVer pre-release syntax, to the ``[workspace.package]``
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


def write_cargo_toml(semver: str) -> None:
    """Rewrite the single ``[workspace.package]`` ``version`` line in ``Cargo.toml``."""
    with open(CARGO_TOML) as handle:
        lines = handle.readlines()

    in_workspace_package = False
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            in_workspace_package = stripped == "[workspace.package]"
            continue
        if in_workspace_package and re.match(r"version\s*=", stripped):
            lines[index] = f'version = "{semver}"\n'
            replaced = True
            break

    if not replaced:
        raise SystemExit("error: could not find [workspace.package] version in Cargo.toml")

    with open(CARGO_TOML, "w") as handle:
        handle.writelines(lines)


def main() -> int:
    """Entry point: parse the requested version and update both manifests."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the new PEP 440 version, e.g. 0.2.0.dev0")
    args = parser.parse_args()

    version = validate(args.version)
    semver = pep440_to_semver(version)

    write_version_txt(version)
    write_cargo_toml(semver)

    print(f"VERSION.txt -> {version}")
    print(f"Cargo.toml  -> {semver}")
    print("Remember to run the build (e.g. `make pyext-dev`) to refresh Cargo.lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
