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

"""Contains the Qiskit Fermions version.

The release number is stored in ``VERSION.txt`` next to this file, which is the single source of
truth and is committed to the repository (so a source tarball is self-contained and needs no git
history to know its version).  When building or importing from a git checkout whose ``HEAD`` is not
a tagged release, a :pep:`440` local version identifier ``+<sha7>`` is appended so that editable
installs report the exact commit they were built from.
"""

import os
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# The repository root, used only to detect a git checkout.  The package lives at
# ``<repo>/python/qiskit_fermions``, so the repo root is two directories above this file.
REPO_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))


def _minimal_ext_cmd(cmd: list[str]) -> bytes:
    # construct minimal environment
    env = {}
    for k in ["SYSTEMROOT", "PATH"]:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env["LANGUAGE"] = "C"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=REPO_DIR,
    ) as proc:
        stdout, stderr = proc.communicate()
        if proc.returncode > 0:
            error_message = stderr.strip().decode("ascii")
            raise OSError(f"Command {cmd} exited with code {proc.returncode}: {error_message}")
    return stdout


def git_version() -> str:
    """Get the current git head sha1."""
    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        git_revision = out.strip().decode("ascii")
    except OSError:
        git_revision = "Unknown"

    return git_revision


with open(os.path.join(ROOT_DIR, "VERSION.txt")) as version_file:
    VERSION = version_file.read().strip()


def get_version_info() -> str:
    """Get the full version string.

    Returns:
        The version stored in ``VERSION.txt``, with ``+<sha7>`` appended when importing from a git
        checkout whose ``HEAD`` is not a tagged release.
    """
    full_version = VERSION

    if not os.path.exists(os.path.join(REPO_DIR, ".git")):
        return full_version
    try:
        release = _minimal_ext_cmd(["git", "tag", "-l", "--points-at", "HEAD"])
    except Exception:
        return full_version
    git_revision = git_version()
    if not release:
        full_version += "+" + git_revision[:7]

    return full_version


__version__ = get_version_info()
