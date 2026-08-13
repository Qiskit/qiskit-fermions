#!/bin/sh
# This code is a Qiskit project.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#
# Prepare an Ubuntu CI machine for running 'make docs'.  Assumes that Python is available.

set -e
if [ -z $GITHUB_TOKEN ]; then
    echo "No github token provided"
    exit 1
fi

python -m pip install --upgrade pip setuptools setuptools_rust wheel tox

sudo apt-get update
sudo apt-get install -y graphviz pandoc

# Keep this in step with the cache key in `.github/workflows/docs.yml`, which caches the tarball this
# downloads so that repeat runs neither re-fetch it nor depend on the release being reachable.
DOXYGEN_VERSION=1.15.0
# The release tag spells the version with underscores.  Written out rather than derived, because this
# script runs under `/bin/sh` (dash on Ubuntu), which has no `${var//./_}` substitution.
DOXYGEN_RELEASE_TAG=Release_1_15_0
DOXYGEN_TARBALL="doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz"

# Fetch the pinned release of doxygen and its linux binaries, unless a cache restored it already.
if [ -f "./$DOXYGEN_TARBALL" ]; then
    echo "Using already-present $DOXYGEN_TARBALL."
else
    wget --header "Authorization: token $GITHUB_TOKEN" \
        "https://github.com/doxygen/doxygen/releases/download/$DOXYGEN_RELEASE_TAG/$DOXYGEN_TARBALL"
fi

# The following commands install the binaries for doxygen
tar -zxf "./$DOXYGEN_TARBALL"
cd "./doxygen-${DOXYGEN_VERSION}"

# Run the remainder of the setup process
sudo make install
