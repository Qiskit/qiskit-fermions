#!/bin/sh
# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
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

# This command fetches the latest release of doxygen and its linux binaries
wget --header "Authorization: token $GITHUB_TOKEN" \
    https://github.com/doxygen/doxygen/releases/download/Release_1_15_0/doxygen-1.15.0.linux.bin.tar.gz

# The following commands install the binaries for doxygen
tar -zxvf ./doxygen-1.15.0.linux.bin.tar.gz
cd ./doxygen-1.15.0

# Run the remainder of the setup process
sudo make install
