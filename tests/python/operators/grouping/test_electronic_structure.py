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

"""Tests for grouping operator terms by their electronic structure."""

from pathlib import Path

from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.grouping import group_terms_by_electronic_structure
from qiskit_fermions.operators.library import FCIDump


def test_group_terms_by_electronic_structure():
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))

    op = FermionOperator.from_fcidump(fcidump)
    # no groups yet
    assert op.groups is None

    normal = op.normal_ordered().simplify(atol=1e-16)

    normal.groups = group_terms_by_electronic_structure(normal, 2 * fcidump.norb)

    assert max(normal.groups) == 16
    # TODO: update this in accordance with the equivalent test inside the core rust crate
