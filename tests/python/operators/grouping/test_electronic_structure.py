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

import pytest
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.grouping import group_terms_by_electronic_structure
from qiskit_fermions.operators.library import FCIDump


def test_grouping_error():
    op = FermionOperator.from_dict({((True, 0),): 1.0})
    with pytest.raises(ValueError, match="operator does not conform to an electronic structure"):
        group_terms_by_electronic_structure(op, 2)


def test_group_terms_by_electronic_structure():
    file_path = Path(__file__).parent / "../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))

    op = FermionOperator.from_fcidump(fcidump)
    assert op.groups is None, "We should not have any group indices yet!"

    normal = op.normal_ordered().simplify(atol=1e-16)

    res = group_terms_by_electronic_structure(
        normal, 2 * fcidump.norb, two_body_physicist_order=False
    )
    assert res is None, "We should not have a GroupingError here!"
    assert normal.groups is not None, "Now we should have group indices!"
    assert max(normal.groups) == 13, (
        "The number of groups we expect is 14, meaning the highest group index should be 13!"
    )
