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

"""Tests for filtering operator terms by their number of unique modes."""

from pathlib import Path

from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.library import FCIDump
from qiskit_fermions.operators.terms.filtering import filter_terms_by_num_unique_modes


def _num_unique_modes(term):
    return len({idx for _, idx in term})


def test_filter_drops_constant_and_number_operators():
    op = FermionOperator.from_dict(
        {
            (): 1.0,  # 0 unique modes (constant)
            ((True, 0), (False, 0)): 2.0,  # 1 unique mode (number operator)
            ((True, 0), (False, 1)): 3.0,  # 2 unique modes (hopping)
        }
    )

    res = filter_terms_by_num_unique_modes(op, 2)
    assert res is None, "The function mutates in place and returns None."

    terms = list(op.iter_terms())
    assert len(terms) == 1, "Only the 2-unique-mode term should survive."
    assert all(_num_unique_modes(term) >= 2 for term, _ in terms)


def test_filter_zero_threshold_keeps_everything():
    data = {(): 1.0, ((True, 0), (False, 0)): 2.0}
    op = FermionOperator.from_dict(data)
    filter_terms_by_num_unique_modes(op, 0)
    assert op.equiv(FermionOperator.from_dict(data))


def test_filter_electronic_structure_hamiltonian():
    file_path = Path(__file__).parent / "../../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))

    op = FermionOperator.from_fcidump(fcidump).normal_ordered().simplify(atol=1e-16)
    num_terms_before = len(list(op.iter_terms()))

    filter_terms_by_num_unique_modes(op, 2)

    terms = list(op.iter_terms())
    # every surviving term acts on at least two unique modes
    assert all(_num_unique_modes(term) >= 2 for term, _ in terms)
    # the constant offset and the number operators have been removed
    assert len(terms) < num_terms_before
