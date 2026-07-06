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

"""Tests for filtering out the diagonal terms of an operator."""

from __future__ import annotations

import copy
from pathlib import Path

from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.library import FCIDump
from qiskit_fermions.operators.terms.filtering import filter_diagonal_terms


def _is_diagonal(term):
    creations = sorted(idx for action, idx in term if action)
    annihilations = sorted(idx for action, idx in term if not action)
    return creations == annihilations


def test_filter_drops_constant_number_and_products():
    op = FermionOperator.from_dict(
        {
            (): 1.0,  # constant (diagonal)
            ((True, 0), (False, 0)): 2.0,  # n_0 (diagonal)
            ((True, 0), (True, 1), (False, 1), (False, 0)): 3.0,  # n_0 n_1 (diagonal)
            ((True, 0), (False, 1)): 4.0,  # a†_0 a_1 (off-diagonal)
        }
    )

    res = filter_diagonal_terms(op)
    assert res is None, "The function mutates in place and returns None."

    terms = [term for term, _ in op.iter_terms()]
    assert terms == [[(True, 0), (False, 1)]], "Only the off-diagonal term should survive."


def test_filter_keeps_off_diagonal_terms():
    data = {((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 2.0}
    op = FermionOperator.from_dict(data)
    filter_diagonal_terms(op)
    assert op.equiv(FermionOperator.from_dict(data))


def test_filter_does_not_mutate_a_deepcopy_source():
    op = FermionOperator.from_dict(
        {(): 1.0, ((True, 0), (False, 1)): 2.0},
    )
    other = copy.deepcopy(op)
    filter_diagonal_terms(other)
    # the original retains both terms (incl. the diagonal constant)
    assert len(list(op.iter_terms())) == 2
    assert len(list(other.iter_terms())) == 1


def test_filter_electronic_structure_hamiltonian():
    file_path = Path(__file__).parent / "../../../../h2.fcidump"
    fcidump = FCIDump.from_file(str(file_path))

    op = FermionOperator.from_fcidump(fcidump).normal_ordered().simplify(atol=1e-16)
    num_terms_before = len(list(op.iter_terms()))

    filter_diagonal_terms(op)

    terms = [term for term, _ in op.iter_terms()]
    # every surviving term is off-diagonal
    assert all(not _is_diagonal(term) for term in terms)
    # the constant offset and the number operators have been removed
    assert len(terms) < num_terms_before
