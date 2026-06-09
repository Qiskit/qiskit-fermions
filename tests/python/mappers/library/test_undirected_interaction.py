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

from qiskit_fermions.mappers.library import (
    undirected_interaction_to_fermion,
    undirected_interaction_to_majorana,
)
from qiskit_fermions.operators import (
    FermionOperator,
    MajoranaOperator,
    UndirectedInteractionOperator,
)


def test_undirected_to_fermion():
    inter_op = UndirectedInteractionOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0})
    fer_op = undirected_interaction_to_fermion(inter_op)
    expected = FermionOperator.from_dict(
        {
            (): 1,
            ((True, 0), (False, 0)): -2,
            ((False, 1), (False, 2)): -2j,
            ((False, 1), (True, 2)): -2j,
            ((True, 1), (False, 2)): -2j,
            ((True, 1), (True, 2)): -2j,
        }
    )
    assert fer_op.equiv(expected)


def test_undirected_to_majorana():
    inter_op = UndirectedInteractionOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0})
    maj_op = undirected_interaction_to_majorana(inter_op)
    expected = MajoranaOperator.from_dict({(0, 1): -1j, (2, 4): -2j})
    assert maj_op.equiv(expected)
