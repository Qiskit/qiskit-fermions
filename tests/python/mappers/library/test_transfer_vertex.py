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
    transfer_vertex_to_edge_vertex,
    transfer_vertex_to_fermion,
    transfer_vertex_to_majorana,
)
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
)


def test_transfer_vertex_to_fermion():
    inter_op = TransferVertexOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0, ((2, 1),): -2})
    fer_op = transfer_vertex_to_fermion(inter_op).simplify()
    expected = FermionOperator.from_dict(
        {
            (): 1,
            ((True, 0), (False, 0)): -2,
            ((False, 1), (False, 2)): 2,
            ((True, 1), (True, 2)): -2,
        }
    )
    assert fer_op.equiv(expected)


def test_transfer_vertex_to_majorana():
    inter_op = TransferVertexOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0, ((2, 1),): -2})
    maj_op = transfer_vertex_to_majorana(inter_op)
    expected = MajoranaOperator.from_dict({(0, 1): -1j, (3, 4): 1j, (2, 5): 1j})
    assert maj_op.equiv(expected)


def test_transfer_vertex_to_edge_vertex():
    inter_op = TransferVertexOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0, ((2, 1),): -2})
    edge_vertex_op = transfer_vertex_to_edge_vertex(inter_op)
    expected = EdgeVertexOperator.from_dict(
        {((0, 0),): 1, ((1, 1), (1, 2)): 1j, ((1, 2), (2, 2)): -1j}
    )
    assert edge_vertex_op.equiv(expected)
