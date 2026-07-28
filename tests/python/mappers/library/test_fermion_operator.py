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
    edge_vertex_to_fermion,
    fermion_operator,
    majorana_to_fermion,
    transfer_vertex_to_fermion,
)
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
)


def test_fermion_operator_from_majorana_operator():
    # MajoranaOperator._fermion_operator_ delegates to majorana_to_fermion
    maj_op = MajoranaOperator.from_dict({(0, 1): 1})
    converted = fermion_operator(maj_op)
    direct = majorana_to_fermion(maj_op)
    assert isinstance(converted, FermionOperator)
    assert converted.normal_ordered().equiv(direct.normal_ordered())


def test_fermion_operator_from_edge_vertex_operator():
    # EdgeVertexOperator._fermion_operator_ delegates to edge_vertex_to_fermion
    inter_op = EdgeVertexOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0})
    converted = fermion_operator(inter_op)
    direct = edge_vertex_to_fermion(inter_op)
    assert isinstance(converted, FermionOperator)
    assert converted.equiv(direct)


def test_fermion_operator_from_transfer_vertex_operator():
    # TransferVertexOperator._fermion_operator_ delegates to transfer_vertex_to_fermion
    inter_op = TransferVertexOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0, ((2, 1),): -2})
    converted = fermion_operator(inter_op)
    direct = transfer_vertex_to_fermion(inter_op)
    assert isinstance(converted, FermionOperator)
    assert converted.equiv(direct)


def test_fermion_operator_dispatches_via_protocol_not_type():
    # the wrapper must call `_fermion_operator_` on whatever it is given, independent of the
    # object's concrete type -- a minimal duck-typed object satisfying the protocol must work too
    class _Fake:
        def _fermion_operator_(self) -> FermionOperator:
            return FermionOperator.from_dict({((True, 2), (False, 2)): 3})

    converted = fermion_operator(_Fake())
    assert converted == FermionOperator.from_dict({((True, 2), (False, 2)): 3})
