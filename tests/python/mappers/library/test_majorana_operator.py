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
    edge_vertex_to_majorana,
    fermion_to_majorana,
    majorana_operator,
    transfer_vertex_to_majorana,
)
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
)


def test_majorana_operator_from_fermion_operator():
    # FermionOperator._majorana_operator_ delegates to fermion_to_majorana
    fer_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    converted = majorana_operator(fer_op)
    direct = fermion_to_majorana(fer_op)
    assert isinstance(converted, MajoranaOperator)
    assert converted.normal_ordered().equiv(direct.normal_ordered())


def test_majorana_operator_from_edge_vertex_operator():
    # EdgeVertexOperator._majorana_operator_ delegates to edge_vertex_to_majorana
    inter_op = EdgeVertexOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0})
    converted = majorana_operator(inter_op)
    direct = edge_vertex_to_majorana(inter_op)
    assert isinstance(converted, MajoranaOperator)
    assert converted.equiv(direct)


def test_majorana_operator_from_transfer_vertex_operator():
    # TransferVertexOperator._majorana_operator_ delegates to transfer_vertex_to_majorana
    inter_op = TransferVertexOperator.from_dict({((0, 0),): 1.0, ((1, 2),): 2.0, ((2, 1),): -2})
    converted = majorana_operator(inter_op)
    direct = transfer_vertex_to_majorana(inter_op)
    assert isinstance(converted, MajoranaOperator)
    assert converted.equiv(direct)


def test_majorana_operator_dispatches_via_protocol_not_type():
    # the wrapper must call `_majorana_operator_` on whatever it is given, independent of the
    # object's concrete type -- a minimal duck-typed object satisfying the protocol must work too
    class _Fake:
        def _majorana_operator_(self) -> MajoranaOperator:
            return MajoranaOperator.from_dict({(2, 3): 3})

    converted = majorana_operator(_Fake())
    assert converted == MajoranaOperator.from_dict({(2, 3): 3})
