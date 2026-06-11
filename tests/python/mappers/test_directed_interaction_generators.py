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

from functools import cache

from qiskit.quantum_info import SparsePauliOp
from qiskit_fermions.mappers import map_directed_interaction_generators
from qiskit_fermions.operators import DirectedInteraction, DirectedInteractionOperator


def jordan_wigner_nearest_neighbor(
    op: DirectedInteractionOperator, num_qubits: int
) -> SparsePauliOp:
    """Custom Jordan-Wigner transformation for nearest neighbor interactions."""

    @cache
    def map_interaction(mode: DirectedInteraction) -> SparsePauliOp:
        match abs(mode[0] - mode[1]):
            case 0:
                pauli = "Z"
                qubits = [mode[0]]
                coeff = 1.0
            case 1:
                pauli = "YY"
                qubits = [mode[0], mode[1]]
                coeff = -0.5
            case _:
                raise NotImplementedError("This mapping only handles nearest neighbor interactions")

        return SparsePauliOp.from_sparse_list([(pauli, qubits, coeff)], num_qubits=num_qubits)

    return map_directed_interaction_generators(
        op,
        map_interaction,
        lambda: SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits=num_qubits),
    )


def test_jordan_wigner():
    op = DirectedInteractionOperator.from_dict(
        {
            ((0, 0),): 2.0,
            ((0, 1),): 0.5,
            ((1, 1), (1, 2)): 1.0,
        }
    )
    num_qubits = 4
    qop = jordan_wigner_nearest_neighbor(op, num_qubits)
    assert isinstance(qop, SparsePauliOp)
    expected = SparsePauliOp.from_sparse_list(
        [("Z", [0], 2), ("YY", [0, 1], -0.25), ("XY", [1, 2], 0.5j)],
        num_qubits,
    )
    diff = (qop - expected).simplify()
    assert diff == SparsePauliOp.from_sparse_list([], num_qubits)
