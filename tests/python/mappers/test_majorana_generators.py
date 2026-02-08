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
from qiskit_fermions.mappers import map_majorana_action_generators
from qiskit_fermions.operators import MajoranaAction, MajoranaOperator


def jordan_wigner(op: MajoranaOperator, num_qubits: int) -> SparsePauliOp:
    """Custom Jordan-Wigner transformation."""

    @cache
    def map_action(mode: MajoranaAction) -> SparsePauliOp:
        idx = mode // 2
        qubits = list(range(idx + 1))
        pauli = "Y" if mode % 2 else "X"
        return SparsePauliOp.from_sparse_list(
            [("Z" * idx + pauli, qubits, 1.0)],
            num_qubits=num_qubits,
        )

    return map_majorana_action_generators(
        op,
        map_action,
        lambda: SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits=num_qubits),
    )


def test_jordan_wigner():
    op = MajoranaOperator.from_dict(
        {
            (): -0.8105479805373262,
            (1, 0): 0.17218393261915554j,
            (3, 2): -0.22575349222402474j,
            (5, 4): 0.17218393261915552j,
            (7, 6): -0.22575349222402477j,
            (3, 2, 1, 0): -0.12091263261776633,
            (5, 4, 1, 0): -0.16892753870087912,
            (5, 4, 3, 2): -0.16614543256382416,
            (6, 5, 2, 1): -0.04523279994605783,
            (6, 5, 3, 0): 0.04523279994605783,
            (7, 4, 2, 1): 0.04523279994605783,
            (7, 4, 3, 0): -0.04523279994605783,
            (7, 6, 1, 0): -0.16614543256382416,
            (7, 6, 3, 2): -0.17464343068300459,
            (7, 6, 5, 4): -0.12091263261776633,
        }
    )
    num_qubits = 4
    qop = jordan_wigner(op, num_qubits)
    assert isinstance(qop, SparsePauliOp)
    expected = SparsePauliOp.from_sparse_list(
        [
            ("", [], -0.8105479805373266),
            ("Z", [0], 0.1721839326191555),
            ("Z", [1], -0.22575349222402474),
            ("Z", [2], 0.17218393261915543),
            ("Z", [3], -0.22575349222402474),
            ("ZZ", [0, 1], 0.12091263261776633),
            ("ZZ", [0, 2], 0.16892753870087912),
            ("ZZ", [0, 3], 0.16614543256382416),
            ("YYYY", [0, 1, 2, 3], 0.04523279994605783),
            ("YYXX", [0, 1, 2, 3], 0.04523279994605783),
            ("XXYY", [0, 1, 2, 3], 0.04523279994605783),
            ("XXXX", [0, 1, 2, 3], 0.04523279994605783),
            ("ZZ", [1, 2], 0.16614543256382416),
            ("ZZ", [1, 3], 0.17464343068300459),
            ("ZZ", [2, 3], 0.12091263261776633),
        ],
        num_qubits,
    )
    diff = (qop - expected).simplify()
    assert diff == SparsePauliOp.from_sparse_list([], num_qubits)
