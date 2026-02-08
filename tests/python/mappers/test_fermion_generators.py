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
from qiskit_fermions.mappers import map_fermion_action_generators
from qiskit_fermions.operators import FermionAction, FermionOperator


def jordan_wigner(op: FermionOperator, layout: dict[int, int]) -> SparsePauliOp:
    """Custom Jordan-Wigner transformation."""
    num_qubits = max(layout.values()) + 1

    @cache
    def map_action(action: FermionAction) -> SparsePauliOp:
        act, idx = action
        idx = layout[idx]
        qubits = list(range(idx + 1))
        return SparsePauliOp.from_sparse_list(
            [
                ("Z" * idx + "X", qubits, 0.5),
                ("Z" * idx + "Y", qubits, -0.5j if act else 0.5j),
            ],
            num_qubits=num_qubits,
        )

    return map_fermion_action_generators(
        op,
        map_action,
        lambda: SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits=num_qubits),
    )


def test_jordan_wigner():
    op = FermionOperator.from_dict(
        {
            ((True, 0), (True, 1), (False, 0), (False, 1)): -0.4836505304710653,
            ((True, 0), (True, 2), (False, 0), (False, 2)): -0.6757101548035165,
            ((True, 0), (True, 3), (False, 0), (False, 3)): -0.6645817302552967,
            ((True, 0), (True, 2), (False, 1), (False, 3)): -0.18093119978423133,
            ((True, 0), (True, 3), (False, 1), (False, 2)): -0.18093119978423133,
            ((True, 1), (True, 2), (False, 0), (False, 3)): -0.18093119978423133,
            ((True, 1), (True, 3), (False, 0), (False, 2)): -0.18093119978423133,
            ((True, 1), (True, 2), (False, 1), (False, 2)): -0.6645817302552967,
            ((True, 1), (True, 3), (False, 1), (False, 3)): -0.6985737227320183,
            ((True, 2), (True, 3), (False, 2), (False, 3)): -0.4836505304710653,
            ((True, 0), (False, 0)): -1.2563390730032502,
            ((True, 1), (False, 1)): -0.4718960072811406,
            ((True, 2), (False, 2)): -1.2563390730032502,
            ((True, 3), (False, 3)): -0.4718960072811406,
        }
    )
    num_qubits = 14
    layout = {0: 13, 1: 12, 2: 11, 3: 10}
    qop = jordan_wigner(op, layout)
    assert isinstance(qop, SparsePauliOp)
    expected = SparsePauliOp.from_sparse_list(
        [
            ("", [], -0.8105479805373266),
            ("Z", [13], 0.1721839326191555),
            ("Z", [12], -0.22575349222402474),
            ("Z", [11], 0.17218393261915543),
            ("Z", [10], -0.22575349222402474),
            ("ZZ", [13, 12], 0.12091263261776633),
            ("ZZ", [13, 11], 0.16892753870087912),
            ("ZZ", [13, 10], 0.16614543256382416),
            ("YYYY", [13, 12, 11, 10], 0.04523279994605783),
            ("YYXX", [13, 12, 11, 10], 0.04523279994605783),
            ("XXYY", [13, 12, 11, 10], 0.04523279994605783),
            ("XXXX", [13, 12, 11, 10], 0.04523279994605783),
            ("ZZ", [12, 11], 0.16614543256382416),
            ("ZZ", [12, 10], 0.17464343068300459),
            ("ZZ", [11, 10], 0.12091263261776633),
        ],
        num_qubits,
    )
    diff = (qop - expected).simplify()
    assert diff == SparsePauliOp.from_sparse_list([], num_qubits)
