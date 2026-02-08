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

from qiskit.quantum_info import SparseObservable
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator


def test_jordan_wigner():
    num_qubits = 4
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
    qop = jordan_wigner(op, num_qubits)
    assert isinstance(qop, SparseObservable)
    expected = SparseObservable.from_sparse_list(
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
    assert diff == SparseObservable.zero(num_qubits)
