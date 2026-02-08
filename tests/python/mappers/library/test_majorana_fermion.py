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

from qiskit_fermions.mappers.library import fermion_to_majorana, majorana_to_fermion
from qiskit_fermions.operators import FermionOperator, MajoranaOperator


def test_fermion_to_majorana():
    fer_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    maj_op = fermion_to_majorana(fer_op)
    canon = maj_op.normal_ordered()
    expected = MajoranaOperator.from_dict({(): 0.5, (1, 0): 0.5j})
    assert canon.equiv(expected)


def test_majorana_to_fermion():
    maj_op = MajoranaOperator.from_dict({(0, 1): 1})
    fer_op = majorana_to_fermion(maj_op)
    canon = fer_op.normal_ordered()
    expected = FermionOperator.from_dict({(): -1j, ((True, 0), (False, 0)): 2j})
    assert canon.equiv(expected)
