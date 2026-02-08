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

from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.library import (
    anti_commutator,
    commutator,
    double_commutator,
)
from qiskit_fermions.operators.library.commutators import SupportsCommutators


def test_commutator():
    op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
    assert isinstance(op1, SupportsCommutators)
    comm = commutator(op1, op2)
    canon = comm.normal_ordered()
    canon.ichop()
    assert canon.equiv(FermionOperator.zero())


def test_anti_commutator():
    op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
    assert isinstance(op1, SupportsCommutators)
    comm = anti_commutator(op1, op2)
    canon = comm.normal_ordered()
    canon.ichop()
    assert canon.equiv(FermionOperator.zero())


def test_double_commutator():
    op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
    op3 = FermionOperator.from_dict({((True, 0), (False, 0)): 1, ((False, 0), (True, 0)): 2 + 0.5j})
    assert isinstance(op1, SupportsCommutators)
    comm = double_commutator(op1, op2, op3, False)
    canon = comm.normal_ordered()
    canon.ichop()
    assert canon.equiv(FermionOperator.zero())
