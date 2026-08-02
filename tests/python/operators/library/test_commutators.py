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


def test_commutator():
    op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
    comm = commutator(op1, op2)
    canon = comm.normal_ordered()
    canon.ichop()
    assert canon.equiv(FermionOperator.zero())


def test_commutator_orientation():
    """Tests that the commutator is ``AB - BA`` and not ``BA - AB``.

    The number operator obeys :math:`[n_0, a_0] = -a_0`, whose sign flips with the operand order.
    The other tests here compare against operators that either vanish or are symmetric under that
    swap, so they cannot distinguish the two.
    """
    num_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    ann_op = FermionOperator.from_dict({((False, 0),): 1})

    comm = commutator(num_op, ann_op).normal_ordered()

    assert comm.equiv(FermionOperator.from_dict({((False, 0),): -1}))


def test_double_commutator_nested_identity():
    """Tests the double-commutator against the nested (anti-)commutators it is defined by."""
    op_a = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0, ((True, 2),): 0.5})
    op_b = FermionOperator.from_dict({((True, 1), (False, 2)): 1.0, ((False, 0),): 0.3j})
    op_c = FermionOperator.from_dict({((True, 0), (False, 2)): 0.7, ((True, 1),): 1.0 - 0.2j})

    # [[A, B], C]/2 + [A, [B, C]]/2
    expected = (
        commutator(commutator(op_a, op_b), op_c) + commutator(op_a, commutator(op_b, op_c))
    ) / 2
    actual = double_commutator(op_a, op_b, op_c, False)
    assert actual.normal_ordered().equiv(expected.normal_ordered())

    # {[A, B], C}/2 + {A, [B, C]}/2
    expected = (
        anti_commutator(commutator(op_a, op_b), op_c)
        + anti_commutator(op_a, commutator(op_b, op_c))
    ) / 2
    actual = double_commutator(op_a, op_b, op_c, True)
    assert actual.normal_ordered().equiv(expected.normal_ordered())


def test_anti_commutator():
    op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
    comm = anti_commutator(op1, op2)
    canon = comm.normal_ordered()
    canon.ichop()
    assert canon.equiv(FermionOperator.zero())


def test_double_commutator():
    op1 = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
    op2 = FermionOperator.from_dict({((False, 0), (True, 0)): 2})
    op3 = FermionOperator.from_dict({((True, 0), (False, 0)): 1, ((False, 0), (True, 0)): 2 + 0.5j})
    comm = double_commutator(op1, op2, op3, False)
    canon = comm.normal_ordered()
    canon.ichop()
    assert canon.equiv(FermionOperator.zero())
