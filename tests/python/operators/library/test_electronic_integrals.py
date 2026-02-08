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

import numpy as np
from qiskit_fermions.operators import FermionOperator


def test_from_1body_tril_spin_sym():
    norb = 2
    one_body_a = np.arange(1, 4, dtype=float)
    op = FermionOperator.from_1body_tril_spin_sym(one_body_a, norb)
    expected = FermionOperator.from_dict(
        {
            ((True, 0), (False, 0)): 1.0,
            ((True, 1), (False, 0)): 2.0,
            ((True, 0), (False, 1)): 2.0,
            ((True, 1), (False, 1)): 3.0,
            ((True, 2), (False, 2)): 1.0,
            ((True, 3), (False, 2)): 2.0,
            ((True, 2), (False, 3)): 2.0,
            ((True, 3), (False, 3)): 3.0,
        }
    )
    assert op.equiv(expected)


def test_from_1body_tril_spin():
    norb = 2
    one_body_a = np.arange(1, 4, dtype=float)
    one_body_b = np.arange(-1, -4, -1, dtype=float)
    op = FermionOperator.from_1body_tril_spin(one_body_a, one_body_b, norb)
    expected = FermionOperator.from_dict(
        {
            ((True, 0), (False, 0)): 1.0,
            ((True, 1), (False, 0)): 2.0,
            ((True, 0), (False, 1)): 2.0,
            ((True, 1), (False, 1)): 3.0,
            ((True, 2), (False, 2)): -1.0,
            ((True, 3), (False, 2)): -2.0,
            ((True, 2), (False, 3)): -2.0,
            ((True, 3), (False, 3)): -3.0,
        }
    )
    assert op.equiv(expected)


def test_from_2body_tril_spin_sym():
    norb = 2
    two_body_aa = np.arange(1, 7, dtype=float)
    op = FermionOperator.from_2body_tril_spin_sym(two_body_aa, norb)
    expected = FermionOperator.from_dict(
        {
            ((True, 0), (True, 0), (False, 0), (False, 0)): 0.5,
            ((True, 0), (True, 0), (False, 0), (False, 1)): 1.0,
            ((True, 0), (True, 0), (False, 1), (False, 0)): 1.0,
            ((True, 0), (True, 0), (False, 1), (False, 1)): 1.5,
            ((True, 0), (True, 1), (False, 0), (False, 0)): 1.0,
            ((True, 0), (True, 1), (False, 0), (False, 1)): 1.5,
            ((True, 0), (True, 1), (False, 1), (False, 0)): 2.0,
            ((True, 0), (True, 1), (False, 1), (False, 1)): 2.5,
            ((True, 0), (True, 2), (False, 2), (False, 0)): 0.5,
            ((True, 0), (True, 2), (False, 2), (False, 1)): 1.0,
            ((True, 0), (True, 2), (False, 3), (False, 0)): 1.0,
            ((True, 0), (True, 2), (False, 3), (False, 1)): 1.5,
            ((True, 0), (True, 3), (False, 2), (False, 0)): 1.0,
            ((True, 0), (True, 3), (False, 2), (False, 1)): 1.5,
            ((True, 0), (True, 3), (False, 3), (False, 0)): 2.0,
            ((True, 0), (True, 3), (False, 3), (False, 1)): 2.5,
            ((True, 1), (True, 0), (False, 0), (False, 0)): 1.0,
            ((True, 1), (True, 0), (False, 0), (False, 1)): 2.0,
            ((True, 1), (True, 0), (False, 1), (False, 0)): 1.5,
            ((True, 1), (True, 0), (False, 1), (False, 1)): 2.5,
            ((True, 1), (True, 1), (False, 0), (False, 0)): 1.5,
            ((True, 1), (True, 1), (False, 0), (False, 1)): 2.5,
            ((True, 1), (True, 1), (False, 1), (False, 0)): 2.5,
            ((True, 1), (True, 1), (False, 1), (False, 1)): 3.0,
            ((True, 1), (True, 2), (False, 2), (False, 0)): 1.0,
            ((True, 1), (True, 2), (False, 2), (False, 1)): 2.0,
            ((True, 1), (True, 2), (False, 3), (False, 0)): 1.5,
            ((True, 1), (True, 2), (False, 3), (False, 1)): 2.5,
            ((True, 1), (True, 3), (False, 2), (False, 0)): 1.5,
            ((True, 1), (True, 3), (False, 2), (False, 1)): 2.5,
            ((True, 1), (True, 3), (False, 3), (False, 0)): 2.5,
            ((True, 1), (True, 3), (False, 3), (False, 1)): 3.0,
            ((True, 2), (True, 0), (False, 0), (False, 2)): 0.5,
            ((True, 2), (True, 0), (False, 0), (False, 3)): 1.0,
            ((True, 2), (True, 0), (False, 1), (False, 2)): 1.0,
            ((True, 2), (True, 0), (False, 1), (False, 3)): 1.5,
            ((True, 2), (True, 1), (False, 0), (False, 2)): 1.0,
            ((True, 2), (True, 1), (False, 0), (False, 3)): 1.5,
            ((True, 2), (True, 1), (False, 1), (False, 2)): 2.0,
            ((True, 2), (True, 1), (False, 1), (False, 3)): 2.5,
            ((True, 2), (True, 2), (False, 2), (False, 2)): 0.5,
            ((True, 2), (True, 2), (False, 2), (False, 3)): 1.0,
            ((True, 2), (True, 2), (False, 3), (False, 2)): 1.0,
            ((True, 2), (True, 2), (False, 3), (False, 3)): 1.5,
            ((True, 2), (True, 3), (False, 2), (False, 2)): 1.0,
            ((True, 2), (True, 3), (False, 2), (False, 3)): 1.5,
            ((True, 2), (True, 3), (False, 3), (False, 2)): 2.0,
            ((True, 2), (True, 3), (False, 3), (False, 3)): 2.5,
            ((True, 3), (True, 0), (False, 0), (False, 2)): 1.0,
            ((True, 3), (True, 0), (False, 0), (False, 3)): 2.0,
            ((True, 3), (True, 0), (False, 1), (False, 2)): 1.5,
            ((True, 3), (True, 0), (False, 1), (False, 3)): 2.5,
            ((True, 3), (True, 1), (False, 0), (False, 2)): 1.5,
            ((True, 3), (True, 1), (False, 0), (False, 3)): 2.5,
            ((True, 3), (True, 1), (False, 1), (False, 2)): 2.5,
            ((True, 3), (True, 1), (False, 1), (False, 3)): 3.0,
            ((True, 3), (True, 2), (False, 2), (False, 2)): 1.0,
            ((True, 3), (True, 2), (False, 2), (False, 3)): 2.0,
            ((True, 3), (True, 2), (False, 3), (False, 2)): 1.5,
            ((True, 3), (True, 2), (False, 3), (False, 3)): 2.5,
            ((True, 3), (True, 3), (False, 2), (False, 2)): 1.5,
            ((True, 3), (True, 3), (False, 2), (False, 3)): 2.5,
            ((True, 3), (True, 3), (False, 3), (False, 2)): 2.5,
            ((True, 3), (True, 3), (False, 3), (False, 3)): 3.0,
        }
    )
    assert op.equiv(expected)


def test_from_2body_tril_spin():
    norb = 2
    two_body_aa = np.arange(1, 7, dtype=float)
    two_body_ab = np.arange(11, 20, dtype=float)
    two_body_bb = np.arange(-1, -7, -1, dtype=float)
    op = FermionOperator.from_2body_tril_spin(two_body_aa, two_body_ab, two_body_bb, norb)
    expected = FermionOperator.from_dict(
        {
            ((True, 0), (True, 0), (False, 0), (False, 0)): 0.5,
            ((True, 0), (True, 0), (False, 0), (False, 1)): 1.0,
            ((True, 0), (True, 0), (False, 1), (False, 0)): 1.0,
            ((True, 0), (True, 0), (False, 1), (False, 1)): 1.5,
            ((True, 0), (True, 1), (False, 0), (False, 0)): 1.0,
            ((True, 0), (True, 1), (False, 0), (False, 1)): 1.5,
            ((True, 0), (True, 1), (False, 1), (False, 0)): 2.0,
            ((True, 0), (True, 1), (False, 1), (False, 1)): 2.5,
            ((True, 0), (True, 2), (False, 2), (False, 0)): 5.5,
            ((True, 0), (True, 2), (False, 2), (False, 1)): 7.0,
            ((True, 0), (True, 2), (False, 3), (False, 0)): 6.0,
            ((True, 0), (True, 2), (False, 3), (False, 1)): 7.5,
            ((True, 0), (True, 3), (False, 2), (False, 0)): 6.0,
            ((True, 0), (True, 3), (False, 2), (False, 1)): 7.5,
            ((True, 0), (True, 3), (False, 3), (False, 0)): 6.5,
            ((True, 0), (True, 3), (False, 3), (False, 1)): 8.0,
            ((True, 1), (True, 0), (False, 0), (False, 0)): 1.0,
            ((True, 1), (True, 0), (False, 0), (False, 1)): 2.0,
            ((True, 1), (True, 0), (False, 1), (False, 0)): 1.5,
            ((True, 1), (True, 0), (False, 1), (False, 1)): 2.5,
            ((True, 1), (True, 1), (False, 0), (False, 0)): 1.5,
            ((True, 1), (True, 1), (False, 0), (False, 1)): 2.5,
            ((True, 1), (True, 1), (False, 1), (False, 0)): 2.5,
            ((True, 1), (True, 1), (False, 1), (False, 1)): 3.0,
            ((True, 1), (True, 2), (False, 2), (False, 0)): 7.0,
            ((True, 1), (True, 2), (False, 2), (False, 1)): 8.5,
            ((True, 1), (True, 2), (False, 3), (False, 0)): 7.5,
            ((True, 1), (True, 2), (False, 3), (False, 1)): 9.0,
            ((True, 1), (True, 3), (False, 2), (False, 0)): 7.5,
            ((True, 1), (True, 3), (False, 2), (False, 1)): 9.0,
            ((True, 1), (True, 3), (False, 3), (False, 0)): 8.0,
            ((True, 1), (True, 3), (False, 3), (False, 1)): 9.5,
            ((True, 2), (True, 0), (False, 0), (False, 2)): 5.5,
            ((True, 2), (True, 0), (False, 0), (False, 3)): 6.0,
            ((True, 2), (True, 0), (False, 1), (False, 2)): 7.0,
            ((True, 2), (True, 0), (False, 1), (False, 3)): 7.5,
            ((True, 2), (True, 1), (False, 0), (False, 2)): 7.0,
            ((True, 2), (True, 1), (False, 0), (False, 3)): 7.5,
            ((True, 2), (True, 1), (False, 1), (False, 2)): 8.5,
            ((True, 2), (True, 1), (False, 1), (False, 3)): 9.0,
            ((True, 2), (True, 2), (False, 2), (False, 2)): -0.5,
            ((True, 2), (True, 2), (False, 2), (False, 3)): -1.0,
            ((True, 2), (True, 2), (False, 3), (False, 2)): -1.0,
            ((True, 2), (True, 2), (False, 3), (False, 3)): -1.5,
            ((True, 2), (True, 3), (False, 2), (False, 2)): -1.0,
            ((True, 2), (True, 3), (False, 2), (False, 3)): -1.5,
            ((True, 2), (True, 3), (False, 3), (False, 2)): -2.0,
            ((True, 2), (True, 3), (False, 3), (False, 3)): -2.5,
            ((True, 3), (True, 0), (False, 0), (False, 2)): 6.0,
            ((True, 3), (True, 0), (False, 0), (False, 3)): 6.5,
            ((True, 3), (True, 0), (False, 1), (False, 2)): 7.5,
            ((True, 3), (True, 0), (False, 1), (False, 3)): 8.0,
            ((True, 3), (True, 1), (False, 0), (False, 2)): 7.5,
            ((True, 3), (True, 1), (False, 0), (False, 3)): 8.0,
            ((True, 3), (True, 1), (False, 1), (False, 2)): 9.0,
            ((True, 3), (True, 1), (False, 1), (False, 3)): 9.5,
            ((True, 3), (True, 2), (False, 2), (False, 2)): -1.0,
            ((True, 3), (True, 2), (False, 2), (False, 3)): -2.0,
            ((True, 3), (True, 2), (False, 3), (False, 2)): -1.5,
            ((True, 3), (True, 2), (False, 3), (False, 3)): -2.5,
            ((True, 3), (True, 3), (False, 2), (False, 2)): -1.5,
            ((True, 3), (True, 3), (False, 2), (False, 3)): -2.5,
            ((True, 3), (True, 3), (False, 3), (False, 2)): -2.5,
            ((True, 3), (True, 3), (False, 3), (False, 3)): -3.0,
        }
    )
    assert op.equiv(expected)
