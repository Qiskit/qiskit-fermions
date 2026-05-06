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

"""Mode initialization tests."""

from __future__ import annotations

import numpy as np
from qiskit_fermions.circuit import FermionCircuit
from qiskit_fermions.circuit.library import InitializeModes


def _test_initialize_modes(occupation):
    num_fermions = len(occupation)
    init = InitializeModes(occupation)
    circ = FermionCircuit(num_fermions)
    circ.append(init, circ.fermions)

    ops = circ.count_ops()
    assert ops == {"InitializeModes": 1}


def test_initialize_modes_gate(subtests):
    with subtests.test("from list"):
        _test_initialize_modes([True, False, True, False])

    with subtests.test("from tuple"):
        _test_initialize_modes((True, False, True, False))

    with subtests.test("from np.ndarray"):
        _test_initialize_modes(np.asarray([True, False, True, False], dtype=bool))
