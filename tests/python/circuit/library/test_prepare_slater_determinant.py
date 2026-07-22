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

"""Structural tests for the PrepareSlaterDeterminant gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)

from ...utils import random_unitary


def test_prepare_slater_determinant_gate():
    """The gate stores its occupation and rotation and reports the right mode count."""
    occupation = [True, True, False, False]
    rotation = random_unitary(4, seed=0)
    gate = PrepareSlaterDeterminant(occupation, rotation)

    assert gate.num_modes == 4
    np.testing.assert_array_equal(gate.occupation, np.asarray(occupation, dtype=bool))
    np.testing.assert_allclose(gate.rotation_unitary, rotation)

    circ = FermionicCircuit(4)
    circ.append(gate, circ.modes)
    assert circ.count_ops() == {"PrepareSlaterDeterminant": 1}


def test_prepare_slater_determinant_rejects_shape_mismatch():
    """A rotation whose dimension disagrees with the occupation length is rejected."""
    with pytest.raises(ValueError, match="to match the occupation length"):
        PrepareSlaterDeterminant([True, False, False], random_unitary(4, seed=1))


def test_prepare_slater_determinant_definition():
    """The gate's definition is an InitializeModes validator followed by the OrbitalRotation."""
    occupation = [True, False, True]
    rotation = random_unitary(3, seed=2)
    definition = PrepareSlaterDeterminant(occupation, rotation)._build_definition()

    ops = list(definition._inner.data)
    assert [type(instr.operation) for instr in ops] == [InitializeModes, OrbitalRotation]
    np.testing.assert_array_equal(ops[0].operation.occupation, np.asarray(occupation, dtype=bool))
    np.testing.assert_allclose(ops[1].operation.rotation_unitary, rotation)
    # every sub-instruction acts on the full register
    assert all(len(instr.qubits) == 3 for instr in ops)
