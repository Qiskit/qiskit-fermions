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

"""Tests for the transpiler pass utility helpers."""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import XXPlusYYGate
from qiskit.dagcircuit import DAGOpNode
from qiskit_fermions.transpiler.passes.utils import map_node_single_register


def test_map_node_single_register_maps_indices():
    """A node acting on one fermionic register maps to global indices and the qubit register."""
    freg = QuantumRegister(4, "f")
    qreg = QuantumRegister(4, "q")
    f2q_layout = {freg: qreg}

    # A two-mode gate acting on modes 1 and 3 of the single register.
    node = DAGOpNode(XXPlusYYGate(0.5), qargs=(freg[1], freg[3]), cargs=())

    freg_indices, mapped_qreg = map_node_single_register(node, f2q_layout)
    assert freg_indices == [1, 3]
    assert mapped_qreg is qreg


def test_map_node_single_register_rejects_multiple_registers():
    """A node whose modes span two fermionic registers is unsupported."""
    freg_a = QuantumRegister(2, "a")
    freg_b = QuantumRegister(2, "b")
    f2q_layout = {freg_a: QuantumRegister(2), freg_b: QuantumRegister(2)}

    # A two-qubit gate straddling the two registers.
    node = DAGOpNode(XXPlusYYGate(0.5), qargs=(freg_a[0], freg_b[1]), cargs=())

    with pytest.raises(NotImplementedError, match="spread across multiple FermionicRegister"):
        map_node_single_register(node, f2q_layout)
