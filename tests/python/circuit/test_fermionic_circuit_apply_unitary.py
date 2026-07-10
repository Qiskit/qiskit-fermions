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

"""Tests for applying a FermionicCircuit to an ffsim state vector (SupportsApplyUnitary)."""

from __future__ import annotations

import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.fermionic_gate import FermionicGate

ffsim = pytest.importorskip("ffsim")


def test_apply_unitary_raises_when_instruction_lacks_protocol():
    """A circuit instruction that does not implement the protocol raises TypeError."""
    norb = 2
    nelec = (1, 1)

    # a bare FermionicGate implements neither _apply_unitary_placed_ nor _apply_unitary_
    circ = FermionicCircuit(2 * norb)
    circ.append(FermionicGate("dummy", 2), [circ.modes[0], circ.modes[1]])

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    with pytest.raises(TypeError, match="does not implement"):
        circ._apply_unitary_(vec0, norb, nelec, copy=True)


def test_apply_unitary_raises_when_instruction_declines():
    """A circuit instruction returning NotImplemented raises ValueError."""

    class _DecliningGate(FermionicGate):
        """A gate that implements the protocol but declines to act."""

        def __init__(self):
            super().__init__("declines", 2)

        def _apply_unitary_(self, vec, norb, nelec, copy):
            return NotImplemented

    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    circ.append(_DecliningGate(), [circ.modes[0], circ.modes[1]])

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    with pytest.raises(ValueError, match="declined to apply"):
        circ._apply_unitary_(vec0, norb, nelec, copy=True)
