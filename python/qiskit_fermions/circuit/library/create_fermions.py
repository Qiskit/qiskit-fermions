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

"""Fermion initialization."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit

from .. import FermionGate


class CreateFermions(FermionGate):
    """TODO."""

    def __init__(self, num_fermions: int, indices: list[int]) -> None:
        """TODO."""
        super().__init__("CreateFermions", num_fermions, indices)

    def _define(self):
        # FIXME: this assumes a Jordan-Wigner like encoding!
        circ = QuantumCircuit(self.num_qubits)
        for idx in self.params:
            circ.x(idx)
        self.definition = circ
