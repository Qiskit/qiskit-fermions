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

"""FermionCircuit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit

from .fermion_gate import FermionGate

Fermion: TypeAlias = Qubit
"""TODO."""

FermionRegister: TypeAlias = QuantumRegister
"""TODO."""

FermionSpecifier: TypeAlias = Fermion | FermionRegister | int | slice | Sequence[Fermion | int]
"""TODO."""


class FermionCircuit:
    """TODO."""

    def __init__(self, num_fermions: int) -> None:
        """TODO."""
        self.register = QuantumRegister(num_fermions, "f")
        """TODO."""

        self._inner = QuantumCircuit(self.register)

    @property
    def fermions(self) -> list[Fermion]:
        """TODO."""
        return cast(list[Fermion], self._inner.qubits)

    def append(
        self, gate: FermionGate, fargs: FermionSpecifier, cargs: None = None, *, copy: bool = True
    ) -> None:
        """TODO."""
        if not isinstance(gate, FermionGate):
            raise ValueError("Unsupported instruction type: %s", type(gate))

        self._inner.append(gate, fargs, cargs, copy=copy)

    def decompose(self) -> QuantumCircuit:
        """TODO."""
        return self._inner.decompose()

    def draw(self, *args, **kwargs):
        """TODO."""
        return self._inner.draw(*args, **kwargs)
