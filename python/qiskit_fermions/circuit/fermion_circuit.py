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

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from qiskit.circuit import Instruction, QuantumCircuit, QuantumRegister

from . import FermionicMode, FermionicSpecifier
from .fermion_gate import FermionGate

if TYPE_CHECKING:
    from . import FermionicRegister


class FermionCircuit:
    """A wrapper around :class:`~qiskit.circuit.QuantumCircuit` for expressing fermionic circuits.

    This class maintains a reduced API compared to the full API of the underlying
    :class:`~qiskit.circuit.QuantumCircuit`. This is done to avoid exposing (amongst other methods)
    the ability to apply qubit-based gates onto a fermionic circuit, which would not be a
    well-defined operation in the general case.
    """

    def __init__(self, num_modes: int) -> None:
        """Initializes a FermionCircuit instance.

        Args:
            num_modes: the number of fermionic modes on which this circuit acts.
        """
        self.register: FermionicRegister = QuantumRegister(num_modes, "f")
        """The inner circuit's :type:`~qiskit_fermions.circuit.FermionicRegister`."""
        self._inner = QuantumCircuit(self.register)

    @property
    def fermions(self) -> list[FermionicMode]:
        """The fermionic mode `bits` that this circuit acts upon."""
        return cast(list[FermionicMode], self._inner.qubits)

    def append(
        self,
        gate: FermionGate,
        fargs: FermionicSpecifier,
        cargs: None = None,
        *,
        copy: bool = True,
    ) -> None:
        """Appends a :class:`.FermionGate` to this circuit.

        Args:
            gate: the fermionic gate to apply.
            fargs: the fermionic modes on which this gate acts.
            cargs: the classical bits on which this gate acts.

              .. warning::
                 No gates of this kind are currently supported.

            copy: forwarded to :meth:`~qiskit.circuit.QuantumCircuit.append`.

        Raises:
            ValueError: if the provided ``gate`` is not an instance of :class:`.FermionGate`.
        """
        if not isinstance(gate, FermionGate):
            raise ValueError("Unsupported instruction type: %s", type(gate))
        self._inner.append(gate, fargs, cargs, copy=copy)

    def count_ops(self) -> OrderedDict[str, int]:
        """Re-exposes :external:meth:`~qiskit.circuit.QuantumCircuit.count_ops`."""
        return cast(OrderedDict[str, int], self._inner.count_ops())

    def decompose(
        self,
        gates_to_decompose: (
            str | type[Instruction] | Sequence[str | type[Instruction]] | None
        ) = None,
        reps: int = 1,
    ) -> FermionCircuit:
        """Re-exposes :external:meth:`~qiskit.circuit.QuantumCircuit.decompose`."""
        inner_decomposed = self._inner.decompose(gates_to_decompose=gates_to_decompose, reps=reps)
        out = FermionCircuit(len(self.register))
        out.register = self.register
        out._inner = inner_decomposed
        return out

    def draw(self, *args, **kwargs) -> Any:
        """Directly exposes the inner circuit's :meth:`~qiskit.circuit.QuantumCircuit.draw` method."""
        return self._inner.draw(*args, **kwargs)
