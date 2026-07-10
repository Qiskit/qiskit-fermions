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

"""FermionicCircuit."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit, QuantumRegister

from . import FermionicMode, FermionicSpecifier
from .fermionic_gate import FermionicGate

if TYPE_CHECKING:
    from . import FermionicRegister


class FermionicCircuit:
    """A wrapper around :class:`~qiskit.circuit.QuantumCircuit` for expressing fermionic circuits.

    This class maintains a reduced API compared to the full API of the underlying
    :class:`~qiskit.circuit.QuantumCircuit`. This is done to avoid exposing (amongst other methods)
    the ability to apply qubit-based gates onto a fermionic circuit, which would not be a
    well-defined operation in the general case.
    """

    def __init__(self, num_modes: int) -> None:
        """Initializing a circuit instance can be done with the arguments listed below.

        Args:
            num_modes: the number of fermionic modes on which this circuit acts.
        """
        self.register: FermionicRegister = QuantumRegister(num_modes, "f")
        """The inner circuit's :type:`~qiskit_fermions.circuit.FermionicRegister`."""
        self._inner = QuantumCircuit(self.register)

    @property
    def metadata(self) -> dict:
        """Re-exposes :external:attr:`~qiskit.circuit.QuantumCircuit.metadata`."""
        return cast(dict, self._inner.metadata)

    @metadata.setter
    def metadata(self, metadata: dict) -> None:
        self._inner.metadata = metadata

    @property
    def modes(self) -> list[FermionicMode]:
        """The fermionic mode ``bits`` that this circuit acts upon."""
        return cast(list[FermionicMode], self._inner.qubits)

    def append(
        self,
        gate: FermionicGate,
        fargs: FermionicSpecifier,
        cargs: None = None,
        *,
        copy: bool = True,
    ) -> None:
        """Appends a :class:`.FermionicGate` to this circuit.

        Args:
            gate: the fermionic gate to apply.
            fargs: the fermionic modes on which this gate acts.
            cargs: the classical bits on which this gate acts.

              .. warning::
                 No gates of this kind are currently supported.

            copy: forwarded to :meth:`~qiskit.circuit.QuantumCircuit.append`.

        Raises:
            ValueError: if the provided ``gate`` is not an instance of :class:`.FermionicGate`.
        """
        if not isinstance(gate, FermionicGate):
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
    ) -> FermionicCircuit:
        """Re-exposes :external:meth:`~qiskit.circuit.QuantumCircuit.decompose`."""
        inner_decomposed = self._inner.decompose(gates_to_decompose=gates_to_decompose, reps=reps)
        out = FermionicCircuit(len(self.register))
        out.register = self.register
        out._inner = inner_decomposed
        return out

    def draw(self, *args, **kwargs) -> Any:
        """Directly exposes the inner circuit's :meth:`~qiskit.circuit.QuantumCircuit.draw` method."""
        return self._inner.draw(*args, **kwargs)

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Applies this circuit to an ffsim state vector, implementing ffsim's protocol.

        This walks the circuit in topological order and applies each instruction's unitary effect to
        the state vector via ffsim's ``SupportsApplyUnitary`` protocol. Each instruction acting on a
        subset of the register has its fermionic modes relabeled to their absolute (global) indices
        before being applied.

        Args:
            vec: the state vector to apply this circuit to.
            norb: the number of spatial orbitals.
            nelec: either a single integer representing the number of fermions for a spinless system,
                or a pair of integers storing the numbers of spin alpha and spin beta fermions.
            copy: whether to copy the vector before operating on it.

        Returns:
            The transformed vector.

        Raises:
            TypeError: if a circuit instruction does not implement ffsim's
                ``SupportsApplyUnitary`` protocol.
            ValueError: if a circuit instruction declines to apply its unitary for the given
                ``norb`` and ``nelec``.
        """
        from qiskit_fermions.transpiler.converters import FermionicCircuitToDAG

        dag = FermionicCircuitToDAG().run(self)

        if copy:
            vec = vec.copy()

        for node in dag.topological_op_nodes():
            instr = node.op

            # the absolute (global) mode indices this instruction acts on
            freg_indices = [dag.find_bit(qubit).index for qubit in node.qargs]

            # prefer the placement-aware variant so the instruction's operator is relabeled onto its
            # absolute modes; fall back to the plain protocol method otherwise (identity placement)
            placed_method = getattr(instr, "_apply_unitary_placed_", None)
            if placed_method is not None:
                result = placed_method(vec, norb, nelec, False, freg_indices)
            else:
                method = getattr(instr, "_apply_unitary_", None)
                if method is None:
                    raise TypeError(
                        f"Circuit instruction of type '{type(instr)}' does not implement "
                        "ffsim's SupportsApplyUnitary protocol!"
                    )
                result = method(vec, norb, nelec, False)

            if result is NotImplemented:
                raise ValueError(
                    f"Circuit instruction of type '{type(instr)}' declined to apply its unitary "
                    f"for {norb=}, {nelec=}!"
                )

            vec = result

        return vec
