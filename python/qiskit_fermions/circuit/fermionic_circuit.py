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
        self, vec: np.ndarray | None, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Applies this circuit to an ffsim state vector, implementing ffsim's protocol.

        This walks the circuit in topological order and applies each instruction's unitary effect to
        the state vector via ffsim's ``SupportsApplyUnitary`` protocol. Each instruction acting on a
        subset of the register has its fermionic modes relabeled to their absolute (global) indices
        before being applied.

        See :meth:`_apply_unitary_placed_` for the details; this method assumes the circuit's modes
        are the vector's modes ``0..num_modes`` (i.e. an identity mode placement).

        Args:
            vec: the state vector to apply this circuit to. May be ``None`` to let the circuit's
                first instruction seed the initial state (e.g. :class:`.InitializeModes`, which
                prepares an occupation determinant from no incoming state); the seeded vector is then
                threaded through the remaining instructions. A ``None`` vector requires the first
                instruction to be able to produce a state -- one that only transforms an incoming
                vector will raise.
            norb: the number of spatial orbitals.
            nelec: either a single integer representing the number of fermions for a spinless system,
                or a pair of integers storing the numbers of spin alpha and spin beta fermions.
            copy: whether to copy the vector before operating on it. Ignored when ``vec`` is ``None``.

        Returns:
            The transformed vector.

        Raises:
            TypeError: if a circuit instruction does not implement ffsim's
                ``SupportsApplyUnitary`` protocol.
            ValueError: if a circuit instruction declines to apply its unitary for the given
                ``norb`` and ``nelec``; if an instruction implementing only the plain
                ``_apply_unitary_`` protocol is placed on a non-identity mode subset; or if the
                circuit is empty and ``vec`` is ``None`` (there is no state to return).
        """
        return self._apply_unitary_placed_(vec, norb, nelec, copy, list(range(len(self.register))))

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray | None,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        """Applies this circuit after placing its modes onto the vector's global modes.

        This walks the circuit in topological order and applies each instruction's unitary effect to
        the state vector via ffsim's ``SupportsApplyUnitary`` protocol. Each instruction's own modes
        are first mapped through this circuit's placement: a circuit-local mode ``m`` maps to the
        global mode ``freg_indices[m]``, so a sub-instruction acting on circuit-local modes
        ``[m0, m1, ...]`` is applied on the global modes ``[freg_indices[m0], freg_indices[m1], ...]``.
        With the identity placement ``freg_indices == 0..num_modes`` this is exactly
        :meth:`_apply_unitary_`; a subset placement lets this circuit act as the definition of a gate
        placed on a subset of a larger register (e.g. :class:`.UCJ`).

        An instruction is placed onto its absolute modes only if it implements the placement-aware
        ``_apply_unitary_placed_`` extension. An instruction implementing only ffsim's plain
        ``_apply_unitary_`` -- which has no mode argument and therefore acts on modes ``0..k`` of the
        vector -- can only be honored when its placement is the identity ``[0, 1, ..., k-1]``; on any
        other subset the placement cannot be expressed and the instruction is rejected rather than
        silently applied on the wrong modes.

        Args:
            vec: the state vector to apply this circuit to. May be ``None`` to let the circuit's
                first instruction seed the initial state (e.g. :class:`.InitializeModes`, which
                prepares an occupation determinant from no incoming state); the seeded vector is then
                threaded through the remaining instructions. A ``None`` vector requires the first
                instruction to be able to produce a state -- one that only transforms an incoming
                vector will raise.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer representing the number of fermions for a spinless system,
                or a pair of integers storing the numbers of spin alpha and spin beta fermions.
            copy: whether to copy the vector before operating on it. Ignored when ``vec`` is ``None``.
            freg_indices: the absolute (global) mode indices that this circuit's modes map onto.

        Returns:
            The transformed vector.

        Raises:
            TypeError: if a circuit instruction does not implement ffsim's
                ``SupportsApplyUnitary`` protocol.
            ValueError: if a circuit instruction declines to apply its unitary for the given
                ``norb`` and ``nelec``; if an instruction implementing only the plain
                ``_apply_unitary_`` protocol is placed on a non-identity mode subset; or if the
                circuit is empty and ``vec`` is ``None`` (there is no state to return).
        """
        from qiskit_fermions.transpiler.converters import FermionicCircuitToDAG

        # PERF: the DAG depends only on the circuit structure (``self``), not on ``vec``/``norb``/
        # ``nelec``, yet it is rebuilt on every call. This is fine for a single ``apply_unitary``, but
        # repeated evolutions of the same circuit (parameter sweeps, time-stepping) redo identical
        # work. Left as-is deliberately: caching would require invalidating on every circuit mutation
        # (a correctness hazard on a mutable circuit). Revisit if a repeated-call use case arises.
        dag = FermionicCircuitToDAG().run(self)

        # ``vec`` may be None to let the first instruction seed the state (e.g. InitializeModes);
        # only copy a real array. The gate-to-gate loop below then threads whatever the first
        # instruction returns into the rest of the circuit.
        if copy and vec is not None:
            vec = vec.copy()

        for node in dag.topological_op_nodes():
            instr = node.op

            # each instruction's circuit-local modes, mapped through this circuit's own placement:
            # local mode ``m`` of this circuit sits at global mode ``freg_indices[m]``
            node_freg_indices = [freg_indices[dag.find_bit(qubit).index] for qubit in node.qargs]

            # prefer the placement-aware variant so the instruction's operator is relabeled onto its
            # absolute modes; fall back to the plain protocol method otherwise
            placed_method = getattr(instr, "_apply_unitary_placed_", None)
            if placed_method is not None:
                result = self._call_apply_unitary(
                    instr, placed_method, vec, norb, nelec, False, node_freg_indices
                )
            else:
                method = getattr(instr, "_apply_unitary_", None)
                if method is None:
                    raise TypeError(
                        f"Circuit instruction of type '{type(instr)}' does not implement "
                        "ffsim's SupportsApplyUnitary protocol!"
                    )
                # ffsim's plain ``_apply_unitary_(vec, norb, nelec, copy)`` protocol has no mode
                # argument: the instruction acts on modes ``0..k`` of the global vector, so it can
                # only be honored when its placement is the identity ``[0, 1, ..., k-1]``. On any
                # other subset the placement cannot be expressed, and applying it anyway would
                # silently act on the wrong modes -- reject instead of producing a wrong state.
                if node_freg_indices != list(range(len(node.qargs))):
                    raise ValueError(
                        f"Circuit instruction of type '{type(instr)}' implements only ffsim's "
                        "'_apply_unitary_' protocol, which has no mode-placement argument, but it "
                        f"is placed on the non-identity modes {node_freg_indices}. Such an "
                        "instruction can only be applied on the identity placement "
                        f"{list(range(len(node.qargs)))}; implement '_apply_unitary_placed_' to "
                        "support subset placement."
                    )
                result = self._call_apply_unitary(instr, method, vec, norb, nelec, False)

            if result is NotImplemented:
                raise ValueError(
                    f"Circuit instruction of type '{type(instr)}' declined to apply its unitary "
                    f"for {norb=}, {nelec=}!"
                )

            vec = result

        if vec is None:
            # no instruction ran (an empty circuit) and no incoming vector was supplied, so there is
            # no state to return. The protocol must yield an array, not ``None``; a caller wanting an
            # identity here must pass the vector to act on.
            raise ValueError(
                "Cannot apply an empty circuit to a vector of None: there is no incoming state and "
                "no instruction to seed one. Pass a state vector for the circuit to act on."
            )

        return vec

    @staticmethod
    def _call_apply_unitary(instr, method, vec, *args):
        """Invokes an instruction's apply-unitary method, clarifying the ``vec is None`` failure.

        A ``None`` incoming vector is only meaningful for a state-*producing* first instruction (e.g.
        :class:`.InitializeModes`); a transform-only instruction (an :class:`.Evolution`, an
        :class:`.OrbitalRotation`, ...) has no state to act on and fails deep inside ffsim/scipy with
        an opaque ``AttributeError`` on the ``None``. Translate that into the clean ``ValueError`` the
        ``_apply_unitary_``/``_apply_unitary_placed_`` docstrings promise, naming the instruction.
        """
        try:
            return method(vec, *args)
        except AttributeError as exc:
            if vec is None:
                raise ValueError(
                    f"Circuit instruction of type '{type(instr)}' cannot seed a state from a None "
                    "vector: it only transforms an incoming state. A None vector requires the "
                    "circuit's first instruction to be a state producer (e.g. InitializeModes), or "
                    "pass an explicit state vector for the circuit to act on."
                ) from exc
            raise
