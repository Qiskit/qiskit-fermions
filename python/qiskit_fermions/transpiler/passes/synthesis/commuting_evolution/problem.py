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

"""The data that flows between the plugin, the candidates and the selector."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Operation
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import PauliList, SparseObservable, SparsePauliOp
from qiskit.transpiler import CouplingMap, Target

from ._layers import sorted_pair

IMAG_TOL = 1e-10
"""Imaginary parts of the coefficients above this make the evolution non-Hermitian, and the gate is
declined."""


class Declined(Exception):
    """Raised by :meth:`Problem.from_gate` for a Pauli evolution that cannot be handled.

    The gate has the right form but its coefficients are complex or its terms do not commute.
    """


@dataclass(frozen=True)
class Problem:
    """One group of mutually commuting Pauli terms to be exponentiated: *what* is synthesized.

    It holds the terms as a matrix of X and Z bits with one row per term, and their coefficients.
    The candidates work on these arrays: conjugation by a Clifford is arithmetic over GF(2) on the
    columns of the bit matrix. Plain arrays also let a candidate be implemented in another language
    against the same data. :attr:`operator` gives the terms as a
    :class:`~qiskit.quantum_info.SparseObservable`.

    .. seealso::

        :class:`.Context`, which holds *where* the circuit will run: what is known about the device,
        and the pairs of qubits the user allows. A candidate receives both.
    """

    x: np.ndarray
    """The X bits of the terms (0/1, terms by qubits)."""
    z: np.ndarray
    """The Z bits of the terms. A qubit with both bits set carries a Y."""
    coeffs: np.ndarray
    """The real coefficients of the terms."""
    time: float
    """The evolution time ``t`` of ``exp(-i t H)``."""
    num_qubits: int
    """The number of qubits."""

    @cached_property
    def operator(self) -> SparseObservable:
        """The terms as a :class:`~qiskit.quantum_info.SparseObservable`.

        It is built from the arrays when first asked for.
        """
        support = self.x | self.z
        rows, qubits = np.nonzero(support)  # row by row, qubits in increasing order
        bit_terms = (self.x[rows, qubits] << 1) | self.z[
            rows, qubits
        ]  # Z = 0b01, X = 0b10, Y = 0b11
        boundaries = np.concatenate([[0], np.cumsum(support.sum(axis=1))])
        return SparseObservable.from_raw_parts(
            self.num_qubits,
            self.coeffs.astype(complex),
            bit_terms.astype(np.uint8),
            qubits.astype(np.uint32),
            boundaries.astype(np.uintp),
        )

    @cached_property
    def paulis(self) -> PauliList:
        """The terms as a :class:`~qiskit.quantum_info.PauliList` without phases.

        It serves to conjugate the terms by Cliffords.
        """
        return PauliList.from_symplectic(self.z.astype(bool), self.x.astype(bool))

    @classmethod
    def from_operator(cls, operator: SparseObservable | SparsePauliOp, time: float) -> Problem:
        """The problem of ``exp(-i t H)`` for an operator whose terms commute and are real.

        Args:
            operator: the terms. Projectors of a :class:`~qiskit.quantum_info.SparseObservable` are
                expanded into Paulis, the terms are simplified, and the imaginary parts of the
                coefficients are dropped.
            time: the evolution time.

        Returns:
            The problem.
        """
        num_qubits = operator.num_qubits
        if isinstance(operator, SparseObservable):
            # read from the raw arrays, without a round trip through a SparsePauliOp
            obs = operator.as_paulis().simplify()
            num_terms = obs.num_terms
            if num_terms == 0:
                # one identity term of coefficient zero, as SparsePauliOp.simplify gives
                x = np.zeros((1, num_qubits), np.uint8)
                z, coeffs = x.copy(), np.zeros(1)
            else:
                bit_terms = np.asarray(obs.bit_terms, np.uint8)  # Z = 0b01, X = 0b10, Y = 0b11
                qubits = np.asarray(obs.indices, np.intp)
                rows = np.repeat(np.arange(num_terms), np.diff(np.asarray(obs.boundaries, np.intp)))
                x = np.zeros((num_terms, num_qubits), np.uint8)
                z = np.zeros((num_terms, num_qubits), np.uint8)
                x[rows, qubits] = (bit_terms >> 1) & 1
                z[rows, qubits] = bit_terms & 1
                coeffs = np.asarray(obs.coeffs)
        else:
            op = operator.simplify()  # leaves no phases on the Paulis: Y is the Hermitian letter
            x, z, coeffs = op.paulis.x.astype(np.uint8), op.paulis.z.astype(np.uint8), op.coeffs
        return cls(
            x=x,
            z=z,
            coeffs=np.ascontiguousarray(np.real(coeffs), dtype=float),
            time=float(time),
            num_qubits=num_qubits,
        )

    @classmethod
    def from_gate(cls, gate: Operation) -> Problem | None:
        """The problem of a gate, or ``None`` if the gate is not one this plugin handles.

        Args:
            gate: the operation the high-level synthesis hands to a plugin.

        Returns:
            The problem of a :class:`~qiskit.circuit.library.PauliEvolutionGate` over a single
            operator with a real time, whose coefficients are real and whose terms mutually commute.
            ``None`` if the operation is not such a gate (a different operation, a list of
            operators, or a time that is not a plain real number).

        Raises:
            Declined: if the gate is a Pauli evolution of the right form, but its coefficients are
                complex or its terms do not commute.
        """
        if not isinstance(gate, PauliEvolutionGate):
            return None
        op = gate.operator
        if not isinstance(op, (SparseObservable, SparsePauliOp)):
            return None  # a list of operators
        time = gate.params[0]
        if not isinstance(time, numbers.Real):
            return None
        # expanding projectors into Paulis keeps the coefficients real, so the given ones decide
        if np.abs(np.asarray(op.coeffs).imag).max(initial=0.0) > IMAG_TOL:
            raise Declined("complex coefficients")
        problem = cls.from_operator(op, float(time))
        x, z = problem.x, problem.z
        if ((x @ z.T + z @ x.T) % 2).any():
            raise Declined("terms do not commute")
        return problem


@dataclass(frozen=True)
class Context:
    """*Where* the circuit will run.

    It holds what the high-level synthesis knows about the device, and the user's CZ pairs.

    All qubit pairs it holds and returns are in the gate's own qubit indices. Every field but
    ``num_qubits`` is keyword-only.
    """

    num_qubits: int
    """The number of qubits of the gate."""
    _: KW_ONLY
    coupling_map: CouplingMap | None = None
    """The device's coupling map, or ``None``."""
    target: Target | None = None
    """The device's target, or ``None``."""
    physical_qubits: tuple[int, ...] | None = None
    """The device qubit of each qubit of the gate, or ``None`` before the layout stage."""
    user_edges: tuple[tuple[int, int], ...] | None = None
    """CZ pairs given by the user, between qubits of the gate. They take precedence over the
    device's. The plugin maps the pairs it is given onto each gate's qubits."""

    @property
    def placed(self) -> bool:
        """Whether the gate's qubits sit on known device qubits of a known coupling map."""
        return self.physical_qubits is not None and self.coupling_map is not None

    @cached_property
    def edges(self) -> list[tuple[int, int]] | None:
        """The pairs ``(i, j)``, ``i < j``, that a CZ gate may act on, or ``None``.

        It is ``None`` when no connectivity is known.

        The user's pairs, if given; otherwise the device's coupling edges among the physical qubits,
        if the gate is placed.
        """
        if self.user_edges is not None:
            return sorted({sorted_pair(e) for e in self.user_edges})
        # the test of `placed`, written out so that the type checker narrows both fields
        if self.coupling_map is None or self.physical_qubits is None:
            return None
        index = {p: i for i, p in enumerate(self.physical_qubits)}
        edges: set[tuple[int, int]] = set()
        for a, b in self.coupling_map.get_edges():
            if a in index and b in index and a != b:
                edges.add(sorted_pair((index[a], index[b])))
        return sorted(edges)


@dataclass(frozen=True)
class Variant:
    """One circuit a candidate offers for a problem."""

    candidate: str
    """The :attr:`~.Candidate.name` of the candidate that built it."""
    variant: str
    """Which of the candidate's circuits it is, or ``""`` for a candidate with a single one."""
    circuit: QuantumCircuit
    """The circuit, on the gate's qubits. It implements the evolution exactly."""

    @property
    def label(self) -> str:
        """``"candidate[variant]"``, or the candidate's name alone when :attr:`variant` is empty."""
        return f"{self.candidate}[{self.variant}]" if self.variant else self.candidate


@dataclass(frozen=True)
class Report:
    """What the plugin did with one gate, handed to its ``callback``."""

    problem: Problem | None
    """The gate's problem, or ``None`` if the gate was declined before one was built."""
    context: Context | None
    """The gate's context, or ``None`` likewise."""
    variants: Sequence[Variant]
    """Every circuit the candidates built, in tie-break order."""
    chosen: Variant | None
    """The selector's choice, or ``None`` if the gate was declined."""
    declined: str | None
    """Why the gate was declined, or ``None`` if it was synthesized."""
