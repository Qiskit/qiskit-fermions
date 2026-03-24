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

"""Fermion-to-qubit circuit translation pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass

from qiskit_fermions.circuit import FermionGate


@dataclass
class F2QLayout:
    """TODO."""

    fermions: QuantumRegister
    """TODO."""

    qubits: QuantumRegister
    """TODO."""

    q2f: dict[Qubit, Qubit | None]
    """TODO."""

    @property
    def f2q(self) -> dict[Qubit, Qubit]:
        """TODO."""
        return {f: q for q, f in self.q2f.items() if f is not None}

    @property
    def num_fermions(self) -> int:
        """TODO."""
        return len(self.fermions)

    @property
    def num_qubits(self) -> int:
        """TODO."""
        return len(self.qubits)

    @classmethod
    def trivial(cls, fermions: QuantumRegister) -> Self:
        """TODO."""
        qubits = QuantumRegister(len(fermions))
        return cls(fermions, qubits, {qubit: fermions[qubit._index] for qubit in qubits})


class F2QEncodingPlugin(Protocol):
    """TODO."""

    def run(self, gate: FermionGate, layout: F2QLayout) -> tuple[DAGCircuit, list[int]]:
        """TODO."""
        ...


class F2QEncoding(TransformationPass):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__()

        self.plugins: dict[type[DAGOpNode], F2QEncodingPlugin] = {}

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """TODO."""
        f2q_layout = self.property_set["f2q_layout"]

        new_dag = dag.copy_empty_like()
        new_dag.add_qreg(f2q_layout.qubits)
        new_dag.remove_qregs(f2q_layout.fermions)
        new_dag.remove_qubits(*f2q_layout.fermions)

        for node in dag.op_nodes():
            if not isinstance(node.op, FermionGate):
                raise NotImplementedError("TODO.")

            plugin = self.plugins.get(type(node.op), None)
            if plugin is None:
                raise TypeError("TODO.")

            translated, qargs = plugin.run(node, f2q_layout)

            new_dag.compose(translated, qubits=qargs, clbits=None, front=False, inplace=True)

        return new_dag
