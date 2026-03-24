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

"""Hamiltonian evolution gate synthesis."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import SparseObservable

from .. import F2QLayout

if TYPE_CHECKING:
    from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
    from qiskit_fermions._lib.operators.majorana_operator import MajoranaOperator
else:
    from qiskit_fermions.operators import FermionOperator, MajoranaOperator


class EvolutionSynthesis:
    """TODO."""

    # TODO: add an OperatorProtocol to avoid hard-coding this list of types from our package
    MapperFunction = Callable[[FermionOperator | MajoranaOperator], SparseObservable]
    """TODO."""

    def __init__(self, mapper_fn: MapperFunction) -> None:
        """TODO."""
        super().__init__()
        self.mapper_fn = mapper_fn

    def run(self, node: DAGOpNode, layout: F2QLayout) -> DAGCircuit:
        """TODO."""
        qubits = [layout.f2q[fermion] for fermion in node.qargs]
        mode_relabeling = [qubit._index for qubit in qubits]

        pauli_op = self.mapper_fn(node.op.operator.relabel_modes(mode_relabeling)).simplify()

        # TODO: verify that the qubit-reording is handled correctly w.r.t. the modes already being
        # relabeled above!
        qubits_reordered = [qubits[idx] for idx in mode_relabeling]
        circ = QuantumCircuit(qubits_reordered)
        circ.append(PauliEvolutionGate(pauli_op, time=node.op.params[0]), qubits_reordered)

        return (circuit_to_dag(circ), qubits_reordered)
