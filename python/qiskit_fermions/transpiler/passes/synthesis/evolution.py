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

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparseObservable
from qiskit.transpiler import TransformationPass

from qiskit_fermions.circuit.library.evolution import Evolution
from qiskit_fermions.operators import FermionOperator, MajoranaOperator


class EvolutionSynthesis(TransformationPass):
    """TODO."""

    # TODO: add an OperatorProtocol to avoid hard-coding this list of types from our package
    MapperFunction = Callable[[FermionOperator | MajoranaOperator], SparseObservable]
    """TODO."""

    def __init__(self, mapper_fn: MapperFunction) -> None:
        """TODO."""
        super().__init__()
        self.mapper_fn = mapper_fn

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """TODO."""
        # FIXME: the provided DAGCircuit still has a "fermion register" rather than one with qubits

        for node in dag.op_nodes():
            if not isinstance(node.op, Evolution):
                continue

            pauli_op = self.mapper_fn(node.op.operator)
            # TODO: add a SparseObservable.real_if_close method
            real_pauli_op = SparseObservable.from_raw_parts(
                pauli_op.num_qubits,
                np.asarray(pauli_op.coeffs).real,
                pauli_op.bit_terms,
                pauli_op.indices,
                pauli_op.boundaries,
                check=False,
            )

            node_circ = QuantumCircuit(dag.qubits)
            node_circ.append(PauliEvolutionGate(real_pauli_op, time=node.op.params[0]), node.qargs)
            node_dag = circuit_to_dag(node_circ)

            dag.substitute_node_with_dag(node, node_dag)

        return dag
