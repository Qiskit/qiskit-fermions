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

"""Minimalistic conversion transpiler passes."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import GenericPass
from qiskit.transpiler import TranspileLayout

from qiskit_fermions.circuit import FermionicCircuit, FermionicDAGCircuit


class FermionicCircuitToDAG(GenericPass[FermionicCircuit, FermionicDAGCircuit]):
    """Converts a :class:`.FermionicCircuit` to a :class:`.FermionicDAGCircuit`."""

    def run(self, passmanager_ir: FermionicCircuit) -> FermionicDAGCircuit:
        """Runs this conversion pass.

        Args:
            passmanager_ir: the input circuit to convert.

        Returns:
            The converted circuit.
        """
        return circuit_to_dag(passmanager_ir._inner, copy_operations=True)


class FermionicDAGToCircuit(GenericPass[FermionicDAGCircuit, FermionicCircuit]):
    """Converts a :class:`.FermionicDAGCircuit` to a :class:`.FermionicCircuit`."""

    def run(self, passmanager_ir: FermionicDAGCircuit) -> FermionicCircuit:
        """Runs this conversion pass.

        Args:
            passmanager_ir: the input circuit to convert.

        Returns:
            The converted circuit.
        """
        out = FermionicCircuit(passmanager_ir.num_qubits())
        out._inner = dag_to_circuit(passmanager_ir, copy_operations=False)
        return out


class QuantumDAGToCircuit(GenericPass[DAGCircuit, QuantumCircuit]):
    """Converts a :class:`~qiskit.dagcircuit.DAGCircuit` to a :class:`~qiskit.circuit.QuantumCircuit`."""

    def run(self, passmanager_ir: DAGCircuit) -> QuantumCircuit:
        """Runs this conversion pass.

        Args:
            passmanager_ir: the input circuit to convert.

        Returns:
            The converted circuit.
        """
        qc = passmanager_ir.to_circuit(copy_operations=False)
        qc._layout = TranspileLayout.from_property_set(passmanager_ir, self.property_set)
        return qc
