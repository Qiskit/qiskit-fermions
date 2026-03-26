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

from typing import Protocol, cast

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass

from qiskit_fermions.circuit import FermionGate

from ... import F2QLayout


class F2QSynthesisPlugin(Protocol):
    """TODO."""

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout):
        """TODO."""
        ...


class F2QSynthesis(TransformationPass):
    """TODO."""

    def __init__(self) -> None:
        """TODO."""
        super().__init__()

        self.plugins: dict[type[DAGOpNode], F2QSynthesisPlugin] = {}

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """TODO."""
        f2q_layout = cast(F2QLayout, self.property_set["f2q_layout"])

        out_dag = dag.copy_empty_like()

        for freg, qreg in f2q_layout.items():
            out_dag.add_qreg(qreg)
            out_dag.remove_qregs(freg)
            out_dag.remove_qubits(*freg)

        for node in dag.op_nodes():
            if not isinstance(node.op, FermionGate):
                raise NotImplementedError("TODO.")

            plugin = self.plugins.get(type(node.op), None)
            if plugin is None:
                raise TypeError("TODO.")

            plugin.run(node, out_dag, f2q_layout=f2q_layout)

        return out_dag
