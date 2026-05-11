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

"""Fermion-to-qubit circuit synthesis pass."""

from __future__ import annotations

from typing import Protocol, cast

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass

from qiskit_fermions.circuit import FermionicGate

from ... import F2QLayout


class F2QSynthesisPlugin(Protocol):
    """The protocol for plugins to the :class:`.F2QSynthesis` transpiler pass."""

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
        """Translates the provided fermion-based circuit instruction to a qubit-based one.

        Args:
            in_node: a fermion-based circuit instruction stored in a
                :class:`~qiskit.dagcircuit.DAGOpNode`. Specifically, this guarantees that
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` is of type :class:`.FermionicGate`.
            out_dag: the qubit-based :class:`~qiskit.dagcircuit.DAGCircuit` into which this plugin
                must insert the translated circuit instruction.
            f2q_layout: the :type:`~qiskit_fermions.transpiler.F2QLayout` setting that is global to
                the transpilation process. It is the plugin's responsibility to respect this mapping
                of :type:`~qiskit_fermions.circuit.FermionicRegister` to
                :class:`~qiskit.circuit.QuantumRegister`.
        """
        ...


class F2QSynthesis(TransformationPass):
    """A transpilation pass to map fermion-based circuit instructions to qubit-based ones.

    This transpilation pass works similarly to Qiskit's
    :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass; given an input
    :class:`~qiskit.dagcircuit.DAGCircuit` with :class:`.FermionicGate` instructions, it iterates them
    and delegates the translation to qubit-based instructions to matching :attr:`plugins`.
    The insertion of the qubit-based circuit instructions into the output
    :class:`~qiskit.dagcircuit.DAGCircuit` is also left to the plugin. This pass will merely have
    prepared the :class:`~qiskit.circuit.QuantumRegister` according to the global transpilation
    :class:`~qiskit_fermions.transpiler.F2QLayout` setting.
    """

    def __init__(self) -> None:
        """Initializes the transpiler pass."""
        super().__init__()

        self.plugins: dict[type[DAGOpNode], F2QSynthesisPlugin] = {}
        """A dictionary of fermion-to-qubit circuit instruction transpilation plugins.

        .. autoclass:: F2QSynthesisPlugin
           :show-inheritance:
           :members:
           :exclude-members: __init__
           :no-inherited-members:
           :no-special-members:
        """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Runs this transpilation pass.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported.

        Returns:
            The output circuit with qubit-based instructions.

        Raises:
            ValueError: when a :class:`~qiskit.dagcircuit.DAGOpNode` is encountered whose
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` is not of type :class:`.FermionicGate`.
            TypeError: when a :class:`.FermionicGate` type is encountered for which no translation
                plugin is present in :attr:`plugins`.
        """
        f2q_layout = cast(F2QLayout, self.property_set["f2q_layout"])

        out_dag = dag.copy_empty_like()

        for freg, qreg in f2q_layout.items():
            out_dag.add_qreg(qreg)
            out_dag.remove_qregs(freg)
            out_dag.remove_qubits(*freg)

        for node in dag.op_nodes():
            op_type = type(node.op)
            if not isinstance(node.op, FermionicGate):
                raise ValueError("Encountered an unsupported circuit instruction type: {}", op_type)

            plugin = self.plugins.get(op_type, None)
            if plugin is None:
                raise TypeError(
                    "No plugin registered for transpiling a circuit instruction of type: {}",
                    op_type,
                )

            plugin.run(node, out_dag, f2q_layout=f2q_layout)

        return out_dag
