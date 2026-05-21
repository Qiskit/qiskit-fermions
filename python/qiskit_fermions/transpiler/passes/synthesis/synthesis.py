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

from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass

from qiskit_fermions.circuit import FermionicGate

from ... import F2QLayout


class F2QSynthesisPlugin(Protocol):
    """The protocol for plugins to the :class:`.F2QSynthesis` transpiler pass."""

    def run(
        self,
        in_node: DAGOpNode,
        freg_indices: list[int],
        out_dag: DAGCircuit,
        qreg: QuantumRegister,
    ) -> None:
        """Translates the provided fermion-based circuit instruction to a qubit-based one.

        Args:
            in_node: a fermion-based circuit instruction stored in a
                :class:`~qiskit.dagcircuit.DAGOpNode`. Specifically, this guarantees that
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` is of type :class:`.FermionicGate`.
            freg_indices: TODO.
            out_dag: the qubit-based :class:`~qiskit.dagcircuit.DAGCircuit` into which this plugin
                must insert the translated circuit instruction.
            qreg: TODO.
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

    def map_register(self, in_node: DAGOpNode) -> tuple[list[int], QuantumRegister]:
        """TODO."""
        f2q_layout = cast(F2QLayout, self.property_set["f2q_layout"])

        encountered_fermionic_registers: set[QuantumRegister] = set()
        freg_indices = []
        for fermion in in_node.qargs:
            for freg in f2q_layout:
                if fermion in freg:
                    encountered_fermionic_registers.add(freg)
                    freg_indices.append(freg.index(fermion))
                    break

        if len(encountered_fermionic_registers) > 1:
            raise NotImplementedError(
                "Cannot map a FermionicGate acting on fermionic modes that are spread across "
                "multiple FermionicRegister instances."
            )

        freg = encountered_fermionic_registers.pop()
        qreg = f2q_layout[freg]

        return freg_indices, qreg

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

            freg_indices, qreg = self.map_register(node)

            plugin.run(node, freg_indices, out_dag, qreg)

        return out_dag
