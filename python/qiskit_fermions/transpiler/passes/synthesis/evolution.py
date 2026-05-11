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

"""Fermion-operator evolution gate synthesis."""

from __future__ import annotations

from collections.abc import Callable

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import SparseObservable

from qiskit_fermions.operators.protocol import OperatorTrait

from ... import F2QLayout
from ..utils import _parse_node_indices

MapperFunction = Callable[[OperatorTrait, int], SparseObservable]
"""The function signature for :attr:`mapper_fn`."""


class EvolutionSynthesis:
    """A :class:`.F2QSynthesisPlugin` for transpiling a :class:`.Evolution`."""

    def __init__(self, mapper_fn: MapperFunction) -> None:
        """Initializes the transpilation plugin.

        Args:
            mapper_fn: the fermion-to-qubit operator mapping function.
        """
        super().__init__()

        self.mapper_fn: MapperFunction = mapper_fn
        """The fermion-to-qubit operator mapping function.

        The two input arguments should be the following:

        1. the operator to be mapped.
        2. the number of qubits that the resulting operator should be defined on.

        .. note::
           It is the user's responsibility to ensure that this function is in-sync with the global
           transpilation :class:`~qiskit_fermions.transpiler.F2QLayout` setting.
        """

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout):
        """Runs this transpilation plugin.

        Args:
            in_node: the input fermion-based circuit instruction. When this plugin gets called, the
                ``in_node.op`` attribute `must` be of type :class:`.Evolution`.
            out_dag: the output qubit-based circuit.
            f2q_layout: the global transpilation :class:`~qiskit_fermions.transpiler.F2QLayout`
                setting.

        .. seealso::
           The documentation of :class:`.F2QSynthesisPlugin` for more detailed explanations of the
           arguments.

        Raises:
            NotImplementedError: when ``in_node`` acts on fermionic modes that are spread across
                multiple :type:`~qiskit_fermions.circuit.FermionicRegister` instances.
        """
        encountered_mode_registers, global_mode_indices = _parse_node_indices(in_node, f2q_layout)

        if len(encountered_mode_registers) > 1:
            raise NotImplementedError(
                "Cannot map an Evolution gate acting on fermionic modes that are spread across "
                "multiple FermionicRegister instances."
            )

        mreg = encountered_mode_registers.pop()
        qreg = f2q_layout[mreg]

        local_op = in_node.op.operator
        global_op = local_op.relabel_modes(global_mode_indices)
        pauli_op = self.mapper_fn(global_op, len(qreg)).simplify()

        circ = QuantumCircuit(qreg)
        circ.append(PauliEvolutionGate(pauli_op, time=in_node.op.params[0]), qreg)
        new_dag = circuit_to_dag(circ)

        out_dag.compose(new_dag, qubits=list(qreg), front=False, inplace=True)
