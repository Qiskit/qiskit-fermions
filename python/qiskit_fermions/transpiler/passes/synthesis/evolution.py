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

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import SparseObservable

from qiskit_fermions.operators.protocol import OperatorTrait

from ... import F2QLayout
from ..utils import map_node_single_register

MapperFunction = Callable[[OperatorTrait, int], SparseObservable]
"""The function signature for :attr:`mapper_fn`."""


class MapperFnEvolutionSynthesis:
    """A :class:`.F2QSynthesisPlugin` for transpiling :class:`.Evolution` under a custom mapping."""

    def __init__(self, mapper_fn: MapperFunction) -> None:
        """Initializing this transpiler pass plugin can be done with the arguments listed below.

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

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
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
        freg_indices, qreg = map_node_single_register(in_node, f2q_layout)
        local_op = in_node.op.operator
        global_op = local_op.relabel_modes(freg_indices)
        pauli_op = self.mapper_fn(global_op, len(qreg)).simplify()
        out_dag.apply_operation_back(PauliEvolutionGate(pauli_op, time=in_node.op.params[0]), qreg)
