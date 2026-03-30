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
from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import SparseObservable

from ... import F2QLayout

if TYPE_CHECKING:
    from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
    from qiskit_fermions._lib.operators.majorana_operator import MajoranaOperator
else:
    from qiskit_fermions.operators import FermionOperator, MajoranaOperator


# TODO: add an OperatorProtocol to avoid hard-coding this list of types from our package
MapperFunction = Callable[[FermionOperator | MajoranaOperator], SparseObservable]
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
                multiple :type:`~qiskit_fermions.circuit.FermionRegister` instances.
        """
        local_op = in_node.op.operator
        # First, we must expand the local node indices to the global fermion register
        encountered_fermion_registers: set[QuantumRegister] = set()
        global_fermion_indices = []
        for fermion in in_node.qargs:
            for freg in f2q_layout:
                if fermion in freg:
                    encountered_fermion_registers.add(freg)
                    global_fermion_indices.append(freg.index(fermion))
                    break

        if len(encountered_fermion_registers) > 1:
            # TODO: improve error message
            raise NotImplementedError("Multiple fermion registers not supported!")

        freg = encountered_fermion_registers.pop()
        qreg = f2q_layout[freg]

        global_op = local_op.relabel_modes(global_fermion_indices)
        pauli_op = self.mapper_fn(global_op).simplify()

        circ = QuantumCircuit(qreg)
        circ.append(PauliEvolutionGate(pauli_op, time=in_node.op.params[0]), qreg)
        new_dag = circuit_to_dag(circ)

        out_dag.compose(new_dag, qubits=list(qreg), front=False, inplace=True)
