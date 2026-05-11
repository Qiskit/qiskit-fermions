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

"""Mode initialization synthesis."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from ... import F2QLayout
from ..utils import _parse_node_indices


class InitializeModesSynthesis:
    """A :class:`.F2QSynthesisPlugin` for transpiling a :class:`.InitializeModes`.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.

    .. warning::
       This transpilation pass plugin is known to have certain limitations, including:

       - assuming an occupation-basis encoding (like Jordan-Wigner)
       - assuming a trivial fermion-to-qubit layout (i.e. no change in their register lengths)
       - assuming a 1-to-1 mapping of fermionic mode indices to qubit indices
    """

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout):
        """Runs this transpilation plugin.

        Args:
            in_node: the input fermion-based circuit instruction. When this plugin gets called, the
                ``in_node.op`` attribute `must` be of type :class:`.InitializeModes`.
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
        encountered_fermionic_registers, global_mode_indices = _parse_node_indices(
            in_node, f2q_layout
        )

        if len(encountered_fermionic_registers) > 1:
            raise NotImplementedError(
                "Cannot map an InitializeModes gate acting on fermionic modes that are spread "
                "across multiple FermionicRegister instances."
            )

        freg = encountered_fermionic_registers.pop()
        qreg = f2q_layout[freg]

        circ = QuantumCircuit(qreg)

        local_occupation = in_node.op.occupation
        global_occupied_indices = np.asarray(global_mode_indices)[np.nonzero(local_occupation)]
        circ.x(global_occupied_indices.tolist())

        new_dag = circuit_to_dag(circ)

        out_dag.compose(new_dag, qubits=list(qreg), front=False, inplace=True)
