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
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import XGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode


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

    def run(
        self,
        in_node: DAGOpNode,
        freg_indices: list[int],
        out_dag: DAGCircuit,
        qreg: QuantumRegister,
    ):
        """Runs this transpilation plugin.

        Args:
            in_node: the input fermion-based circuit instruction. When this plugin gets called, the
                ``in_node.op`` attribute `must` be of type :class:`.InitializeModes`.
            freg_indices: TODO.
            out_dag: the output qubit-based circuit.
            qreg: TODO.

        .. seealso::
           The documentation of :class:`.F2QSynthesisPlugin` for more detailed explanations of the
           arguments.

        Raises:
            NotImplementedError: when ``in_node`` acts on fermionic modes that are spread across
                multiple :type:`~qiskit_fermions.circuit.FermionicRegister` instances.
        """
        local_occupation = in_node.op.occupation
        global_occupied_indices = np.asarray(freg_indices)[np.nonzero(local_occupation)]
        for idx in global_occupied_indices:
            out_dag.apply_operation_back(XGate(), (qreg[idx],))
