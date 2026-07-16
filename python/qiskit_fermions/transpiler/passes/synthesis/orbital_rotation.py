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

"""Orbital rotation synthesis."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import PhaseGate, XXPlusYYGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit_fermions.linalg import givens_decomposition

from ... import F2QLayout
from ..utils import map_node_single_register


class GivensDecompositionOrbitalRotationSynthesis:
    """A :class:`.F2QSynthesisPlugin` for transpiling :class:`.OrbitalRotation` into Givens rotations.

    .. warning::
       This transpilation pass plugin makes the following assumptions:

       - an occupation-basis encoding (like Jordan-Wigner)
       - a trivial fermion-to-qubit layout (i.e. no change in their register lengths)
       - a 1-to-1 mapping of fermionic mode indices to qubit indices
    """

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
        """Runs this transpilation plugin.

        Args:
            in_node: the input fermion-based circuit instruction. When this plugin gets called, the
                ``in_node.op`` attribute `must` be of type :class:`.OrbitalRotation`.
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
        givens_rotations, phase_shifts = givens_decomposition(in_node.op.rotation_unitary)
        for c, s, i, j in givens_rotations:
            c_angle = np.acos(c)
            if not np.isclose(c_angle, 0.0):
                out_dag.apply_operation_back(
                    XXPlusYYGate(2 * c_angle, np.angle(s) - 0.5 * np.pi),
                    (qreg[freg_indices[i]], qreg[freg_indices[j]]),
                )
        for i, p in enumerate(phase_shifts):
            p_angle = np.angle(p)
            if not np.isclose(p_angle, 0.0):
                out_dag.apply_operation_back(PhaseGate(p_angle), (qreg[freg_indices[i]],))
