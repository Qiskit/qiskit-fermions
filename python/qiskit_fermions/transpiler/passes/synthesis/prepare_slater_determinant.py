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

"""Slater determinant preparation synthesis."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import XGate, XXPlusYYGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit_fermions.linalg import givens_decomposition_slater

from ... import F2QLayout
from ..utils import map_node_single_register


class GivensDecompositionSlaterDeterminantSynthesis:
    r"""A :class:`.F2QSynthesisPlugin` for transpiling :class:`.PrepareSlaterDeterminant`.

    This plugin exploits the known reference occupation to synthesize the gate with the rectangular
    :func:`~qiskit_fermions.linalg.givens_decomposition_slater`: only the :math:`m` occupied orbitals
    of the rotation need be realized, so at most :math:`m (n - m)` ``XXPlusYYGate`` rotations are
    emitted (versus the :math:`n (n - 1) / 2` brick-wall plus :math:`n` phase gates that
    :class:`.GivensDecompositionOrbitalRotationSynthesis` would use for the full
    :class:`.OrbitalRotation`). The reduced decomposition carries no diagonal phases -- a global phase
    and any rotation within the occupied space leave the prepared Slater determinant unchanged -- so
    the state is prepared correct up to a global phase.

    .. note::
       :func:`~qiskit_fermions.linalg.givens_decomposition_slater` is defined against a fixed
       *leading*-:math:`m` reference (the first :math:`m` modes occupied). The emitted ``XGate``\ s
       therefore always seed the first :math:`m` qubits, **not** the qubits pointed to by
       ``occupation`` -- the Givens sweep then transports the occupation onto the physical target
       orbitals. The positions of ``occupation``'s occupied modes enter only through which columns of
       ``rotation_unitary`` are selected (``rotation_unitary[:, occupied]``); their *count* :math:`m`
       determines the reference. The final prepared state is correct for any occupation, contiguous
       or not.

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
                ``in_node.op`` attribute `must` be of type :class:`.PrepareSlaterDeterminant`.
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

        occupation = in_node.op.occupation
        rotation_unitary = in_node.op.rotation_unitary

        # the occupied local modes select the occupied orbitals; their columns of the rotation
        # unitary (following a†_i ↦ Σ_j U_ji a†_j) span the prepared Slater determinant's occupied
        # space. ``givens_decomposition_slater`` takes them as the m rows of an m x n matrix.
        occupied = np.nonzero(occupation)[0]
        orbital_coeffs = rotation_unitary[:, occupied].T

        # the decomposition prepares the target from the reference in which the first m orbitals are
        # occupied; seed that reference by setting the first m modes of the register occupied. The
        # Givens sweep below then moves the occupation onto the physical target orbitals.
        for i in range(len(occupied)):
            out_dag.apply_operation_back(XGate(), (qreg[freg_indices[i]],))

        # apply the Givens rotations in order (same XXPlusYYGate convention as the square plugin).
        # No phase gates: the reduced decomposition carries none, so the state is prepared up to a
        # global phase (physically irrelevant for state preparation).
        for c, s, i, j in givens_decomposition_slater(orbital_coeffs):
            c_angle = np.acos(c)
            if not np.isclose(c_angle, 0.0):
                out_dag.apply_operation_back(
                    XXPlusYYGate(2 * c_angle, np.angle(s) - 0.5 * np.pi),
                    (qreg[freg_indices[i]], qreg[freg_indices[j]]),
                )
