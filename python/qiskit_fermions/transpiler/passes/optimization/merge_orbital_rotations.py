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

"""An optimization pass merging consecutive OrbitalRotation gates into a single one."""

from __future__ import annotations

import numpy as np

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library import OrbitalRotation

from ... import FermionicDAGCircuitPass


class MergeOrbitalRotations(FermionicDAGCircuitPass):
    r"""A transpilation pass merging runs of consecutive orbital rotations into a single one.

    Two :class:`.OrbitalRotation` gates applied back-to-back on the same modes compose into a single
    orbital rotation whose ``rotation_unitary`` is the matrix product of the two. This pass detects
    maximal runs of such consecutive rotations in a :class:`.FermionicDAGCircuit` and rewrites each
    into one :class:`.OrbitalRotation`, so the synthesis stage lowers a single (still square)
    decomposition instead of one per gate in the run.

    A "run" is a maximal chain of :class:`.OrbitalRotation` nodes acting on the *same* mode set with
    nothing else intervening on those wires. Following the :class:`.OrbitalRotation` convention
    :math:`a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j`, applying a rotation :math:`U_1` and then a
    rotation :math:`U_2` maps :math:`a^\dagger_i \mapsto \sum_j (U_2 U_1)_{ji} a^\dagger_j`, so the
    run's rotations are multiplied in circuit order (later rotation on the left) to form the merged
    ``rotation_unitary``. Because the merged gate implements exactly that composed basis change, the
    rewrite leaves the simulated state vector unchanged.

    Only rotations on the *identical* mode set are merged: two rotations on different (even
    overlapping) mode sets, or rotations separated by any other operation on a shared wire, break the
    run and are left untouched. This is the shape produced, e.g., by two adjacent per-spin rotations
    on the same spin half, or by consecutive :class:`.UCJ` layers whose trailing and leading orbital
    rotations meet on the same modes.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.

    .. seealso::
       :class:`.OrbitalRotation`.
    """

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Collects every maximal run of consecutive :class:`.OrbitalRotation` gates on the same modes
        and replaces each run of length two or more with a single :class:`.OrbitalRotation` whose
        ``rotation_unitary`` is the run's rotations multiplied in circuit order. Runs of a single
        rotation (and all other nodes) are left untouched. The input DAG is modified in place.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported.

        Returns:
            The output circuit which is still acting on a fermionic register.
        """
        for run in dag.collect_runs(["OrbitalRotation"]):
            if len(run) < 2:
                continue

            # ``collect_runs`` guarantees every node in the run acts on the same wires, so all their
            # rotations share a dimension. Multiply them in circuit order -- the later rotation
            # multiplies from the left, mirroring the composed basis change
            # ``a^_i -> sum_j (U_k ... U_1)_{ji} a^_j``.
            combined = np.eye(run[0].op.num_modes, dtype=complex)
            for node in run:
                combined = node.op.rotation_unitary @ combined

            wire_pos = {qubit: index for index, qubit in enumerate(run[0].qargs)}
            dag.replace_block_with_op(run, OrbitalRotation(combined), wire_pos, cycle_check=False)

        return dag
