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

"""An optimization pass fusing InitializeModes + OrbitalRotation into PrepareSlaterDeterminant."""

from __future__ import annotations

import numpy as np
from qiskit.dagcircuit import DAGOpNode

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library import (
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)

from ... import FermionicDAGCircuitPass


class MergeSlaterDeterminantPreparation(FermionicDAGCircuitPass):
    r"""A transpilation pass fusing an initialization and rotation into a Slater determinant prep.

    An :class:`.InitializeModes` immediately followed by an :class:`.OrbitalRotation` prepares a
    Slater determinant: the modes start in the reference occupation and are then rotated into the
    target single-particle basis. This pass detects that pattern in a :class:`.FermionicDAGCircuit`
    and rewrites it into a single :class:`.PrepareSlaterDeterminant` gate, which a later synthesis
    stage can lower with the reduced-gate-count :func:`.givens_decomposition_slater` (via
    :class:`.GivensDecompositionSlaterDeterminantSynthesis`) rather than the full square orbital
    rotation.

    The rewrite is state-preserving: it only unlocks the cheaper synthesis and leaves the prepared
    state unchanged (see the *validate-then-rotate* semantics of :class:`.PrepareSlaterDeterminant`).

    Three patterns are recognized, all keyed off the block-spin mode convention
    (modes ``0..norb`` are the alpha sector, ``norb..2*norb`` the beta sector):

    1. **Full-register / spinless** -- an :class:`.InitializeModes` on a mode set immediately
       followed by an :class:`.OrbitalRotation` on the *same* mode set fuses into one
       :class:`.PrepareSlaterDeterminant`.
    2. **Per-sector** -- the same shape as pattern 1 but on a single spin half; it fuses into one
       :class:`.PrepareSlaterDeterminant` per sector.
    3. **Global init + per-spin rotations** -- a full-register (``2*norb``) :class:`.InitializeModes`
       immediately followed by an :class:`.OrbitalRotation` on *either or both* contiguous spin halves
       (in either order) splits the occupation per sector and emits **two**
       :class:`.PrepareSlaterDeterminant` gates. A half that has no rotation is prepared with an
       identity rotation, which synthesizes to only the reference X gates -- exactly what the
       :class:`.InitializeModes` would have emitted for that half anyway -- so padding it costs no
       extra gates while still unlocking the reduced Slater synthesis on the rotated half. This is the
       shape produced by placing an :class:`.InitializeModes` (e.g.
       :meth:`.InitializeModes.from_hartree_fock`) at the front of a circuit and appending a decomposed
       :class:`.UCJ`, whose first per-spin rotations directly follow the initialization.

    .. seealso::
       The :ref:`Slater determinant preparation guide <merge_slater_determinant_explanation>` walks
       through each of these patterns with before/after circuit drawings.

    "Immediately followed" is understood over the DAG: an :class:`.OrbitalRotation` node fuses only
    when the :class:`.InitializeModes` is its *sole* predecessor across all of its modes, i.e. no
    other operation intervenes on those wires. Any arrangement not matching one of the three shapes
    above -- non-adjacent gates, mismatched mode sets, or an :class:`.OrbitalRotation` with no
    preceding :class:`.InitializeModes` -- is left untouched.

    .. important::
       Run this pass **after** :class:`.MergeOrbitalRotations`. The fusion contracts a *single*
       :class:`.OrbitalRotation` immediately following the :class:`.InitializeModes`. Faced with an
       initialization followed by a *run* of two or more consecutive rotations, this pass only sees
       the first rotation immediately following the initialization -- it fuses that one into a
       :class:`.PrepareSlaterDeterminant` but leaves the remaining rotations of the run as separate
       trailing :class:`.OrbitalRotation` gates, which synthesize with their full (phase-carrying)
       square decomposition. Running :class:`.MergeOrbitalRotations` first collapses the whole run
       into one rotation, so this pass can then contract the entire run into a single
       :class:`.PrepareSlaterDeterminant` and the cheaper Slater synthesis covers all of it. The
       preset Jordan-Wigner pipeline (:func:`.generate_preset_jw_pass_manager`) wires the two passes
       in this order.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.

    .. seealso::
       :class:`.PrepareSlaterDeterminant`, :class:`.InitializeModes`, :class:`.OrbitalRotation`,
       :class:`.MergeOrbitalRotations`, and
       :class:`.GivensDecompositionSlaterDeterminantSynthesis`.
    """

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Walks the input DAG in topological order, rewriting each matched
        :class:`.InitializeModes`-then-:class:`.OrbitalRotation` pattern into
        :class:`.PrepareSlaterDeterminant` gate(s) and copying every other node through unchanged.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported.

        Returns:
            The output circuit which is still acting on a fermionic register.

        Raises:
            NotImplementedError: when the provided input circuit has more than a single register.
        """
        if len(dag.qregs) > 1:
            raise NotImplementedError(
                "Cannot apply the MergeSlaterDeterminantPreparation pass to a circuit with more "
                "than a single register."
            )

        # Plan the rewrites before emitting anything: map each matched InitializeModes node to the
        # replacement gate(s) it produces, and collect the OrbitalRotation nodes those replacements
        # consume so they are not re-emitted when the topological walk reaches them. DAG nodes are
        # hashable and compare by identity (node id), so we key on the nodes directly.
        replacements: dict[DAGOpNode, list[tuple[PrepareSlaterDeterminant, list[int]]]] = {}
        consumed: set[DAGOpNode] = set()
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, InitializeModes):
                continue
            match = self._match(dag, node)
            if match is None:
                continue
            gates, consumed_rotations = match
            replacements[node] = gates
            consumed.update(consumed_rotations)

        out_dag = dag.copy_empty_like()
        (register,) = out_dag.qregs.values()
        for node in dag.topological_op_nodes():
            if node in consumed:
                # an OrbitalRotation already folded into a PrepareSlaterDeterminant at its init
                continue
            if node in replacements:
                for gate, modes in replacements[node]:
                    out_dag.apply_operation_back(gate, qargs=[register[m] for m in modes])
                continue
            out_dag.apply_operation_back(node.op, qargs=node.qargs)

        return out_dag

    def _match(
        self, dag: FermionicDAGCircuit, init_node: DAGOpNode
    ) -> tuple[list[tuple[PrepareSlaterDeterminant, list[int]]], list[DAGOpNode]] | None:
        """Matches one of the three fusion patterns rooted at an ``InitializeModes`` node.

        Returns ``None`` when nothing matches. Otherwise returns a pair ``(gates, consumed)`` where
        ``gates`` is the list of ``(PrepareSlaterDeterminant, modes)`` replacements to emit in place
        of the initialization and ``consumed`` the ``OrbitalRotation`` nodes folded into them.
        """
        init_modes = self._modes(dag, init_node)

        # gather the OrbitalRotation successors for which this InitializeModes is the sole predecessor
        # across all their wires (i.e. they immediately follow it, nothing intervening)
        rotations = [
            succ
            for succ in dag.quantum_successors(init_node)
            if isinstance(succ, DAGOpNode)
            and isinstance(succ.op, OrbitalRotation)
            and self._immediately_follows(dag, init_node, succ)
        ]

        if not rotations:
            # a bare InitializeModes with no rotation immediately following it: nothing to fuse
            return None

        occupation = init_node.op.occupation

        # Pattern 1 / 2: a single rotation on exactly the initialized modes.
        if len(rotations) == 1:
            (rotation,) = rotations
            if self._modes(dag, rotation) == init_modes:
                gate = PrepareSlaterDeterminant(occupation, rotation.op.rotation_unitary)
                return [(gate, init_modes)], [rotation]
            # otherwise fall through: it may be a per-spin rotation on one half of a full register

        # Pattern 3: a full-register init whose two contiguous spin halves are rotated by per-spin
        # OrbitalRotations. Fires with one *or* both halves rotated -- an unrotated half is prepared
        # with an identity rotation, which synthesizes to just the reference X gates (no extra gates),
        # so padding the missing half costs nothing while still unlocking the reduced Slater synthesis
        # on the rotated half.
        num_modes = len(init_modes)
        if num_modes % 2 != 0 or init_modes != list(range(num_modes)):
            return None
        norb = num_modes // 2
        alpha_modes = list(range(norb))
        beta_modes = list(range(norb, num_modes))

        by_modes = {tuple(self._modes(dag, rot)): rot for rot in rotations}
        alpha_rot = by_modes.get(tuple(alpha_modes))
        beta_rot = by_modes.get(tuple(beta_modes))
        # every gathered rotation must sit on exactly one of the two halves -- otherwise this is not
        # the per-spin shape (e.g. a rotation on a partial sub-range straddling the sector boundary).
        # At least one rotation exists here (bare inits returned above), so this also guarantees at
        # least one half is rotated.
        if len(rotations) != (alpha_rot is not None) + (beta_rot is not None):
            return None

        gates: list[tuple[PrepareSlaterDeterminant, list[int]]] = []
        consumed: list[DAGOpNode] = []
        for rot, modes, occ in (
            (alpha_rot, alpha_modes, occupation[:norb]),
            (beta_rot, beta_modes, occupation[norb:]),
        ):
            if rot is None:
                unitary = np.eye(norb, dtype=complex)
            else:
                unitary = rot.op.rotation_unitary
                consumed.append(rot)
            gates.append((PrepareSlaterDeterminant(occ, unitary), modes))

        return gates, consumed

    @staticmethod
    def _modes(dag: FermionicDAGCircuit, node: DAGOpNode) -> list[int]:
        """Returns the sorted global mode indices a node acts on."""
        return sorted(dag.find_bit(qubit).index for qubit in node.qargs)

    @staticmethod
    def _immediately_follows(
        dag: FermionicDAGCircuit, init_node: DAGOpNode, rotation: DAGOpNode
    ) -> bool:
        """Whether ``init_node`` is the sole operation feeding ``rotation``, nothing intervening.

        This holds exactly when the only quantum-predecessor *op node* of ``rotation`` is
        ``init_node``: any operation intervening on one of ``rotation``'s modes would appear as a
        different predecessor. (Register-boundary ``DAGInNode`` predecessors are filtered out first,
        so a rotation wire that the initialization does not cover contributes no op-predecessor
        rather than a disqualifying one -- this check therefore does *not* by itself guarantee the
        initialization covers all of the rotation's modes; :meth:`_match` enforces that separately
        via its per-pattern mode-set checks before fusing.)
        """
        predecessors = [
            pred for pred in dag.quantum_predecessors(rotation) if isinstance(pred, DAGOpNode)
        ]
        # DAG nodes compare by identity (node id), so `==` matches the same node even when the DAG
        # hands back a distinct wrapper instance for it.
        return len(predecessors) == 1 and predecessors[0] == init_node
