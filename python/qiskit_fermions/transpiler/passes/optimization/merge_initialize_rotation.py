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

    Because :class:`.PrepareSlaterDeterminant` is *validate-then-rotate* under simulation -- it
    validates the reference occupation and then applies the rotation, exactly as the separate
    :class:`.InitializeModes` and :class:`.OrbitalRotation` gates do -- the rewrite leaves the
    simulated state vector unchanged; it only unlocks the cheaper synthesis.

    Three patterns are recognized, all keyed off the block-spin mode convention
    (modes ``0..norb`` are the alpha sector, ``norb..2*norb`` the beta sector):

    1. **Full-register / spinless** -- an :class:`.InitializeModes` on a mode set immediately
       followed by an :class:`.OrbitalRotation` on the *same* mode set fuses into one
       :class:`.PrepareSlaterDeterminant`.
    2. **Per-sector** -- the same shape as pattern 1 but on a single spin half; it fuses into one
       :class:`.PrepareSlaterDeterminant` per sector.
    3. **Global init + per-spin rotations** -- a full-register (``2*norb``) :class:`.InitializeModes`
       immediately followed by two :class:`.OrbitalRotation`\ s, one on each contiguous spin half
       (in either order), splits the occupation per sector and emits **two**
       :class:`.PrepareSlaterDeterminant` gates. This is the shape produced by placing an
       :class:`.InitializeModes` (e.g. :meth:`.InitializeModes.from_hartree_fock`) at the front of a
       circuit and appending a decomposed :class:`.UCJ`, whose first two per-spin rotations directly
       follow the initialization.

    "Immediately followed" is understood over the DAG: an :class:`.OrbitalRotation` node fuses only
    when the :class:`.InitializeModes` is its *sole* predecessor across all of its modes, i.e. no
    other operation intervenes on those wires. Any arrangement not matching one of the three shapes
    above -- non-adjacent gates, mismatched mode sets, an :class:`.OrbitalRotation` with no preceding
    :class:`.InitializeModes`, or only one of the two per-spin rotations present -- is left untouched.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.

    .. seealso::
       :class:`.PrepareSlaterDeterminant`, :class:`.InitializeModes`, :class:`.OrbitalRotation`, and
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
        # consume so they are not re-emitted when the topological walk reaches them.
        replacements: dict[int, list[tuple[PrepareSlaterDeterminant, list[int]]]] = {}
        consumed: set[int] = set()
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, InitializeModes) or node._node_id in consumed:
                continue
            match = self._match(dag, node)
            if match is None:
                continue
            gates, consumed_rotations = match
            replacements[node._node_id] = gates
            consumed.update(consumed_rotations)

        out_dag = dag.copy_empty_like()
        (register,) = out_dag.qregs.values()
        for node in dag.topological_op_nodes():
            if node._node_id in consumed:
                # an OrbitalRotation already folded into a PrepareSlaterDeterminant at its init
                continue
            if node._node_id in replacements:
                for gate, modes in replacements[node._node_id]:
                    out_dag.apply_operation_back(gate, qargs=[register[m] for m in modes])
                continue
            out_dag.apply_operation_back(node.op, qargs=node.qargs)

        return out_dag

    def _match(
        self, dag: FermionicDAGCircuit, init_node: DAGOpNode
    ) -> tuple[list[tuple[PrepareSlaterDeterminant, list[int]]], list[int]] | None:
        """Matches one of the three fusion patterns rooted at an ``InitializeModes`` node.

        Returns ``None`` when nothing matches. Otherwise returns a pair ``(gates, consumed)`` where
        ``gates`` is the list of ``(PrepareSlaterDeterminant, modes)`` replacements to emit in place
        of the initialization and ``consumed`` the node ids of the ``OrbitalRotation`` nodes folded
        into them.
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

        occupation = init_node.op.occupation

        # Pattern 1 / 2: a single rotation on exactly the initialized modes.
        if len(rotations) == 1:
            (rotation,) = rotations
            if self._modes(dag, rotation) == init_modes:
                gate = PrepareSlaterDeterminant(occupation, rotation.op.rotation_unitary)
                return [(gate, init_modes)], [rotation._node_id]
            return None

        # Pattern 3: a full-register init split by two per-spin rotations on the two contiguous halves.
        if len(rotations) == 2:
            num_modes = len(init_modes)
            if num_modes % 2 != 0 or init_modes != list(range(num_modes)):
                return None
            norb = num_modes // 2
            alpha_modes = list(range(norb))
            beta_modes = list(range(norb, num_modes))

            by_modes = {tuple(self._modes(dag, rot)): rot for rot in rotations}
            alpha_rot = by_modes.get(tuple(alpha_modes))
            beta_rot = by_modes.get(tuple(beta_modes))
            if alpha_rot is None or beta_rot is None:
                return None

            alpha_gate = PrepareSlaterDeterminant(occupation[:norb], alpha_rot.op.rotation_unitary)
            beta_gate = PrepareSlaterDeterminant(occupation[norb:], beta_rot.op.rotation_unitary)
            return (
                [(alpha_gate, alpha_modes), (beta_gate, beta_modes)],
                [alpha_rot._node_id, beta_rot._node_id],
            )

        return None

    @staticmethod
    def _modes(dag: FermionicDAGCircuit, node: DAGOpNode) -> list[int]:
        """Returns the sorted global mode indices a node acts on."""
        return sorted(dag.find_bit(qubit).index for qubit in node.qargs)

    @staticmethod
    def _immediately_follows(
        dag: FermionicDAGCircuit, init_node: DAGOpNode, rotation: DAGOpNode
    ) -> bool:
        """Whether ``rotation`` directly follows ``init_node`` on every one of its wires.

        This holds exactly when the only quantum predecessor of ``rotation`` is ``init_node``: any
        intervening operation on one of ``rotation``'s modes would appear as a different predecessor,
        so requiring a single predecessor guarantees the initialization feeds the rotation directly
        and covers all of its modes.
        """
        predecessors = [
            pred for pred in dag.quantum_predecessors(rotation) if isinstance(pred, DAGOpNode)
        ]
        return len(predecessors) == 1 and predecessors[0]._node_id == init_node._node_id
