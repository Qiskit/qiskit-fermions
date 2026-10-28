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

"""An optimization pass merging consecutive Evolution gates of the same operator."""

from __future__ import annotations

import numpy as np
from qiskit.dagcircuit import DAGOpNode

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library import Evolution

from ... import FermionicDAGCircuitPass


def _same_value(a: object, b: object) -> bool:
    """Whether two attribute values are equal, which ``==`` cannot say for arrays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b))
    return bool(a == b)


def _same_synthesis(a: object, b: object) -> bool:
    """Whether two synthesis methods are of one kind and hold identical settings.

    This compares the settings one by one rather than as a whole, because a method may hold an array
    (sampling weights, say), whose comparison has no single truth value.
    """
    if type(a) is not type(b):
        return False
    settings_a, settings_b = vars(a), vars(b)
    return settings_a.keys() == settings_b.keys() and all(
        _same_value(settings_a[key], settings_b[key]) for key in settings_a
    )


def _mergeable(first: DAGOpNode, second: DAGOpNode, atol: float) -> bool:
    """Whether two neighbouring :class:`.Evolution` nodes combine into a single one."""
    a, b = first.op, second.op

    # the operator's mode labels are relative to the gate's wires, so their order must agree
    if first.qargs != second.qargs:
        return False
    if type(a.operator) is not type(b.operator):
        return False
    # an atomic gate must stay atomic, and a decomposable one must keep the way it decomposes
    if a.atomic != b.atomic:
        return False
    if not a.atomic and not _same_synthesis(a.synthesis, b.synthesis):
        return False

    return bool(a.operator.equiv(b.operator, atol))


class MergeEvolutions(FermionicDAGCircuitPass):
    r"""A transpilation pass merging consecutive :class:`.Evolution` gates of the same operator.

    Two evolutions under the same operator :math:`H` satisfy

    .. math::

        e^{-i t_2 H} e^{-i t_1 H} = e^{-i (t_1 + t_2) H},

    because :math:`H` commutes with itself. This pass finds neighbouring :class:`.Evolution` gates
    of equivalent operators and replaces them by one gate whose evolution time is the sum of theirs.
    The rewrite is exact, so it changes neither the simulated state nor the Trotter error of the
    circuit; it only removes a gate, and with it the synthesized operations that gate would have
    produced.

    Such neighbours arise wherever two sequences of factors meet on the same factor. Examples are the
    seams between the repetitions of :meth:`.FermionicCircuit.repeat`, between consecutive
    :class:`.UCC` or :class:`.UCJ` layers, or between consecutive samples of
    :class:`.QDriftTrotterization`. :class:`.FermionicSuzukiTrotter` already fuses the seams within
    its own output (see :attr:`.FermionicSuzukiTrotter.merge_adjacent`); this pass is the general
    mechanism which also acts across gate boundaries.

    Two gates merge only when all of the following hold:

    - they act on the identical wires, in the identical order, with nothing else in between;
    - their operators are of the same type and equivalent up to :attr:`atol`;
    - they are either both :attr:`.Evolution.atomic` or both not, and in the latter case they are
      decomposed by the same kind of :attr:`.Evolution.synthesis` with identical settings.

    The merged gate keeps the operator, and thus the
    :attr:`~qiskit_fermions.operators.OperatorTrait.groups`, of the first gate of the run.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.

    .. seealso::
       :class:`.Evolution`, :class:`.MergeOrbitalRotations`.
    """

    def __init__(self, atol: float = 1e-12) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            atol: the absolute tolerance up to which the operators of two gates are considered
                equivalent.
        """
        super().__init__()

        self.atol = atol
        """The absolute tolerance up to which two operators are considered equivalent."""

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Collects every maximal run of consecutive :class:`.Evolution` gates on the same wires,
        splits it into stretches of mergeable gates and replaces each stretch of length two or more
        with a single :class:`.Evolution` carrying the summed time. Everything else is left
        untouched. The input DAG is modified in place.

        A :class:`~qiskit.circuit.library.Barrier` interrupts a run, so the gates on either side of
        one are not merged. Place a barrier between two gates to keep them apart.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported, plus
                :class:`~qiskit.circuit.library.Barrier` and :class:`.FermionicMeasure`, which are
                carried through untouched (a measurement keeps the classical bit it writes into).

        Returns:
            The output circuit which is still acting on a fermionic register.
        """
        # `collect_runs` returns the runs already collected (a set), not a lazy view, and the runs
        # are disjoint, so replacing one block in place does not affect the others. No circuit
        # order is needed across runs for the same reason; within a run the nodes are ordered.
        for run in dag.collect_runs(["Evolution"]):
            stretches: list[list[DAGOpNode]] = [[run[0]]]
            for node in run[1:]:
                if _mergeable(stretches[-1][-1], node, self.atol):
                    stretches[-1].append(node)
                else:
                    stretches.append([node])

            for stretch in stretches:
                if len(stretch) < 2:
                    continue

                first = stretch[0].op
                merged = Evolution(
                    first.num_modes,
                    first.operator,
                    time=sum(node.op.params[0] for node in stretch),
                    synthesis=first.synthesis,
                    atomic=first.atomic,
                )
                wire_pos = {qubit: index for index, qubit in enumerate(stretch[0].qargs)}
                dag.replace_block_with_op(stretch, merged, wire_pos, cycle_check=False)

        return dag
