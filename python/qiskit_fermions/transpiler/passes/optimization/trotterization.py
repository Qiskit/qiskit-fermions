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

"""An optimization pass selecting the fermionic synthesis of Evolution gates."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qiskit.dagcircuit import DAGOpNode

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library import Evolution

from ... import FermionicDAGCircuitPass

if TYPE_CHECKING:
    from qiskit_fermions.circuit.library.synthesis import FermionicEvolutionSynthesis


class FermionicTrotterization(FermionicDAGCircuitPass):
    r"""A transpilation pass selecting the fermion-to-fermion synthesis of :class:`.Evolution` gates.

    An :class:`.Evolution` gate carries the synthesis method with which it gets decomposed in fermionic
    space (see :attr:`.Evolution.synthesis`). Setting it per gate means threading the choice through
    everything that constructs one -- including :class:`.UCC` and :class:`.UCJ`, which build their own
    internally. This pass applies one method to every :class:`.Evolution` in a circuit instead, so the
    choice can be made once for a whole transpilation pipeline:

    .. code-block:: python

       pm.optimization = FermionicPassManager(
           [FermionicTrotterization(FermionicSuzukiTrotter(order=2, reps=4))]
       )

    Nodes that are not :class:`.Evolution` gates are left untouched, as are those rejected by an
    optional :attr:`filter`.

    .. note::
       The pass *selects* a synthesis method rather than expanding the evolution there and then. The
       expansion happens later, when the gate's definition is built, which keeps each
       :class:`.Evolution` intact as a single node for the passes that follow -- notably
       :class:`.RelabelModes`, which reads the operator of every :class:`.Evolution` to build its
       mode-relabeling model and would otherwise see a fragment per factor.

       This is the opposite choice from :class:`.QDriftTrotterization`, which replaces each gate with
       its sampled factors immediately. That pass has no alternative: its sampling is random and
       one-shot, so deferring it would draw a different sample every time the definition were rebuilt.
       A deterministic product formula is a pure function of the gate and can safely be deferred.

    .. caution::
       Not every synthesis method suits every operator. An :class:`.Evolution` whose operator groups
       all mutually commute (the diagonal-Coulomb operators of a :class:`.UCJ`, for example) is
       synthesized exactly at any order, so a higher order only adds depth. Use :attr:`filter` to
       restrict the pass to the gates that benefit.
    """

    def __init__(
        self,
        synthesis: FermionicEvolutionSynthesis,
        *,
        filter: Callable[[DAGOpNode], bool] | None = None,  # noqa: A002
    ) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            synthesis: the fermion-to-fermion synthesis method to apply to the :class:`.Evolution`
                gates of the circuit.
            filter: an optional predicate deciding which :class:`.Evolution` nodes to apply
                ``synthesis`` to. It is called with the :class:`~qiskit.dagcircuit.DAGOpNode` and the
                node is left untouched unless it returns ``True``. If ``None`` (the default), every
                :class:`.Evolution` node is selected.
        """
        super().__init__()

        self.synthesis = synthesis
        """The fermion-to-fermion synthesis method applied to the selected gates."""

        self.filter = filter
        """The predicate selecting which :class:`.Evolution` nodes to apply :attr:`synthesis` to."""

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Every :class:`.Evolution` node accepted by :attr:`filter` is replaced by an equivalent gate
        carrying :attr:`synthesis`. All other nodes are left untouched. The input DAG is modified in
        place.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported.

        Returns:
            The output circuit which is still acting on a fermionic register.
        """
        for node in dag.op_nodes():
            if not isinstance(node.op, Evolution):
                continue
            if self.filter is not None and not self.filter(node):
                continue

            # A *new* gate rather than an assignment to `node.op`: a gate instance can be shared
            # between circuits (Qiskit copies gates with a shallow `__dict__` copy), so mutating it
            # would retroactively change the synthesis of every other circuit holding the same object.
            dag.substitute_node(
                node,
                Evolution(
                    node.op.num_modes,
                    node.op.operator,
                    time=node.op.params[0],
                    synthesis=self.synthesis,
                    atomic=node.op.atomic,
                ),
                inplace=True,
            )

        return dag
