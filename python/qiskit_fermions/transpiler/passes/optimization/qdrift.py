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

"""A qDRIFT Trotterization that skips the draws which cannot move a particle."""

from __future__ import annotations

import warnings

import numpy as np
from qiskit.dagcircuit import DAGOpNode

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library import (
    Evolution,
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)
from qiskit_fermions.circuit.library.synthesis import FermionicQDrift

from ... import FermionicDAGCircuitPass


def _global_modes(dag: FermionicDAGCircuit, node: DAGOpNode) -> np.ndarray:
    """Returns the global mode indices that ``node`` acts on, in the order of its ``qargs``."""
    return np.array([dag.find_bit(qubit).index for qubit in node.qargs])


class QDriftTrotterization(FermionicDAGCircuitPass):
    r"""A transpilation pass to Trotterize :class:`.Evolution` gates via qDRIFT, skipping trivial draws.

    The qDRIFT protocol itself is the synthesis method :class:`.FermionicQDrift`, which is on an
    equal footing with the other product formulas and is used through the :class:`.F2FSynthesis`
    pass. This pass adds the one thing a synthesis method cannot do, since it sees a single gate
    at a time: it tracks which modes are known to be occupied or unoccupied through the circuit, and
    rejects a drawn term unless its support couples a known-occupied mode with a known-unoccupied
    one, drawing a replacement in its place.

    That exists to stop the sampling from spending one of the ``num_terms`` slots on an excitation
    that cannot move a particle, and which therefore tells you nothing about the sampled bitstrings.
    It buys that with the protocol's convergence guarantee. To sample without rejecting anything,
    use :class:`.FermionicQDrift` with :class:`.F2FSynthesis` instead.

    .. warning::
       Filtering does not make the circuit shorter. ``num_terms`` is fixed, so a rejected draw is
       replaced rather than dropped, and the cheap term it would have contributed (a diagonal
       rotation, say) gives way to a coupling excitation that costs more to synthesize. Expect the
       filtered circuit to be deeper than the unfiltered one: the budget of sampled slots is what the
       filtering conserves, not the depth.

       Rejecting a drawn term and re-drawing renormalizes the sampling distribution over the
       accepted terms only, so the sampled product no longer averages to :math:`e^{-i t H}` for the
       Hamiltonian you passed in: the effective coefficient of every retained term is inflated by
       the reciprocal of the acceptance probability, and the rejected terms drop out. Neither
       :math:`\lambda` nor :math:`\delta` is adjusted to compensate, so the distortion does not
       cancel.

       Use it only when the sampled bitstrings are the quantity of interest, as in the SqDRIFT
       workflow of :ref:`sqdrift_getting_started`. Do not use it when the sampled circuits are used
       to estimate an expectation value, a time-evolved observable, or anything else that relies
       on the qDRIFT error bound: those results are biased by an amount the pass does not track.

    Filtering requires an :class:`.InitializeModes` or :class:`.PrepareSlaterDeterminant` gate to
    precede the :class:`.Evolution` gates being Trotterized, to seed the initial occupied and
    unoccupied mode sets. If none is found, or if the sets it seeds turn out to be entirely occupied
    or entirely unoccupied, filtering is skipped for that gate and a :class:`UserWarning` is emitted
    instead, and the gate is sampled as :class:`.FermionicQDrift` would.

    The sets are updated as the circuit is traversed: a drawn term is accepted only if its support
    intersects *both* sets, and once it is, every mode in its support becomes "uncertain" and is
    added to *both* sets, making it eligible to participate in either role for subsequent draws. Any
    :class:`.OrbitalRotation` gate does the same to every mode it acts on, since the rotation can
    mix it with any other mode in its support. A :class:`.PrepareSlaterDeterminant` gate is treated
    as its :class:`.InitializeModes` and :class:`.OrbitalRotation` components applied back-to-back:
    its ``occupation`` first seeds the sets, and then every mode it acts on is marked "uncertain".

    .. rubric:: Filtering diagnostics

    When the filtering actually filters a gate, the returned :class:`.FermionicDAGCircuit` records
    how many draws it discarded in its :attr:`~qiskit.dagcircuit.DAGCircuit.metadata`, under
    ``filter_trivial.discarded`` and ``filter_trivial.emitted``. Both hold one entry per filtered
    :class:`.Evolution` gate, in circuit order. Their ratio estimates the acceptance probability,
    and hence the factor by which the filtering inflated the coefficients of the terms it kept: an
    acceptance probability close to one means the filtering barely moved the distribution, while a
    small one means the retained terms were weighted far above their true share of the Hamiltonian.

    .. important::
       Neither field is present when no gate was filtered, which covers every case in which
       filtering was skipped with a :class:`UserWarning`. Read them defensively, for example with
       ``qcirc.metadata.get("filter_trivial.discarded")``. A discarded count of zero is different
       from an absent field: it says the filtering ran on that gate and accepted every draw, so it
       left the sampling distribution untouched.
    """

    MAX_SAMPLE_RETRIES = 1_000_000
    """The maximum number of consecutive rejected samples tolerated before :meth:`run` gives up and
    raises :class:`RuntimeError`. This guards against an infinite loop when the Hamiltonian's
    remaining terms cannot bridge the tracked occupied/unoccupied mode sets. That happens when both
    sets remain small and disjoint (few modes have been marked occupied or unoccupied, and none has
    yet become "uncertain") and no remaining term's support touches both."""

    def __init__(self, qdrift: FermionicQDrift) -> None:
        """Initializing this transpiler pass can be done with the argument below.

        Args:
            qdrift: the synthesis method that draws the terms. Its ``num_terms``, ``weights`` and
                ``rng`` apply as they do outside of this pass.
        """
        super().__init__()

        self.qdrift = qdrift
        """The synthesis method that draws the terms."""

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Each :class:`.Evolution` node is replaced by :attr:`~.FermionicQDrift.num_terms` sampled
        :class:`.Evolution` gates, one per accepted draw. The emitted gates are marked
        :attr:`.Evolution.atomic` and carry over the :attr:`.Evolution.synthesis` method of the node
        they replace. Nodes that are not :class:`.Evolution` gates are copied to the output
        unchanged. Since the sampling is random, the output varies between runs unless the ``rng``
        was seeded.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported, plus
                :class:`~qiskit.circuit.library.Barrier` and :class:`.FermionicMeasure`, which are
                carried through untouched (a measurement keeps the classical bit it writes into).

        Returns:
            The output circuit which is still acting on a fermionic register. When filtering
            actually ran, its :attr:`~qiskit.dagcircuit.DAGCircuit.metadata` also carries the
            ``filter_trivial.discarded`` and ``filter_trivial.emitted`` counts described in the
            class docstring.

        Raises:
            RuntimeError: if :attr:`MAX_SAMPLE_RETRIES` consecutive draws are rejected without any
                of them coupling the tracked occupied and unoccupied mode sets.
            ValueError: if :attr:`~.FermionicQDrift.weights` was supplied and its length does not
                match the number of groups (or terms) of an evolved operator.
        """
        out_dag = dag.copy_empty_like()

        # The sets of modes that are currently known to be occupied/unoccupied, respectively. Seeded
        # by any preceding `InitializeModes` gate(s); both remain empty until the first one is
        # encountered, which is how we distinguish "no InitializeModes seen yet" from "seeded, but
        # every mode landed in the same set" (see the warnings below). Once a term is accepted, the
        # modes it touches move into "uncertain" (see below), so they end up in *both* sets going
        # forward.
        occupied: set[int] = set()
        unoccupied: set[int] = set()

        # Per-gate rejection statistics, recorded into the output DAG's metadata below. One entry is
        # appended per `Evolution` gate that was actually filtered, so a gate whose filtering was
        # skipped contributes nothing and both lists stay empty when filtering is not in effect
        # anywhere.
        discarded_per_gate: list[int] = []
        emitted_per_gate: list[int] = []

        # circuit order, not storage order: the latter is out of sync with the wire dependencies
        # after a pass substituted a node by a sub-circuit. The output is rebuilt node by node and
        # the occupation tracking below depends on seeing the gates in the order they act.
        for node in dag.topological_op_nodes():
            if isinstance(node.op, InitializeModes):
                modes = _global_modes(dag, node)
                occupation = node.op.occupation
                # Several `InitializeModes` gates may be placed in parallel (e.g. one per spin
                # sector), so accumulate rather than overwrite.
                occupied |= set(modes[occupation].tolist())
                unoccupied |= set(modes[~occupation].tolist())

            elif isinstance(node.op, OrbitalRotation):
                # An `OrbitalRotation` mixes creation operators across all of the modes it acts
                # on (its `rotation_unitary` carries no per-mode sparsity information we could use
                # to do better - see the `OrbitalRotation` docstring), so every mode it touches
                # becomes "uncertain" just like an accepted qDRIFT term below: no longer known to
                # be occupied or unoccupied, and thus eligible for both roles going forward.
                modes = set(_global_modes(dag, node).tolist())
                occupied |= modes
                unoccupied |= modes

            elif isinstance(node.op, PrepareSlaterDeterminant):
                # This gate is the composition of an `InitializeModes` reference occupation
                # followed by an `OrbitalRotation` (see its class docstring), so it updates the
                # tracked sets the same way those two gates would back-to-back: first seed from
                # `occupation`, then immediately promote every mode it touches to "uncertain"
                # because of the rotation it also carries.
                modes = _global_modes(dag, node)
                occupation = node.op.occupation
                occupied |= set(modes[occupation].tolist())
                unoccupied |= set(modes[~occupation].tolist())
                all_modes = set(modes.tolist())
                occupied |= all_modes
                unoccupied |= all_modes

            if not isinstance(node.op, Evolution):
                # Forward ``cargs``: a non-``Evolution`` node is copied through as-is and may carry
                # classical bits (e.g. a ``FermionicMeasure``); ``apply_operation_back`` silently
                # defaults to none, which would drop them.
                out_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
                continue

            sampler = self.qdrift.sampler(node.op)

            if not (occupied and unoccupied):
                match (bool(occupied), bool(unoccupied)):
                    case (True, False):
                        reason = (
                            "the preceding InitializeModes gate(s) marked every mode as occupied, "
                            "so no unoccupied mode is available to filter against"
                        )
                    case (False, True):
                        reason = (
                            "the preceding InitializeModes gate(s) marked every mode as "
                            "unoccupied, so no occupied mode is available to filter against"
                        )
                    case _:
                        reason = (
                            "it is not preceded by an InitializeModes gate, so no occupation "
                            "information is available to filter against"
                        )
                warnings.warn(
                    f"The filtering has no effect on this Evolution gate because {reason}.",
                    category=UserWarning,
                    stacklevel=2,
                )
                for evo in sampler.sample():
                    out_dag.apply_operation_back(evo, qargs=out_dag.qubits)
                continue

            added_terms = 0
            # `failed_attempts` is reset on every acceptance because it guards against an infinite
            # loop, so it counts *consecutive* rejections only. `discarded` is the running total for
            # this gate, which is what the metadata reports.
            failed_attempts = 0
            discarded = 0
            while added_terms < self.qdrift.num_terms:
                evo = sampler.draw()

                term_support = evo.operator.get_support()
                # A term is retained when it couples a known-occupied mode with a known-unoccupied
                # one. This is a support-based over-approximation of "cannot change the sampled
                # bitstring", not an equivalence: see the class docstring's warning.
                if not (term_support & occupied and term_support & unoccupied):
                    failed_attempts += 1
                    discarded += 1
                    if failed_attempts > self.MAX_SAMPLE_RETRIES:
                        raise RuntimeError(
                            f"Failed to sample a non-trivial term after "
                            f"{self.MAX_SAMPLE_RETRIES} consecutive attempts. The remaining "
                            "Hamiltonian terms may no longer be able to couple the tracked "
                            "occupied and unoccupied mode sets."
                        )
                    continue
                failed_attempts = 0
                # The modes touched by an accepted term become "uncertain": we no longer know
                # whether they end up occupied or not, so they must be considered eligible for
                # both roles by subsequent samples.
                occupied |= term_support
                unoccupied |= term_support

                out_dag.apply_operation_back(evo, qargs=out_dag.qubits)
                added_terms += 1

            discarded_per_gate.append(discarded)
            emitted_per_gate.append(added_terms)

        # Only record the diagnostics when filtering actually ran, matching `RelabelModes`, which
        # leaves the metadata untouched when it has no effect. Callers must therefore read these
        # fields defensively (see the class docstring).
        if discarded_per_gate:
            out_dag.metadata["filter_trivial.discarded"] = discarded_per_gate
            out_dag.metadata["filter_trivial.emitted"] = emitted_per_gate

        return out_dag
