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

"""A QDrift Trotterization optimization pass."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from qiskit.dagcircuit import DAGOpNode

from qiskit_fermions.circuit import FermionicDAGCircuit
from qiskit_fermions.circuit.library import (
    Evolution,
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)

from ... import FermionicDAGCircuitPass


def _global_modes(dag: FermionicDAGCircuit, node: DAGOpNode) -> np.ndarray:
    """Returns the global mode indices that ``node`` acts on, in the order of its ``qargs``."""
    return np.array([dag.find_bit(qubit).index for qubit in node.qargs])


class QDriftTrotterization(FermionicDAGCircuitPass):
    r"""A transpilation pass to Trotterize :class:`.Evolution` gates via the qDRIFT protocol.

    This pass replaces the exact evolution :math:`e^{-i t H}` of each :class:`.Evolution` gate by a
    randomized product formula: it draws ``num_terms`` samples from the Hamiltonian's terms (or
    :attr:`~qiskit_fermions.operators.FermionOperator.groups`, if assigned), with each term sampled
    with a probability proportional to the magnitude of its coefficient, and emits one
    :class:`.Evolution` gate per sample. Every sampled gate evolves its (unit-magnitude,
    sign-preserving) term for the same time

    .. math::

        \delta = \frac{\lambda t}{\texttt{num\_terms}}, \qquad \lambda = \sum_j |c_j|,

    where the :math:`c_j` are the coefficients of the sampled terms/groups. The ordered product of
    the sampled evolutions does not reproduce :math:`e^{-i t H}` exactly; rather, its expectation
    over the sampling approximates the exact evolution, with an error that decreases as
    ``num_terms`` grows. Because the output depends on the random draws, it differs from run to run
    unless a fixed ``rng`` is supplied.

    .. hint::
       Terms that are diagonal in the occupation-number basis (i.e. products of number operators)
       have no effect on the sampled bitstrings, so including them only increases the sampling
       overhead. Filter them out with
       :func:`~qiskit_fermions.operators.terms.filtering.filter_diagonal_terms` on the Hamiltonian
       *before* constructing the :class:`.Evolution` gate, rather than on every call to :meth:`run`:
       this pass runs once per transpiled circuit, so filtering upstream avoids repeating the same
       filtering work for every circuit generated from the same Hamiltonian.

    .. seealso::
       The qDRIFT protocol was introduced in `arXiv:1811.08017 <https://arxiv.org/abs/1811.08017>`_.
    """

    MAX_SAMPLE_RETRIES = 1_000_000
    """The maximum number of consecutive rejected samples tolerated by ``filter_trivial`` before
    :meth:`run` gives up and raises :class:`RuntimeError`. This guards against an infinite loop when
    the Hamiltonian's remaining terms cannot bridge the tracked occupied/unoccupied mode sets — for
    example, when both sets remain small and disjoint (few modes have been marked occupied or
    unoccupied, and none have yet become "uncertain") and no remaining term's support touches both."""

    def __init__(
        self,
        num_terms: int,
        *,
        filter_trivial: bool = False,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            num_terms: the number of terms to sample for the qDRIFT Trotterization. This equals the
                number of :class:`.Evolution` gates emitted per input gate; a larger value reduces
                the Trotterization error at the cost of a deeper circuit.
            filter_trivial: when set to ``True``, the sampling loop rejects a sampled term unless it
                couples a mode known to be occupied with a mode known to be unoccupied. Any term
                acting only within one of these two sets cannot change the occupation and, thus, has
                no effect on a sampled bitstring, so re-drawing avoids wasting one of the
                ``num_terms`` slots on it. This requires an :class:`.InitializeModes` or
                :class:`.PrepareSlaterDeterminant` gate to precede the :class:`.Evolution` gates
                being Trotterized (to seed the initial occupied and unoccupied mode sets); if none is
                found, or if the mode sets it seeds turn out to be entirely occupied or entirely
                unoccupied, filtering is skipped for that gate and a :class:`UserWarning` is emitted
                instead. Any :class:`.OrbitalRotation` gate encountered before or between the
                :class:`.Evolution` gates also updates these sets: every mode it acts on becomes
                "uncertain" (since the rotation may mix it with any other mode it touches), just like
                a mode touched by an accepted qDRIFT term. A :class:`.PrepareSlaterDeterminant` gate
                updates these sets the same way its :class:`.InitializeModes` and
                :class:`.OrbitalRotation` components would if applied in sequence: it seeds the
                occupied/unoccupied sets from its ``occupation``, then immediately marks every mode
                it acts on as "uncertain" because of its rotation. See the :meth:`run` docstring for
                the precise acceptance rule.
            rng: the random number generator (rng) to be used. When this is an ``int``, the internal
                rng will be initialized with ``np.random.default_rng(seed=rng)``.
        """
        super().__init__()

        self.num_terms = num_terms
        """The number of terms to include in the qDRIFT Trotterization."""

        self.filter_trivial = filter_trivial
        """Whether to reject sampled terms that cannot affect the sampled bitstring (see the
        class docstring for the ``filter_trivial`` argument)."""

        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Each :class:`.Evolution` node is replaced by ``num_terms`` sampled single-term
        :class:`.Evolution` gates (see the class docstring). Nodes that are not :class:`.Evolution`
        gates are copied to the output unchanged. Since the sampling is random, the output varies
        between runs unless the ``rng`` was seeded.

        When :attr:`filter_trivial` is set, this method tracks the sets of modes that are known to
        be occupied or unoccupied, seeded from any :class:`.InitializeModes` gate(s) preceding the
        :class:`.Evolution` gates in the circuit (several such gates placed in parallel, e.g. one per
        spin sector, are accumulated together). A sampled term is only accepted if its support
        intersects *both* sets, i.e. it couples a known-occupied mode with a known-unoccupied one;
        otherwise it is discarded and re-sampled, since it cannot affect the sampled bitstring. Once a
        term is accepted, every mode in its support becomes "uncertain" and is added to *both* sets,
        making it eligible to participate in either role for subsequent samples. Any
        :class:`.OrbitalRotation` gate found in the circuit updates these sets the same way: every
        mode it acts on becomes "uncertain" too, since the rotation may mix it with any other mode
        in its support. A :class:`.PrepareSlaterDeterminant` gate is treated as its
        :class:`.InitializeModes` and :class:`.OrbitalRotation` components applied back-to-back: its
        ``occupation`` first seeds the occupied/unoccupied sets, and then every mode it acts on is
        immediately marked "uncertain", since it also carries a rotation.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported.

        Returns:
            The output circuit which is still acting on a fermionic register.

        Raises:
            RuntimeError: if ``filter_trivial`` is ``True`` and :attr:`MAX_SAMPLE_RETRIES`
                consecutive samples are rejected without finding a non-trivial term to emit.
        """
        out_dag = dag.copy_empty_like()

        # The sets of modes that are currently known to be occupied/unoccupied, respectively. Seeded
        # by any preceding `InitializeModes` gate(s); both remain empty until the first one is
        # encountered, which is how we distinguish "no InitializeModes seen yet" from "seeded, but
        # every mode landed in the same set" (see the two warnings below). `filter_trivial` uses these
        # sets to reject sampled terms that cannot change the occupation. Once a term is accepted, the
        # modes it touches move into "uncertain" (see below), so they end up in *both* sets going
        # forward.
        occupied: set[int] = set()
        unoccupied: set[int] = set()

        for node in dag.op_nodes():
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
                out_dag.apply_operation_back(node.op, qargs=node.qargs)
                continue

            hamil = node.op.operator
            time = node.op.params[0]
            num_modes = len(node.qargs)

            terms: list[Any]  # can be either a list of operator terms or operator instances
            if hamil.groups is None:
                # NOTE: the qDRIFT protocol normalizes each term to unit magnitude because the
                # evolution time is entirely dictated by `delta` (computed below). Only the
                # magnitude of a coefficient sets its sampling probability, but its sign fixes the
                # direction of the rotation and must be preserved for the Trotterization to
                # approximate the target time evolution.
                terms = [(actions, np.sign(coeff)) for actions, coeff in hamil.iter_terms()]
                weights = np.abs(hamil.get_coeffs())
            else:
                all_coeffs = hamil.get_coeffs()
                groups = hamil.groups
                weights = np.zeros((hamil.num_groups(),))
                np.add.at(weights, groups, np.abs(all_coeffs))
                weights /= np.unique(groups, return_counts=True)[1]
                # NOTE: we do not pre-process the coefficients of these different group terms here,
                # because we would unnecessarily need to loop over all terms in all groups. Instead,
                # we replace the coefficients by identity values only once that particular group
                # term actually gets sampled.
                terms = hamil.split_out_groups()

            def _unit_terms(term):
                if isinstance(term, tuple):
                    # in this case, we have already normalized the coefficient to its sign
                    return [term]

                # NOTE: as per the comment earlier, we have not yet normalized the coefficients
                # of grouped operator terms. Keep each term's sign (dropping only its magnitude)
                # so that the sampled rotation points in the correct direction.
                return [(actions, np.sign(coeff)) for actions, coeff in term.iter_terms()]

            lambd = np.sum(weights)
            delta = (lambd * time) / self.num_terms
            probabilities = weights / lambd

            filter_trivial = self.filter_trivial and (bool(occupied) and bool(unoccupied))
            if self.filter_trivial and not filter_trivial:
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
                    f"filter_trivial=True has no effect on this Evolution gate because {reason}.",
                    category=UserWarning,
                    stacklevel=2,
                )

            if not filter_trivial:
                # No rejection sampling is needed, so we can draw every index for this gate in a
                # single batched call instead of `num_terms` separate scalar draws, which is
                # considerably faster (each scalar `choice()` call rebuilds the cumulative-
                # probability structure from scratch, whereas a batched call builds it once).
                sampled_indices = self._rng.choice(
                    np.arange(len(weights)), size=self.num_terms, p=probabilities
                )
                for sampled_idx in sampled_indices:
                    unit_terms = _unit_terms(terms[sampled_idx])
                    op = hamil.__class__.from_terms(unit_terms)
                    evo = Evolution(num_modes, op, time=delta)
                    out_dag.apply_operation_back(evo, qargs=out_dag.qubits)
                continue

            # Precomputed once here (only reached once rejection sampling is actually needed) so
            # the loop below can draw against it directly via `searchsorted` instead of calling
            # `rng.choice(..., p=probabilities)`, which would rebuild this same cumulative sum from
            # scratch on every single draw.
            cdf = np.cumsum(probabilities)
            cdf /= cdf[-1]  # guards against float-sum drift from 1.0, matching numpy's own choice()

            added_terms = 0
            failed_attempts = 0
            while added_terms < self.num_terms:
                # Equivalent to `self._rng.choice(np.arange(len(weights)), p=probabilities)`:
                # this is numpy's own implementation of weighted sampling (see `Generator.choice`),
                # just reusing the `cdf` precomputed above instead of rebuilding it on every draw.
                sampled_idx = cdf.searchsorted(self._rng.random(), side="right")
                unit_terms = _unit_terms(terms[sampled_idx])
                op = hamil.__class__.from_terms(unit_terms)

                term_support = op.get_support()
                # The term is non-trivial exactly when it couples a known-occupied mode with a
                # known-unoccupied one; otherwise it cannot change which bitstring gets sampled.
                if not (term_support & occupied and term_support & unoccupied):
                    failed_attempts += 1
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

                evo = Evolution(num_modes, op, time=delta)
                out_dag.apply_operation_back(evo, qargs=out_dag.qubits)
                added_terms += 1

        return out_dag
