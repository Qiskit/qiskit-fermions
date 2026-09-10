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
from collections.abc import Sequence
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
from qiskit_fermions.operators.terms.grouping import group_coeff_means

from ... import FermionicDAGCircuitPass


def _global_modes(dag: FermionicDAGCircuit, node: DAGOpNode) -> np.ndarray:
    """Returns the global mode indices that ``node`` acts on, in the order of its ``qargs``."""
    return np.array([dag.find_bit(qubit).index for qubit in node.qargs])


class QDriftTrotterization(FermionicDAGCircuitPass):
    r"""A transpilation pass to Trotterize :class:`.Evolution` gates via the qDRIFT protocol.

    This pass replaces the exact evolution :math:`e^{-i t H}` of each :class:`.Evolution` gate by a
    randomized product formula: it draws ``num_terms`` samples from the Hamiltonian's terms (or
    :attr:`~qiskit_fermions.operators.FermionOperator.groups`, if assigned), with each term sampled
    with a probability proportional to the magnitude of its sampling weight :math:`w_j`, and emits
    one :class:`.Evolution` gate per sample. Every sampled gate evolves its (unit-magnitude,
    sign-preserving) term (or, when the Hamiltonian carries groups, its whole sampled group) for
    the same time

    .. math::

        \delta = \frac{\lambda t}{\texttt{num\_terms}}, \qquad
        p_j = \frac{w_j}{\lambda}, \qquad
        \lambda = \sum_j w_j,

    so that each draw contributes :math:`\delta \cdot p_j = w_j t / \texttt{num\_terms}`. By default
    :math:`w_j = |c_j|`, the magnitude of the sampled term's (or group's) coefficient, which recovers
    the textbook qDRIFT normalization :math:`\lambda = \sum_j |c_j|`; the sign of :math:`c_j` is not
    part of the weight but of the sampled operator, and is read off it directly. Supply
    :attr:`weights` to precompute that array once instead of deriving it on every call.

    The ordered product of the sampled evolutions does not reproduce :math:`e^{-i t H}` exactly;
    rather, its expectation over the sampling approximates the exact evolution, with an error that
    decreases as ``num_terms`` grows. Because the output depends on the random draws, it differs
    from run to run unless a fixed ``rng`` is supplied.

    .. note::
       The sampled gates are marked :attr:`.Evolution.atomic`: the random draw *is* the
       Trotterization this pass performs, so the emitted gates are terminal factors and
       :meth:`~.FermionicCircuit.decompose` leaves them in place rather than splitting them further.
       Each of them carries over the :attr:`.Evolution.synthesis` method of the gate it was sampled
       from, even though an atomic gate never consults it.

    .. note::
       A fixed ``rng`` reproduces the *sequence* of randomizations, not an individual member of it.
       Successive :meth:`run` calls draw from the same generator, so they yield different circuits;
       replaying from the same seed reproduces all of them in the same order, but the nth circuit
       cannot be obtained without drawing the preceding ones first. Generating a batch of
       randomizations and recording the seed therefore works as expected (see
       :ref:`sqdrift_getting_started`); addressing one member directly is not supported.

    .. warning::
       The optional ``filter_trivial`` mode exists to stop the sampling from spending one of the
       ``num_terms`` slots on an excitation that cannot move a particle, and which therefore tells
       you nothing about the sampled bitstrings. It buys that with the protocol's convergence
       guarantee, and it is off by default for that reason.

       It does not make the circuit shorter. ``num_terms`` is fixed, so a rejected draw is replaced
       rather than dropped, and the cheap term it would have contributed (a diagonal rotation, say)
       gives way to a coupling excitation that costs more to synthesize. Expect the filtered circuit
       to be deeper than the unfiltered one: the budget of sampled slots is what the filtering
       conserves, not the depth.

       Rejecting a drawn term and re-drawing renormalizes the sampling distribution over the
       accepted terms only, so the sampled product no longer averages to :math:`e^{-i t H}` for the
       Hamiltonian you passed in: the effective coefficient of every retained term is inflated by
       the reciprocal of the acceptance probability, and the rejected terms drop out. Neither
       :math:`\lambda` nor :math:`\delta` is adjusted to compensate, so the distortion does not
       cancel.

       Use it only when the sampled bitstrings are the quantity of interest, as in the SqDRIFT
       workflow of :ref:`sqdrift_getting_started`. Do not use it when the sampled circuits are used
       to estimate an expectation value, a time-evolved observable, or anything else that relies on
       the qDRIFT error bound: those results are biased by an amount the pass does not track. See
       :attr:`filter_trivial` for the acceptance rule and its prerequisites.

    .. hint::
       Terms that are diagonal in the occupation-number basis (that is, products of number operators)
       have no effect on the sampled bitstrings, so including them only increases the sampling
       overhead. Filter them out with
       :func:`~qiskit_fermions.operators.terms.filter_diagonal_terms` on the Hamiltonian
       *before* constructing the :class:`.Evolution` gate, rather than on every call to :meth:`run`:
       this pass runs once per transpiled circuit, so filtering upstream avoids repeating the same
       filtering work for every circuit generated from the same Hamiltonian.

    .. caution::
       The *scale* of :attr:`weights` is not free: it sets the evolution time. Since
       :math:`\delta \cdot p_j = w_j t / \texttt{num\_terms}` above depends on :math:`w_j` itself and
       not merely on its share of :math:`\lambda`, rescaling every weight by :math:`\gamma` leaves the
       distribution untouched but evolves for :math:`\gamma t` rather than :math:`t`. Weights are
       therefore absolute magnitudes, not relative preferences, and only :math:`w_j = |c_j|`
       reproduces :math:`e^{-i t H}`.

       A distribution whose *shape* differs from :math:`|c_j|` no longer approximates
       :math:`e^{-i t H}` on its own either. Recovering the target evolution then requires
       reweighting the measured outcomes, which is the caller's responsibility: this pass emits
       circuits and cannot post-process their results. See :attr:`weights`.

    .. seealso::
       The qDRIFT protocol was introduced in `arXiv:1811.08017 <https://arxiv.org/abs/1811.08017>`_.

    .. rubric:: Filtering diagnostics

    When ``filter_trivial`` actually filters a gate, the returned :class:`.FermionicDAGCircuit`
    records how many draws it discarded in its :attr:`~qiskit.dagcircuit.DAGCircuit.metadata`, under
    ``filter_trivial.discarded`` and ``filter_trivial.emitted``. Both hold one entry per filtered
    :class:`.Evolution` gate, in circuit order. Their ratio estimates the acceptance probability,
    and hence the factor by which the filtering inflated the coefficients of the terms it kept: an
    acceptance probability close to one means the filtering barely moved the distribution, while a
    small one means the retained terms were weighted far above their true share of the Hamiltonian.

    .. important::
       Neither field is present when no gate was filtered, which covers ``filter_trivial=False`` and
       every case in which filtering was skipped with a :class:`UserWarning`. Read them defensively,
       for example with ``qcirc.metadata.get("filter_trivial.discarded")``. A discarded count of
       zero is different from an absent field: it says the filtering ran on that gate and accepted
       every draw, so it left the sampling distribution untouched.
    """

    MAX_SAMPLE_RETRIES = 1_000_000
    """The maximum number of consecutive rejected samples tolerated by ``filter_trivial`` before
    :meth:`run` gives up and raises :class:`RuntimeError`. This guards against an infinite loop when
    the Hamiltonian's remaining terms cannot bridge the tracked occupied/unoccupied mode sets. That
    happens when both sets remain small and disjoint (few modes have been marked occupied or
    unoccupied, and none has yet become "uncertain") and no remaining term's support touches
    both."""

    def __init__(
        self,
        num_terms: int,
        *,
        filter_trivial: bool = False,
        weights: Sequence[float] | np.ndarray | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initializing this transpiler pass can be done with the arguments listed below.

        Args:
            num_terms: the number of terms to sample for the qDRIFT Trotterization. This equals the
                number of :class:`.Evolution` gates emitted per input gate; a larger value reduces
                the Trotterization error at the cost of a deeper circuit.
            filter_trivial: when set to ``True``, the sampling loop rejects a drawn term unless its
                support couples a mode tracked as occupied with a mode tracked as unoccupied, and
                draws a replacement in its place. This spends every one of the ``num_terms``
                slots on an excitation that can move a particle, at the cost of biasing the
                Trotterization and of a deeper circuit, so it defaults to ``False``. See also
                :attr:`filter_trivial` for the acceptance rule and its prerequisites, and the
                warning in the class docstring for the bias it introduces.
            weights: the sampling weights to use instead of the coefficient magnitudes derived from
                the Hamiltonian. If ``None`` (the default), they are computed from the evolved
                operator on every call, which reproduces the textbook qDRIFT distribution. See
                :attr:`weights` for the expected length, the sign convention and the effect on the
                evolution time.
            rng: the random number generator (rng) to be used. When this is an ``int``, the internal
                rng will be initialized with ``np.random.default_rng(seed=rng)``.

        Raises:
            ValueError: if ``weights`` is not one-dimensional, is empty, holds a non-finite or
                negative entry, or sums to zero.
        """
        sampling_weights: np.ndarray | None = None
        if weights is not None:
            sampling_weights = np.asarray(weights, dtype=float)
            if sampling_weights.ndim != 1:
                raise ValueError(
                    f"The sampling weights must be a one-dimensional array, but got one with "
                    f"{sampling_weights.ndim} dimensions."
                )
            if sampling_weights.size == 0:
                raise ValueError("The sampling weights must not be empty, but got an empty array.")
            if not np.all(np.isfinite(sampling_weights)):
                raise ValueError(
                    "The sampling weights must all be finite, but got an array containing NaN or "
                    "an infinity."
                )
            negative = sampling_weights < 0.0
            if negative.any():
                raise ValueError(
                    f"Negative sampling weights are not supported yet: sampling from a signed "
                    f"(quasi-probability) distribution additionally requires the accumulated sign "
                    f"of the sampled entries to be reported for the post-processing of the measured "
                    f"outcomes, which this pass does not do. Pass the magnitudes instead, but got "
                    f"{int(negative.sum())} negative entries."
                )
            if sampling_weights.sum() == 0.0:
                raise ValueError(
                    "The sampling weights must not sum to zero, since that sum normalizes the "
                    "sampling distribution and scales the evolution time, but got an array whose "
                    "entries are all zero."
                )

        super().__init__()

        self.num_terms = num_terms
        """The number of terms to include in the qDRIFT Trotterization."""

        self.filter_trivial = filter_trivial
        """Whether to reject drawn terms that do not couple the tracked occupied/unoccupied modes.

        When this is ``True``, the sampling loop accepts a drawn term only if its support intersects
        both the set of modes tracked as occupied and the set tracked as unoccupied, and it draws a
        replacement for every term it rejects, so that none of the :attr:`num_terms` slots is spent
        on a term that leaves the occupation unchanged and therefore says nothing about the sampled
        bitstrings. Rejection renormalizes the sampling distribution over the accepted terms, which
        biases the Trotterization, and the replacement it draws is more expensive to synthesize than
        the term it displaced: see the warning in the class docstring before enabling this.

        Filtering requires an :class:`.InitializeModes` or :class:`.PrepareSlaterDeterminant` gate
        to precede the :class:`.Evolution` gates being Trotterized, to seed the initial occupied and
        unoccupied mode sets. If none is found, or if the sets it seeds turn out to be entirely
        occupied or entirely unoccupied, filtering is skipped for that gate and a
        :class:`UserWarning` is emitted instead.

        Any :class:`.OrbitalRotation` gate encountered before or between the :class:`.Evolution`
        gates also updates these sets: every mode it acts on becomes "uncertain", since the rotation
        can mix it with any other mode it touches, just like a mode touched by an accepted term. A
        :class:`.PrepareSlaterDeterminant` gate updates the sets the same way its
        :class:`.InitializeModes` and :class:`.OrbitalRotation` components would if applied in
        sequence: it seeds the occupied and unoccupied sets from its ``occupation``, then
        immediately marks every mode it acts on as "uncertain" because of its rotation. See the
        :meth:`run` docstring for the precise acceptance rule."""

        self.weights: np.ndarray | None = sampling_weights
        r"""The sampling weights :math:`w_j`, or ``None`` to derive them from the evolved operator.

        Supplying them hoists their computation out of the transpilation: the default path recomputes
        them on *every* call to :meth:`run`, which is repeated work when generating an ensemble from a
        single Hamiltonian. Deriving them once with
        :func:`~qiskit_fermions.operators.terms.group_coeff_means` and passing the result here keeps
        this pass stateless while paying that cost a single time. That is the reason to reach for this
        argument; passing anything other than the Hamiltonian's own coefficient magnitudes changes
        which evolution the ensemble approximates (see the caution below).

        One entry is expected per *group* when the evolved operator carries
        :attr:`~qiskit_fermions.operators.OperatorTrait.groups`, and per *term* otherwise; a
        mismatch raises :class:`ValueError`. Because such an array describes one specific operator,
        a circuit holding more than one :class:`.Evolution` gate is rejected as well: leave this
        unset for such a circuit, so that every gate derives its own weights, or transpile one gate
        at a time.

        The granularity follows what the pass samples, which is why it is the grouping that decides
        it: a grouped operator is sampled group-wise, so a weight describes a whole group. Whether
        the grouping is the appropriate unit for a given operator is a property of that operator, not
        of this argument -- see :ref:`grouping_explanation`, and
        :func:`~qiskit_fermions.operators.terms.groups_are_hermitian` to check the most
        common convention.

        Entries must be non-negative. A weight is the magnitude :math:`h_j` of the qDRIFT
        decomposition :math:`H = \sum_j h_j H_j`, in which a coefficient's sign belongs to
        :math:`H_j` rather than to :math:`h_j` and is read off the operator directly, so a sign here
        would have nothing to describe. Sampling from a signed (quasi-probability) distribution is a
        separate feature: it additionally requires the accumulated sign of the sampled entries to be
        reported back for post-processing, which this pass does not do.

        .. caution::
           These are absolute magnitudes, not relative preferences: their sum also sets the evolution
           time, so rescaling every entry by :math:`\gamma` evolves for :math:`\gamma t` while
           sampling identically. See the class docstring.
        """

        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def run(self, dag: FermionicDAGCircuit) -> FermionicDAGCircuit:
        """Runs this transpilation pass.

        Each :class:`.Evolution` node is replaced by ``num_terms`` sampled :class:`.Evolution`
        gates, one per drawn term (or per drawn group, when the Hamiltonian carries groups; see the
        class docstring). The emitted gates are marked :attr:`.Evolution.atomic` and carry over the
        :attr:`.Evolution.synthesis` method of the node they replace. Nodes that are not
        :class:`.Evolution` gates are copied to the output unchanged. Since the sampling is random,
        the output varies between runs unless the ``rng`` was seeded.

        The sampling weights are recomputed from each evolved operator here, unless :attr:`weights`
        was supplied, in which case that array is used as-is and this method never touches the
        operator's coefficients. A supplied array is validated against the operator it is applied to,
        and restricts the circuit to a single :class:`.Evolution` gate (see :attr:`weights`).

        When :attr:`filter_trivial` is set, this method tracks the sets of modes that are known to
        be occupied or unoccupied, seeded from any :class:`.InitializeModes` gate(s) preceding the
        :class:`.Evolution` gates in the circuit (several such gates placed in parallel, for example
        one per spin sector, are accumulated together). A drawn term is accepted only if its support
        intersects *both* sets, that is, it couples a known-occupied mode with a known-unoccupied
        one; otherwise it is discarded and a replacement is drawn. Rejection renormalizes the
        sampling distribution over the accepted terms, which biases the Trotterization: see the
        warning in the class docstring. Once a term is accepted, every mode in its support becomes
        "uncertain" and is added to *both* sets, making it eligible to participate in either role
        for subsequent samples. Any :class:`.OrbitalRotation` gate found in the circuit updates
        these sets the same way: every mode it acts on becomes "uncertain" too, since the rotation
        can mix it with any other mode in its support. A :class:`.PrepareSlaterDeterminant` gate is
        treated as its :class:`.InitializeModes` and :class:`.OrbitalRotation` components applied
        back-to-back: its ``occupation`` first seeds the occupied/unoccupied sets, and then every
        mode it acts on is immediately marked "uncertain", since it also carries a rotation.

        Args:
            dag: the input circuit with fermion-based instructions. Only
                :class:`~qiskit.dagcircuit.DAGOpNode` with :class:`.FermionicGate` instances as their
                :attr:`~qiskit.dagcircuit.DAGOpNode.op` are supported.

        Returns:
            The output circuit which is still acting on a fermionic register. When filtering
            actually ran, its :attr:`~qiskit.dagcircuit.DAGCircuit.metadata` also carries the
            ``filter_trivial.discarded`` and ``filter_trivial.emitted`` counts described in the
            class docstring.

        Raises:
            RuntimeError: if ``filter_trivial`` is ``True`` and :attr:`MAX_SAMPLE_RETRIES`
                consecutive draws are rejected without any of them coupling the tracked occupied and
                unoccupied mode sets.
            ValueError: if :attr:`weights` was supplied and its length does not match the number of
                groups (or terms) of an evolved operator, or if the circuit holds more than one
                :class:`.Evolution` gate.
        """
        out_dag = dag.copy_empty_like()

        # Counts the `Evolution` gates Trotterized by this call, to reject a circuit holding several
        # of them when `weights` was supplied (see below). Deliberately a local rather than an
        # instance attribute: the pass must stay stateless across `run` calls.
        num_evolutions = 0

        # The sets of modes that are currently known to be occupied/unoccupied, respectively. Seeded
        # by any preceding `InitializeModes` gate(s); both remain empty until the first one is
        # encountered, which is how we distinguish "no InitializeModes seen yet" from "seeded, but
        # every mode landed in the same set" (see the two warnings below). `filter_trivial` uses these
        # sets to reject sampled terms that cannot change the occupation. Once a term is accepted, the
        # modes it touches move into "uncertain" (see below), so they end up in *both* sets going
        # forward.
        occupied: set[int] = set()
        unoccupied: set[int] = set()

        # Per-gate rejection statistics, recorded into the output DAG's metadata below. One entry is
        # appended per `Evolution` gate that was actually filtered, so a gate whose filtering was
        # skipped (see the warnings below) contributes nothing and both lists stay empty when
        # `filter_trivial` is not in effect anywhere.
        discarded_per_gate: list[int] = []
        emitted_per_gate: list[int] = []

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

            num_evolutions += 1
            if self.weights is not None and num_evolutions > 1:
                raise ValueError(
                    "A custom sampling weights array describes the terms (or groups) of one "
                    "specific operator, so it cannot be applied to a circuit holding more than one "
                    "Evolution gate. Either leave weights unset, so that each gate derives its own, "
                    "or transpile one Evolution gate at a time."
                )

            has_groups = hamil.has_groups()

            if self.weights is not None:
                # One weight per group when the operator is grouped, else one per term. `len()`
                # is free, while `num_groups()` walks the group indices natively -- a cost only
                # paid when weights were supplied, and small next to the `split_out_groups` lookups
                # that follow, but the price of catching a wrong-length array here rather than
                # sampling against the wrong operator.
                expected = hamil.num_groups() if has_groups else len(hamil)
                if len(self.weights) != expected:
                    granularity = "groups" if has_groups else "terms"
                    raise ValueError(
                        f"The sampling weights must hold exactly one entry per sampled piece, that "
                        f"is {expected} entries for an operator with {expected} {granularity}, but "
                        f"got {len(self.weights)}."
                    )

            # `terms` is a list of operator terms, or None when `hamil.has_groups()` is False
            terms: list[Any] | None
            if not has_groups:
                # NOTE: the qDRIFT protocol normalizes each term to unit magnitude because the
                # evolution time is entirely dictated by `delta` (computed below). Only the
                # magnitude of a coefficient sets its sampling probability, but its sign fixes the
                # direction of the rotation and must be preserved for the Trotterization to
                # approximate the target time evolution.
                terms = [(actions, np.sign(coeff)) for actions, coeff in hamil.iter_terms()]
                # NOTE: skipped entirely when the caller supplied the weights, which is what makes
                # hoisting this out of a large ensemble loop worthwhile: `get_coeffs()` copies one
                # value per term out of the operator on every call.
                if self.weights is None:
                    weights = np.abs(hamil.get_coeffs())
            else:
                if self.weights is None:
                    # NOTE: computed natively rather than by reducing `hamil.get_coeffs()` and
                    # `hamil.groups` here. Those two accessors each copy one value per *ungrouped*
                    # term out of the operator, only for both arrays to be aggregated straight back
                    # down to one weight per group -- which dominates the cost of the reduction
                    # itself for a Hamiltonian holding far more terms than groups. The mean is the
                    # magnitude of one atomic group, which is the scale the protocol needs: grouping
                    # is what makes each sampled piece Hermitian, and hence its evolution unitary,
                    # to begin with.
                    weights = np.array(group_coeff_means(hamil))
                # NOTE: we do not materialize the group operators here. Since only a small
                # fraction of the (potentially much larger) set of groups ends up being sampled,
                # we look up each sampled group's operator lazily via `split_out_groups`, once we
                # know which indices were actually drawn.
                terms = None

            if self.weights is not None:
                weights = self.weights

            def _unit_terms(term):
                if isinstance(term, tuple):
                    # in this case, we have already normalized the coefficient to its sign
                    return [term]

                # NOTE: as per the comment earlier, we have not yet normalized the coefficients
                # of grouped operator terms. Keep each term's sign (dropping only its magnitude)
                # so that the sampled rotation points in the correct direction.
                return [(actions, np.sign(coeff)) for actions, coeff in term.iter_terms()]

            # NOTE: the weights are non-negative, so this sum is the protocol's `lambda`. It both
            # normalizes the sampling distribution and sets the shared evolution time, which is why
            # a supplied array's scale is not free (see the class docstring).
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
                if terms is None:
                    # Only look up the (typically much smaller) set of distinct sampled groups,
                    # rather than materializing every group in the Hamiltonian, then map each
                    # draw back to its (possibly repeated) fetched operator.
                    unique_indices, inverse = np.unique(sampled_indices, return_inverse=True)
                    sampled_terms = hamil.split_out_groups(group_indices=unique_indices.tolist())
                    draws = [sampled_terms[i] for i in inverse]
                else:
                    draws = [terms[sampled_idx] for sampled_idx in sampled_indices]
                for term in draws:
                    unit_terms = _unit_terms(term)
                    op = hamil.__class__.from_terms(unit_terms)
                    evo = Evolution(
                        num_modes,
                        op,
                        time=delta,
                        synthesis=node.op.synthesis,
                        atomic=True,
                    )
                    out_dag.apply_operation_back(evo, qargs=out_dag.qubits)
                continue

            # Precomputed once here (only reached once rejection sampling is actually needed) so
            # the loop below can draw against it directly via `searchsorted` instead of calling
            # `rng.choice(..., p=probabilities)`, which would rebuild this same cumulative sum from
            # scratch on every single draw.
            cdf = np.cumsum(probabilities)
            cdf /= cdf[-1]  # guards against float-sum drift from 1.0, matching numpy's own choice()

            added_terms = 0
            # `failed_attempts` is reset on every acceptance because it guards against an infinite
            # loop, so it counts *consecutive* rejections only. `discarded` is the running total for
            # this gate, which is what the metadata reports.
            failed_attempts = 0
            discarded = 0
            while added_terms < self.num_terms:
                # Equivalent to `self._rng.choice(np.arange(len(weights)), p=probabilities)`:
                # this is numpy's own implementation of weighted sampling (see `Generator.choice`),
                # just reusing the `cdf` precomputed above instead of rebuilding it on every draw.
                sampled_idx = cdf.searchsorted(self._rng.random(), side="right")
                if terms is None:
                    term = hamil.split_out_groups(group_indices=[int(sampled_idx)])[0]
                else:
                    term = terms[sampled_idx]
                unit_terms = _unit_terms(term)
                op = hamil.__class__.from_terms(unit_terms)

                term_support = op.get_support()
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

                evo = Evolution(
                    num_modes,
                    op,
                    time=delta,
                    synthesis=node.op.synthesis,
                    atomic=True,
                )
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
