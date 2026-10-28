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

"""The qDRIFT synthesis of an evolution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from qiskit_fermions.operators.terms.grouping import group_coeff_means

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit

    from ..evolution import Evolution


class QDriftSampler:
    """The draws of the qDRIFT protocol for one :class:`.Evolution` gate.

    This is what :meth:`.FermionicQDrift.sampler` returns. It holds the distribution derived from
    the gate, and exists so that a caller that needs to look at each draw (the
    :class:`.QDriftTrotterization` pass, when it rejects draws) can take them one at a time, whereas
    :meth:`.FermionicQDrift.synthesize` takes them all at once.
    """

    def __init__(
        self,
        gate: Evolution,
        num_terms: int,
        weights: np.ndarray | None,
        rng: np.random.Generator,
    ) -> None:
        """Initializing a sampler can be done with the arguments below.

        Args:
            gate: the gate to sample.
            num_terms: the number of gates the whole sample holds.
            weights: the sampling weights, or ``None`` to derive them from the operator of ``gate``.
            rng: the random number generator to draw from.

        Raises:
            ValueError: if ``weights`` does not hold one entry per group (or term) of the operator.
        """
        operator = gate.operator
        has_groups = operator.has_groups()

        if weights is not None:
            # One weight per group when the operator is grouped, else one per term. `len()` is free,
            # while `num_groups()` walks the group indices natively -- a cost only paid when weights
            # were supplied, and the price of catching a wrong-length array here rather than
            # sampling against the wrong operator.
            expected = operator.num_groups() if has_groups else len(operator)
            if len(weights) != expected:
                granularity = "groups" if has_groups else "terms"
                raise ValueError(
                    f"The sampling weights must hold exactly one entry per sampled piece, that "
                    f"is {expected} entries for an operator with {expected} {granularity}, but "
                    f"got {len(weights)}."
                )

        self._operator = operator
        self._synthesis = gate.synthesis
        self._num_modes = gate.num_modes
        self._rng = rng

        # `terms` is a list of operator terms, or None when the operator carries groups
        self._terms: list[Any] | None
        if not has_groups:
            # NOTE: the qDRIFT protocol normalizes each term to unit magnitude because the
            # evolution time is entirely dictated by `delta` (computed below). Only the magnitude of
            # a coefficient sets its sampling probability, but its sign fixes the direction of the
            # rotation and must be preserved for the Trotterization to approximate the target time
            # evolution.
            self._terms = [(actions, np.sign(coeff)) for actions, coeff in operator.iter_terms()]
            # NOTE: skipped entirely when the caller supplied the weights, which is what makes
            # hoisting this out of a large ensemble loop worthwhile: `get_coeffs()` copies one value
            # per term out of the operator on every call.
            if weights is None:
                weights = np.abs(operator.get_coeffs())
        else:
            if weights is None:
                # NOTE: computed natively rather than by reducing `get_coeffs()` and `groups`. Those
                # two accessors each copy one value per *ungrouped* term out of the operator, only
                # for both arrays to be aggregated straight back down to one weight per group. The
                # mean is the magnitude of one atomic group, which is the scale the protocol needs:
                # grouping is what makes each sampled piece Hermitian, and hence its evolution
                # unitary, to begin with.
                weights = np.array(group_coeff_means(operator))
            # NOTE: the groups are not materialized here. Only a small fraction of the (potentially
            # much larger) set of groups ends up being sampled, so each sampled group's operator is
            # looked up lazily via `split_out_groups`, once it is known which were drawn.
            self._terms = None

        # NOTE: the weights are non-negative, so this sum is the protocol's `lambda`. It both
        # normalizes the sampling distribution and sets the shared evolution time, which is why a
        # supplied array's scale is not free (see `FermionicQDrift`).
        lambd = np.sum(weights)
        self.time = (lambd * gate.params[0]) / num_terms
        """The evolution time :math:`\\delta` of every sampled gate."""

        self._num_terms = num_terms
        self._probabilities = weights / lambd
        self._cdf: np.ndarray | None = None

    def _gate(self, term: Any) -> Evolution:
        from ..evolution import Evolution

        if isinstance(term, tuple):
            # in this case, we have already normalized the coefficient to its sign
            unit_terms = [term]
        else:
            # NOTE: the coefficients of a grouped operator's terms have not been normalized yet. Keep
            # each term's sign (dropping only its magnitude) so that the sampled rotation points in
            # the correct direction.
            unit_terms = [(actions, np.sign(coeff)) for actions, coeff in term.iter_terms()]
        op = self._operator.__class__.from_terms(unit_terms)
        # the synthesis is carried over although an atomic gate never consults it
        return Evolution(
            self._num_modes, op, time=self.time, synthesis=self._synthesis, atomic=True
        )

    def sample(self, size: int | None = None) -> list[Evolution]:
        """Draws ``size`` gates in a single batched call.

        Args:
            size: the number of gates to draw. Defaults to ``num_terms``.

        Returns:
            The drawn atomic :class:`.Evolution` gates, in the order of the draws.
        """
        size = self._num_terms if size is None else size
        # a single batched call builds the cumulative-probability structure once, whereas each
        # scalar `choice()` call would rebuild it from scratch
        indices = self._rng.choice(
            np.arange(len(self._probabilities)), size=size, p=self._probabilities
        )
        if self._terms is None:
            # Only look up the (typically much smaller) set of distinct sampled groups, then map each
            # draw back to its (possibly repeated) fetched operator.
            unique_indices, inverse = np.unique(indices, return_inverse=True)
            sampled = self._operator.split_out_groups(group_indices=unique_indices.tolist())
            return [self._gate(sampled[i]) for i in inverse]
        return [self._gate(self._terms[i]) for i in indices]

    def draw(self) -> Evolution:
        """Draws a single gate.

        Returns:
            The drawn atomic :class:`.Evolution` gate.
        """
        if self._cdf is None:
            # Precomputed once, on first use, so that this can draw against it directly via
            # `searchsorted` instead of calling `rng.choice(..., p=probabilities)`, which would
            # rebuild this same cumulative sum from scratch on every single draw.
            self._cdf = np.cumsum(self._probabilities)
            # guards against float-sum drift from 1.0, matching numpy's own choice()
            self._cdf /= self._cdf[-1]
        # Equivalent to `rng.choice(np.arange(len(weights)), p=probabilities)`: numpy's own
        # implementation of weighted sampling, reusing the cached `cdf`.
        index = int(self._cdf.searchsorted(self._rng.random(), side="right"))
        if self._terms is None:
            return self._gate(self._operator.split_out_groups(group_indices=[index])[0])
        return self._gate(self._terms[index])


class FermionicQDrift:
    r"""A synthesis method that Trotterizes an :class:`.Evolution` via the qDRIFT protocol.

    This method replaces the exact evolution :math:`e^{-i t H}` of an :class:`.Evolution` gate by a
    randomized product formula: it draws ``num_terms`` samples from the Hamiltonian's terms (or
    :attr:`~qiskit_fermions.operators.FermionOperator.groups`, if assigned), with each term sampled
    with a probability proportional to the magnitude of its sampling weight :math:`w_j`, and emits
    one :class:`.Evolution` gate per sample. Every sampled gate evolves its (unit-magnitude,
    sign-preserving) term (or, when the Hamiltonian carries groups, its whole sampled group) for the
    same time

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

    The method is registered as ``Evolution.QDrift``, so it is on an equal footing with the product
    formulas :class:`.FermionicLieTrotter` and :class:`.FermionicSuzukiTrotter`:

    .. code-block:: python

       F2FSynthesis({"Evolution": ("QDrift", (100,), {"rng": 42})})

    A gate can also carry the method itself, as its :attr:`~.FermionicGate.synthesis`, and be broken
    down by :meth:`~.FermionicCircuit.decompose` (or Qiskit's ``Decompose`` pass). This is how each
    gate of a circuit gets its own ``num_terms``, ``weights`` and ``rng``:

    .. code-block:: python

       Evolution(num_modes, operator, time, synthesis=FermionicQDrift(100, weights=weights))

    .. note::
       The sampled gates are marked :attr:`.Evolution.atomic`: the random draw *is* the
       Trotterization this method performs, so the emitted gates are terminal factors and
       :meth:`~.FermionicCircuit.decompose` leaves them in place rather than splitting them further.

    .. note::
       A fixed ``rng`` reproduces the *sequence* of randomizations, not an individual member of it.
       Successive calls draw from the same generator, so they yield different circuits; replaying
       from the same seed reproduces all of them in the same order, but the nth circuit cannot be
       obtained without drawing the preceding ones first. Generating a batch of randomizations and
       recording the seed therefore works as expected (see :ref:`sqdrift_getting_started`);
       addressing one member directly is not supported.

    .. hint::
       Terms that are diagonal in the occupation-number basis (that is, products of number operators)
       have no effect on the sampled bitstrings, so including them only increases the sampling
       overhead. Filter them out with
       :func:`~qiskit_fermions.operators.terms.filter_diagonal_terms` on the Hamiltonian
       *before* constructing the :class:`.Evolution` gate, rather than on every call: this method
       runs once per transpiled circuit, so filtering upstream avoids repeating the same filtering
       work for every circuit generated from the same Hamiltonian.

    .. caution::
       The *scale* of :attr:`weights` is not free: it sets the evolution time. Since
       :math:`\delta \cdot p_j = w_j t / \texttt{num\_terms}` above depends on :math:`w_j` itself and
       not merely on its share of :math:`\lambda`, rescaling every weight by :math:`\gamma` leaves the
       distribution untouched but evolves for :math:`\gamma t` rather than :math:`t`. Weights are
       therefore absolute magnitudes, not relative preferences, and only :math:`w_j = |c_j|`
       reproduces :math:`e^{-i t H}`.

       A distribution whose *shape* differs from :math:`|c_j|` no longer approximates
       :math:`e^{-i t H}` on its own either. Recovering the target evolution then requires
       reweighting the measured outcomes, which is the caller's responsibility: this method emits
       circuits and cannot post-process their results. See :attr:`weights`.

    .. seealso::
       The qDRIFT protocol was introduced in `arXiv:1811.08017 <https://arxiv.org/abs/1811.08017>`_.
       To skip the draws that cannot move a particle, see :class:`.QDriftTrotterization`.
    """

    def __init__(
        self,
        num_terms: int,
        *,
        weights: Sequence[float] | np.ndarray | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """Initializing an instance of this synthesis method can be done with the arguments below.

        Args:
            num_terms: the number of terms to sample for the qDRIFT Trotterization. This equals the
                number of :class:`.Evolution` gates emitted per input gate; a larger value reduces
                the Trotterization error at the cost of a deeper circuit.
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
                    f"outcomes, which this method does not do. Pass the magnitudes instead, but got "
                    f"{int(negative.sum())} negative entries."
                )
            if sampling_weights.sum() == 0.0:
                raise ValueError(
                    "The sampling weights must not sum to zero, since that sum normalizes the "
                    "sampling distribution and scales the evolution time, but got an array whose "
                    "entries are all zero."
                )

        self.num_terms = num_terms
        """The number of terms to include in the qDRIFT Trotterization."""

        self.weights: np.ndarray | None = sampling_weights
        r"""The sampling weights :math:`w_j`, or ``None`` to derive them from the evolved operator.

        Supplying them hoists their computation out of the synthesis: the default path recomputes
        them on *every* call, which is repeated work when generating an ensemble from a single
        Hamiltonian. Deriving them once with
        :func:`~qiskit_fermions.operators.terms.group_coeff_means` and passing the result here keeps
        this method stateless while paying that cost a single time. That is the reason to reach for
        this argument; passing anything other than the Hamiltonian's own coefficient magnitudes
        changes which evolution the ensemble approximates (see the caution in the class docstring).

        One entry is expected per *group* when the evolved operator carries
        :attr:`~qiskit_fermions.operators.OperatorTrait.groups`, and per *term* otherwise; a
        mismatch raises :class:`ValueError`. Because such an array describes one specific operator,
        it belongs to the gate of that operator: pass this method as the ``synthesis`` of that
        :class:`.Evolution` gate, so that every gate of a circuit can hold the weights of its own
        operator. This method is stateless and checks only the length against each gate it is given,
        so it cannot tell that the array was meant for another operator of the same size. That is why
        giving one method by gate name to :class:`.F2FSynthesis` suits a circuit with a single
        :class:`.Evolution` gate, or one that leaves the weights unset so that each gate derives its
        own.

        The granularity follows what is sampled, which is why it is the grouping that decides it: a
        grouped operator is sampled group-wise, so a weight describes a whole group. Whether the
        grouping is the appropriate unit for a given operator is a property of that operator, not of
        this argument -- see :ref:`grouping_explanation`, and
        :func:`~qiskit_fermions.operators.terms.groups_are_hermitian` to check the most common
        convention.

        Entries must be non-negative. A weight is the magnitude :math:`h_j` of the qDRIFT
        decomposition :math:`H = \sum_j h_j H_j`, in which a coefficient's sign belongs to
        :math:`H_j` rather than to :math:`h_j` and is read off the operator directly, so a sign here
        would have nothing to describe.
        """

        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def sampler(self, gate: Evolution) -> QDriftSampler:
        """Prepares the draws for a gate without taking any of them.

        Args:
            gate: the gate to sample.

        Returns:
            The sampler, from which gates can be drawn in a batch or one at a time.

        Raises:
            ValueError: if :attr:`weights` was supplied and its length does not match the number of
                groups (or terms) of the operator of ``gate``.
        """
        return QDriftSampler(gate, self.num_terms, self.weights, self._rng)

    def synthesize(self, gate: Evolution) -> FermionicCircuit:
        """Synthesizes the gate into ``num_terms`` sampled gates.

        Args:
            gate: the gate to synthesize.

        Returns:
            A :class:`.FermionicCircuit` holding one atomic :class:`.Evolution` gate per draw.

        Raises:
            ValueError: if :attr:`weights` was supplied and its length does not match the number of
                groups (or terms) of the operator of ``gate``.
        """
        # deferred import: `qiskit_fermions.circuit` imports this subpackage's siblings at module load
        from qiskit_fermions.circuit import FermionicCircuit

        definition = FermionicCircuit(gate.num_modes)
        for sampled in self.sampler(gate).sample():
            definition.append(sampled, definition.modes)
        return definition
