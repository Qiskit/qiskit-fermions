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

"""The higher-order Suzuki-Trotter fermionic product formula."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .evolution_synthesis import FermionicEvolutionSynthesis

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit

    from ..evolution import Evolution


def _recurse(order: int, blocks: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Recursively builds the symmetric Suzuki decomposition of one sweep over ``blocks``.

    Args:
        order: the order of the formula. Assumed to be ``1`` or even, as validated by
            :class:`FermionicSuzukiTrotter`.
        blocks: one ``(index, time_scale)`` pair per factor of a single sweep.

    Returns:
        The ordered factors of the decomposition, as ``(index, time_scale)`` pairs. Indices repeat.
    """
    if order == 1:
        return list(blocks)

    if order == 2:
        # the symmetric sandwich: halve every factor but the last, which becomes the middle
        halves = [(index, scale / 2) for index, scale in blocks[:-1]]
        return [*halves, blocks[-1], *reversed(halves)]

    # Suzuki's recursion: five order-(k-2) sweeps, three at `reduction` and one at `1 - 4 * reduction`
    reduction = 1 / (4 - 4 ** (1 / (order - 1)))
    outer = _recurse(order - 2, [(index, scale * reduction) for index, scale in blocks])
    inner = _recurse(order - 2, [(index, scale * (1 - 4 * reduction)) for index, scale in blocks])
    # NOTE: `2 * outer` repeats the *list*, it does not scale the coefficients
    return 2 * outer + inner + 2 * outer


def _suzuki_schedule(order: int, num_factors: int, reps: int) -> list[tuple[int, float]]:
    """Builds the full evolution schedule of a Suzuki-Trotter formula.

    Args:
        order: the order of the formula.
        num_factors: the number of factors the evolved operator is split into.
        reps: the number of times to repeat the formula.

    Returns:
        The ordered ``(factor_index, time_scale)`` pairs to emit. Each scale is a fraction of the
        gate's total evolution time; the scales of any one factor always sum to ``1``.
    """
    if num_factors == 1:
        # A single factor commutes with itself, so the formula is exact at any order and the
        # decomposition collapses to one gate carrying the whole evolution time. Recursing here would
        # emit up to 5**((order - 2) / 2) gates that merely re-split the same operator.
        return [(0, 1.0)]

    sweep = _recurse(order, [(index, 1.0) for index in range(num_factors)])
    return reps * [(index, scale / reps) for index, scale in sweep]


class FermionicSuzukiTrotter(FermionicEvolutionSynthesis):
    r"""The higher-order Suzuki-Trotter product formula, applied in fermionic space.

    A product formula composes the factors of an operator symmetrically, so that the leading error
    terms cancel. For :math:`\texttt{order} = 2` and factors :math:`H_j`, the approximation is the
    palindrome

    .. math::

        e^{-i t \sum_j H_j} \approx
        \left( \prod_{j<n} e^{-i \frac{t}{2} H_j} \right) e^{-i t H_n}
        \left( \prod_{j<n} e^{-i \frac{t}{2} H_j} \right)^{\mathrm{R}},

    where :math:`\mathrm{R}` denotes the reversed order. Higher even orders are built from this by
    Suzuki's recursion, each level composing five sweeps of the level below.

    The evolved operator is split by its :attr:`~qiskit_fermions.operators.OperatorTrait.groups` when
    it has them (see :ref:`grouping_explanation`), and term-by-term otherwise. Every order splits it
    the same way: what a higher order changes is the ordering of the factors and their time scales, not
    the partition.

    .. note::
       A higher order buys accuracy with depth: an order-:math:`k` formula emits roughly
       :math:`5^{(k-2)/2}` times as many factors as an order-2 one. Increasing :attr:`reps` instead
       divides the evolution time into more, shorter steps, which reduces the error at a different
       rate for the same kind of cost. Which is the better trade depends on the operator.

    .. note::
       Applying this in fermionic space at all is optional, and it composes with the product formula
       chosen for the fermion-to-qubit stage. Where both approximate, the accuracy of the result is
       governed by the weaker of the two -- so raising the order here while the qubit-side formula
       stays first-order buys little. See :mod:`~qiskit_fermions.circuit.library.synthesis`.

    .. caution::
       Each factor must be Hermitian for its exponential to be unitary, which does not follow from
       their sum being Hermitian: splitting a Hermitian operator can produce non-Hermitian groups (for
       example, separating :math:`a^\dagger_0 a_1` from its conjugate partner :math:`a^\dagger_1 a_0`).
       It is the caller's responsibility to group accordingly; this is not verified.

       Note that the symmetrization partially cancels the error of a non-Hermitian factor, so an
       incorrectly grouped operator can look markedly better at an even order than at first order while
       still being wrong.
    """

    def __init__(self, order: int = 2, reps: int = 1) -> None:
        """Initializing an instance of this synthesis method can be done with the arguments below.

        Args:
            order: the order of the product formula. Must be ``1`` (which reduces to
                :class:`.FermionicLieTrotter`) or even, since the Suzuki formulas are symmetric.
            reps: the number of times to repeat the formula, each repetition evolving for
                ``time / reps``.

        Raises:
            ValueError: if ``order`` is neither ``1`` nor even, or if ``reps`` is not positive.
        """
        if order != 1 and order % 2 != 0:
            raise ValueError(
                "The Suzuki product formulas are symmetric and therefore only defined for an order "
                f"of 1 or an even order, but got {order}."
            )
        if reps < 1:
            raise ValueError(f"The number of repetitions must be positive, but got {reps}.")

        self.order = order
        """The order of the product formula."""

        self.reps = reps
        """The number of times the product formula is repeated."""

    def synthesize(self, gate: Evolution) -> FermionicCircuit:
        """Synthesizes the gate into the ordered factors of the product formula.

        See the class documentation for the formula this implements.

        Args:
            gate: the gate to synthesize.

        Returns:
            A :class:`.FermionicCircuit` holding one narrowed :class:`.Evolution` gate per factor of
            the formula. A factor may appear more than once, at different evolution times.
        """
        # deferred import: `qiskit_fermions.circuit` imports this subpackage's siblings at module load
        from qiskit_fermions.circuit import FermionicCircuit

        from ..evolution import Evolution

        definition = FermionicCircuit(gate.num_modes)

        # keep the factors around rather than re-splitting per factor: the formula revisits each of
        # them, so building them once is cheaper
        if gate.operator.has_groups():
            factors = gate.operator.split_out_groups()
        else:
            factors = [
                gate.operator.__class__.from_terms([term]) for term in gate.operator.iter_terms()
            ]

        time = gate.params[0]
        for index, scale in _suzuki_schedule(self.order, len(factors), self.reps):
            factor = factors[index]

            # reduce each operator to act only on the non-idle part of the register
            active = factor.get_support()
            num_active = len(active)
            active_idx = iter(range(num_active))
            idle_idx = iter(range(num_active, gate.num_modes))
            permutation = [
                next(active_idx) if idx in active else next(idle_idx)
                for idx in range(gate.num_modes)
            ]
            relabeled = factor.relabel_modes(permutation)

            # the time carries the formula's scale; scaling the operator instead would drop its
            # groups and make the emitted gate's label misreport the physics
            definition.append(
                Evolution(num_active, relabeled, time=time * scale, atomic=True),
                [definition.modes[idx] for idx in sorted(active)],
            )

        return definition
