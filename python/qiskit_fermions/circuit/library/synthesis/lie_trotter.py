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

"""The first-order Lie-Trotter fermionic product formula."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .evolution_synthesis import FermionicEvolutionSynthesis

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit

    from ..evolution import Evolution


class FermionicLieTrotter(FermionicEvolutionSynthesis):
    r"""The first-order Lie-Trotter product formula, applied in fermionic space.

    Given an :class:`.Evolution` gate with the evolution time :math:`t`, this synthesizes the
    approximation

    .. math::

        e^{-i t \sum_j H_j} \approx \prod_j e^{-i t H_j},

    emitting one :class:`.Evolution` gate per factor :math:`H_j`, each evolving for the `full` time
    :math:`t`.

    How the evolved operator is split into the :math:`H_j` depends on the operator itself: when it has
    :attr:`~qiskit_fermions.operators.OperatorTrait.groups` assigned (see
    :ref:`grouping_explanation`), each group becomes one factor; otherwise every individual term does.
    Each factor is reduced to act only on the modes it actually touches, so the emitted gate spans as
    few modes as possible.

    .. note::
       This product formula is `exact` only when the factors mutually commute. Grouping an operator so
       that each group collects mutually commuting terms therefore both shortens the circuit and
       reduces the Trotter error.

       Note also that applying it at all is optional: leaving the gate undecomposed and letting the
       fermion-to-qubit stage handle the whole operator incurs no Trotter error at this level. See
       :mod:`~qiskit_fermions.circuit.library.synthesis`.

    .. caution::
       Each factor :math:`H_j` must itself be Hermitian for :math:`e^{-i t H_j}` to be unitary. This
       does `not` follow from their sum being Hermitian: splitting a Hermitian operator can produce
       non-Hermitian groups (for example, separating :math:`a^\dagger_0 a_1` from its conjugate
       partner :math:`a^\dagger_1 a_0`). It is the caller's responsibility to group accordingly; this
       is not verified.
    """

    def synthesize(self, gate: Evolution) -> FermionicCircuit:
        """Synthesizes the gate into one :class:`.Evolution` factor per group (or per term).

        See the class documentation for the product formula this implements.

        Args:
            gate: the gate to synthesize.

        Returns:
            A :class:`.FermionicCircuit` holding one narrowed :class:`.Evolution` gate per factor.
        """
        # deferred import: `qiskit_fermions.circuit` imports this subpackage's siblings at module load
        from qiskit_fermions.circuit import FermionicCircuit

        from ..evolution import Evolution

        definition = FermionicCircuit(gate.num_modes)

        # when the operator being evolved has groups use those for the decomposition, otherwise
        # decompose into all individual terms
        iterator = (
            gate.operator.split_out_groups
            if gate.operator.has_groups()
            else gate.operator.iter_terms
        )

        for item in iterator():
            if isinstance(item, tuple):
                # iterating over terms rather than operator groups
                item = gate.operator.__class__.from_terms([item])

            # reduce each operator to act only on the non-idle part of the register
            active = item.get_support()
            num_active = len(active)
            active_idx = iter(range(num_active))
            idle_idx = iter(range(num_active, gate.num_modes))
            permutation = [
                next(active_idx) if idx in active else next(idle_idx)
                for idx in range(gate.num_modes)
            ]
            relabeled = item.relabel_modes(permutation)

            # the factors are marked atomic: splitting them again would discard the very partition
            # this formula chose, and for a single-term factor there would be nothing left to split
            # anyway -- it would just re-emit itself indefinitely
            definition.append(
                Evolution(num_active, relabeled, time=gate.params[0], atomic=True),
                [definition.modes[idx] for idx in sorted(active)],
            )

        return definition
