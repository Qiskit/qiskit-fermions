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

"""The fermionic synthesis interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit

    from ..evolution import Evolution


class FermionicEvolutionSynthesis(ABC):
    r"""The interface for `fermion-to-fermion` synthesis of an :class:`.Evolution` gate.

    An implementation of this interface rewrites the time evolution :math:`e^{-i t H}` carried by an
    :class:`.Evolution` gate into a :class:`.FermionicCircuit` of *smaller* :class:`.FermionicGate`
    instances. Both sides of that rewrite live in `fermionic` space, which is what distinguishes this
    from the fermion-to-qubit synthesis performed later by :class:`.F2QSynthesis`: the fermionic
    structure of ``H`` (in particular its :attr:`~qiskit_fermions.operators.OperatorTrait.groups`
    (see :ref:`grouping_explanation`)) survives and is still available to that later stage.

    .. important::
       This fermion-to-fermion step is **optional**. An :class:`.Evolution` gate does not need to be
       broken down in fermionic space at all: the fermion-to-qubit stage can map and synthesize it
       directly, however many terms its operator holds. Splitting it up first is a *choice*, made
       because the resulting factors are individually cheaper to implement, or because the split
       exposes structure (such as mutually commuting groups) that the later stage can exploit.

    Because the fermion-to-fermion and fermion-to-qubit steps compose, an evolution may end up
    approximated at either level, at both, or at neither. Where an approximation is involved at both,
    the accuracy of the result is governed by the weaker of the two.

    .. note::
       Qiskit's :class:`~qiskit.synthesis.EvolutionSynthesis` is the closest analogue, but it belongs
       to the qubit layer: it synthesizes a :class:`~qiskit.circuit.library.PauliEvolutionGate` into a
       :class:`~qiskit.circuit.QuantumCircuit`, after the operator has already been mapped.
    """

    @abstractmethod
    def synthesize(self, gate: Evolution) -> FermionicCircuit:
        r"""Synthesizes the provided :class:`.Evolution` gate.

        Args:
            gate: the gate to synthesize. Its
                :attr:`~qiskit_fermions.circuit.library.Evolution.operator` is the Hermitian operator
                :math:`H` and its ``params[0]`` the evolution time :math:`t` of :math:`e^{-i t H}`.

        Returns:
            A :class:`.FermionicCircuit` on ``gate.num_modes`` modes implementing the evolution.
        """
