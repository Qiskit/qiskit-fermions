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

# ruff: noqa: D205,D212,D415
"""
===================
Fermionic Synthesis
===================

.. currentmodule:: qiskit_fermions.circuit.library.synthesis

This module provides the synthesis methods with which an :class:`.Evolution` gate is broken down into
a :class:`.FermionicCircuit` of smaller fermionic gates. Pass one to the gate's ``synthesis`` argument
to choose how its time evolution gets decomposed:

.. code-block:: python

   Evolution(num_modes, operator, time=1.0, synthesis=FermionicLieTrotter())

Both sides of this rewrite live in `fermionic` space, so these methods can exploit the
:attr:`~qiskit_fermions.operators.OperatorTrait.groups` of the evolved operator (see
:ref:`grouping_explanation`) -- unlike the fermion-to-qubit synthesis that follows, which only sees the
mapped operator, where the grouping is no longer available.

.. important::
   This fermion-to-fermion step is **optional**. An :class:`.Evolution` gate can be handed straight to
   the fermion-to-qubit stage regardless of how many terms its operator holds; decomposing it in
   fermionic space first is a choice, taken to obtain cheaper factors or to expose structure that the
   later stage can exploit.

.. autosummary::
   :toctree: ../stubs/

   FermionicEvolutionSynthesis
   FermionicLieTrotter
   FermionicSuzukiTrotter
"""

from .evolution_synthesis import FermionicEvolutionSynthesis
from .lie_trotter import FermionicLieTrotter
from .suzuki_trotter import FermionicSuzukiTrotter

__all__ = [
    "FermionicEvolutionSynthesis",
    "FermionicLieTrotter",
    "FermionicSuzukiTrotter",
]
