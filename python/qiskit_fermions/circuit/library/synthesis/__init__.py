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
========================
Fermionic Gate Synthesis
========================

.. currentmodule:: qiskit_fermions.circuit.library.synthesis

This module provides the synthesis methods with which a gate is broken down into a
:class:`.FermionicCircuit` of smaller fermionic gates, such as the Trotter formulas for an
:class:`.Evolution`. Pass one to the gate's ``synthesis`` argument to choose how it gets decomposed:

.. code-block:: python

   Evolution(num_modes, operator, time=1.0, synthesis=FermionicLieTrotter())

or choose them for a whole circuit at once with the :class:`.F2FSynthesis` transpiler pass. Any class
with a ``synthesize(gate)`` method is a method (see :class:`.F2FSynthesisPlugin`), and the ones
registered under the ``qiskit_fermions.circuit.synthesis`` entry point are available by name through
the :class:`.F2FSynthesisPluginManager`.

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

   F2FSynthesisPlugin
   F2FSynthesisPluginManager
   FermionicLieTrotter
   FermionicSuzukiTrotter

The table below lists the builtin plugins, organized by the :class:`.FermionicGate` that they act
upon. These are the entry points of the ``qiskit_fermions.circuit.synthesis`` group, named
``<gate>.<method>``:

.. _builtin_f2f_synthesis_plugins:

.. table:: An overview of the builtin ``qiskit_fermions.circuit.synthesis`` entry-points.

   +---------------------+-------------------+-----------------------------------+
   | Gate name           | Plugin method     | Reference                         |
   +=====================+===================+===================================+
   | :class:`.Evolution` | ``LieTrotter``    | :class:`FermionicLieTrotter`      |
   +---------------------+-------------------+-----------------------------------+
   | :class:`.Evolution` | ``SuzukiTrotter`` | :class:`FermionicSuzukiTrotter`   |
   +---------------------+-------------------+-----------------------------------+
"""

from .lie_trotter import FermionicLieTrotter
from .plugin import F2FSynthesisPlugin, F2FSynthesisPluginManager
from .suzuki_trotter import FermionicSuzukiTrotter

__all__ = [
    "F2FSynthesisPlugin",
    "F2FSynthesisPluginManager",
    "FermionicLieTrotter",
    "FermionicSuzukiTrotter",
]
