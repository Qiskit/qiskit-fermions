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

# ruff: noqa: D212,D415
"""
.. _qiskit_fermions-transpiler-passes-synthesis-plugins:

===================================
Transpiler Pass Synthesis (``F2Q``)
===================================

.. currentmodule:: qiskit_fermions.transpiler.passes.synthesis

The :class:`.F2QSynthesis` transpiler pass exposes a `plugin interface` (see
:class:`.F2QSynthesisPlugin`) through which implementations for mapping fermion-based instructions
to qubit-based ones can be registered. It is the counterpart of the fermion-to-fermion synthesis
that the :class:`.F2FSynthesis` pass performs one stage earlier (see
:mod:`~qiskit_fermions.circuit.library.synthesis`).

The available plugins are managed by the :class:`.F2QSynthesisPluginManager`. Choosing the right
combination of synthesis plugins is crucial and the :class:`.F2QSynthesis` transpilation pass does
**not** have any default plugins defined! For this, refer to the transpiler
:mod:`~qiskit_fermions.transpiler.presets`.

.. autosummary::
   :toctree: ../stubs/

   F2QSynthesisPlugin
   F2QSynthesisPluginManager

The plugins to use at runtime have to be registered in the :attr:`.F2QSynthesis.methods` attribute.
This can be done easily through the ``config`` argument of :class:`.F2QSynthesis` which specifies
the plugins to use from the :class:`.F2QSynthesisPluginManager` or it can be done manually by
writing into :attr:`.F2QSynthesis.methods` directly.

The table below lists the builtin synthesis plugins, organized by the :class:`.FermionicGate` that
they act upon:

.. _builtin_f2q_synthesis_plugins:

.. table:: An overview of the builtin ``qiskit_fermions.transpiler.synthesis`` entry-points.

   +----------------------------+-----------------------+-------------------------------------------------------+
   | Circuit instruction name   | Plugin method name    | Reference                                             |
   +============================+=======================+=======================================================+
   | :class:`.Evolution`        | ``MapperFn``          | :class:`MapperFnEvolutionSynthesis`                   |
   +----------------------------+-----------------------+-------------------------------------------------------+
   | :class:`.FermionicMeasure` | ``TrivialOccupation`` | :class:`TrivialOccupationFermionicMeasureSynthesis`   |
   +----------------------------+-----------------------+-------------------------------------------------------+
   | :class:`.InitializeModes`  | ``TrivialOccupation`` | :class:`TrivialOccupationInitializeModesSynthesis`    |
   +----------------------------+-----------------------+-------------------------------------------------------+

.. autosummary::
   :toctree: ../stubs/
   :class: sd-d-none

   MapperFnEvolutionSynthesis
   TrivialOccupationFermionicMeasureSynthesis
   TrivialOccupationInitializeModesSynthesis

Qubit-stage synthesis
---------------------

The plugins above map fermionic instructions to qubit ones. The following is instead a plugin of
Qiskit's :class:`~qiskit.transpiler.passes.HighLevelSynthesis` for the qubit stage, which recognizes
the image of a fermionic hop between adjacent modes and emits it as a single
:class:`~qiskit.circuit.library.XXPlusYYGate`. The Jordan-Wigner preset registers it.

.. autosummary::
   :toctree: ../stubs/

   HopEvolutionSynthesis

Additional helper functions
---------------------------

The Pauli term order that a mapper function produces is preserved through synthesis, since a product
formula synthesizes the terms in the order it receives them. These adapters wrap a mapper function to
choose that order:

.. autosummary::
   :toctree: ../stubs/

   group_wise
   simplify
"""

from ._deprecated_givens import (
    GivensDecompositionOrbitalRotationSynthesis,
    GivensDecompositionSlaterDeterminantSynthesis,
)
from .evolution import MapperFnEvolutionSynthesis, group_wise, simplify
from .f2q_synthesis import F2QSynthesis, F2QSynthesisConfig
from .fermionic_measure import TrivialOccupationFermionicMeasureSynthesis
from .hop_evolution import HopEvolutionSynthesis
from .initialize_modes import TrivialOccupationInitializeModesSynthesis
from .plugin import F2QSynthesisPlugin, F2QSynthesisPluginManager

__all__ = [
    "F2QSynthesis",
    "F2QSynthesisConfig",
    "F2QSynthesisPlugin",
    "F2QSynthesisPluginManager",
    "GivensDecompositionOrbitalRotationSynthesis",
    "GivensDecompositionSlaterDeterminantSynthesis",
    "HopEvolutionSynthesis",
    "MapperFnEvolutionSynthesis",
    "TrivialOccupationFermionicMeasureSynthesis",
    "TrivialOccupationInitializeModesSynthesis",
    "group_wise",
    "simplify",
]
