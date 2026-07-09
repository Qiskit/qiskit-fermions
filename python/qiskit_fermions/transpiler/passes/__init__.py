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
=================
Transpiler Passes
=================

.. currentmodule:: qiskit_fermions.transpiler.passes

This module provides various transpiler passes for the stages explained in
:mod:`qiskit_fermions.transpiler`.

.. _qiskit_fermions-transpiler-passes-optimization:

Optimization Passes
-------------------

These passes provide different kinds of optimization of :class:`.FermionicDAGCircuit` instances.

.. autosummary::
   :toctree: ../stubs/

   QDriftTrotterization
   RelabelModes

.. _qiskit_fermions-transpiler-passes-layout:

Layouting Passes
----------------

These passes are designed to help configuring the global
:type:`~qiskit_fermions.transpiler.F2QLayout` setting for the transpilation process. This setting
needs to be placed in the ``f2q_layout`` field of the
:attr:`~qiskit.passmanager.PassManagerState.property_set`, from where it will be read during the
:ref:`qiskit_fermions-transpiler-passes-synthesis`.

.. autosummary::
   :toctree: ../stubs/

   TrivialF2QLayout
   CustomF2QLayout

.. _qiskit_fermions-transpiler-passes-synthesis:

Synthesis Passes
----------------

The main logic for mapping a :class:`.FermionicDAGCircuit` to a
:class:`~qiskit.dagcircuit.DAGCircuit` is implemented by a single synthesis pass, namely the
:class:`.F2QSynthesis`. It is conceptually similar to Qiskit's
:class:`~qiskit.transpiler.passes.HighLevelSynthesis` because it simply iterates over the
:class:`.FermionicDAGCircuit` instructions and delegates the mapping to qubit-based instructions to
:ref:`qiskit_fermions-transpiler-passes-synthesis-plugins` for each of the encountered types of
:class:`.FermionicGate`.

.. autosummary::
   :toctree: ../stubs/

   F2QSynthesis
   ~synthesis.synthesis.F2QSynthesisConfig
"""

from .layout import CustomF2QLayout, TrivialF2QLayout
from .optimization import QDriftTrotterization, RelabelModes
from .synthesis import (
    F2QSynthesis,
    F2QSynthesisConfig,
    F2QSynthesisPlugin,
    F2QSynthesisPluginManager,
    GivensDecompositionOrbitalRotationSynthesis,
    MapperFnEvolutionSynthesis,
    TrivialOccupationInitializeModesSynthesis,
)

__all__ = [
    "CustomF2QLayout",
    "F2QSynthesis",
    "F2QSynthesisConfig",
    "F2QSynthesisPlugin",
    "F2QSynthesisPluginManager",
    "GivensDecompositionOrbitalRotationSynthesis",
    "MapperFnEvolutionSynthesis",
    "QDriftTrotterization",
    "RelabelModes",
    "TrivialF2QLayout",
    "TrivialOccupationInitializeModesSynthesis",
]
