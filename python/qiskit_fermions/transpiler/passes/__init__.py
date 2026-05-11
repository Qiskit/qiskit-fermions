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
====================
Transpilation Passes
====================

.. currentmodule:: qiskit_fermions.transpiler.passes

This module provides various transpiler passes for the stages explained in
:mod:`qiskit_fermions.transpiler`.

.. _qiskit_fermions-transpiler-passes-optimization:

Optimization Passes
-------------------

These passes provide different kinds of optimization of :class:`.FermionicCircuit` instances.

.. autosummary::
   :toctree: ../stubs/

   QDriftTrotterization

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

The main logic for mapping a :class:`.FermionicCircuit` to a :class:`~qiskit.circuit.QuantumCircuit`
is implemented by a single synthesis pass, namely the :class:`.F2QSynthesis`. It is conceptually
similar to Qiskit's :class:`~qiskit.transpiler.passes.HighLevelSynthesis` because it simply iterates
over the :class:`.FermionicCircuit` instructions and delegates the mapping to qubit-based instructions
to :ref:`qiskit_fermions-transpiler-passes-synthesis-plugins` for each of the encountered types of
:class:`.FermionicGate`.

.. autosummary::
   :toctree: ../stubs/

   F2QSynthesis

.. _qiskit_fermions-transpiler-passes-synthesis-plugins:

Plugins
^^^^^^^

As mentioned above, the :class:`.F2QSynthesis` transpiler pass exposes a `plugin interface` through
which implementations for mapping fermion-based instructions to qubit-based ones can be registered.
For more details on how to implement your own plugin, refer to :attr:`.F2QSynthesis.plugins`.

For most common gates provided by :mod:`qiskit_fermions.circuit.library`, this module already
provides builtin plugins:

.. autosummary::
   :toctree: ../stubs/

   EvolutionSynthesis
   InitializeModesSynthesis
   OrbitalRotationSynthesis
"""

from .layout import CustomF2QLayout, TrivialF2QLayout
from .optimization import QDriftTrotterization
from .synthesis import (
    EvolutionSynthesis,
    F2QSynthesis,
    F2QSynthesisPlugin,
    InitializeModesSynthesis,
    OrbitalRotationSynthesis,
)

__all__ = [
    "CustomF2QLayout",
    "EvolutionSynthesis",
    "F2QSynthesis",
    "F2QSynthesisPlugin",
    "InitializeModesSynthesis",
    "OrbitalRotationSynthesis",
    "QDriftTrotterization",
    "TrivialF2QLayout",
]
