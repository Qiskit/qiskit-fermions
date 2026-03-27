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

.. hint::
   Coming soon!

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

.. autosummary::
   :toctree: ../stubs/

   F2QSynthesis

.. _qiskit_fermions-transpiler-passes-synthesis-plugins:

Plugins
^^^^^^^

.. autosummary::
   :toctree: ../stubs/

   EvolutionSynthesis
"""

from .layout import CustomF2QLayout, TrivialF2QLayout
from .synthesis import EvolutionSynthesis, F2QSynthesis

__all__ = [
    "CustomF2QLayout",
    "EvolutionSynthesis",
    "F2QSynthesis",
    "TrivialF2QLayout",
]
