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

Principal Components
--------------------

.. autosummary::
   :toctree: ../stubs/

   F2QSynthesis

.. _qiskit_fermions-transpiler-passes-layout:

Layouting Passes
----------------

.. autosummary::
   :toctree: ../stubs/

   TrivialF2QLayout

.. _qiskit_fermions-transpiler-passes-synthesis-plugins:

Synthesis Plugins
-----------------

.. autosummary::
   :toctree: ../stubs/

   EvolutionSynthesis
"""

from .layout import TrivialF2QLayout
from .synthesis import EvolutionSynthesis, F2QSynthesis

__all__ = [
    "EvolutionSynthesis",
    "F2QSynthesis",
    "TrivialF2QLayout",
]
