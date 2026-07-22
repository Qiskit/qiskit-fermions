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
===============
Circuit Library
===============

.. currentmodule:: qiskit_fermions.circuit.library

This module provides a library of :class:`.FermionicGate` implementations.


.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_without_inheritance.rst

   Evolution
   InitializeModes
   OrbitalRotation
   PrepareSlaterDeterminant
   UCJ
"""

from .evolution import Evolution
from .initialize_modes import InitializeModes
from .orbital_rotation import OrbitalRotation
from .prepare_slater_determinant import PrepareSlaterDeterminant
from .ucj import UCJ

__all__ = [
    "UCJ",
    "Evolution",
    "InitializeModes",
    "OrbitalRotation",
    "PrepareSlaterDeterminant",
]
