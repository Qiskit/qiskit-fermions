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
======================
Representation Mappers
======================

.. currentmodule:: qiskit_fermions.mappers

This module provides a framework for implementing custom representation mapper routines.

.. note::
   The functions listed below do not have a counterpart in the C API.

.. autosummary::
   :toctree: ../stubs/

   map_fermion_action_generators
   map_majorana_action_generators
"""

from .fermion_generators import map_fermion_action_generators
from .majorana_generators import map_majorana_action_generators

__all__ = [
    "map_fermion_action_generators",
    "map_majorana_action_generators",
]
