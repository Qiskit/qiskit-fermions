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
==================
Transpiler Presets
==================

.. currentmodule:: qiskit_fermions.transpiler.presets

This module provides various functions for generating pre-defined transpiler pipelines. All
available ones are listed in the table below.

.. autosummary::
   :toctree: ../stubs/

   generate_preset_jw_pass_manager
"""

from .jordan_wigner import generate_preset_jw_pass_manager

__all__ = [
    "generate_preset_jw_pass_manager",
]
