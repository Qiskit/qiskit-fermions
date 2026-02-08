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
==============
Mapper Library
==============

.. currentmodule:: qiskit_fermions.mappers.library

This module provides efficient implementations of commonly used operator representation mapper
routines.

.. autosummary::
   :toctree: ../stubs/

   jordan_wigner
   fermion_to_majorana
   majorana_to_fermion
"""

from qiskit_fermions._lib.mappers.mappers_library.jordan_wigner import jordan_wigner
from qiskit_fermions._lib.mappers.mappers_library.majorana_fermion import (
    fermion_to_majorana,
    majorana_to_fermion,
)

__all__ = [
    "fermion_to_majorana",
    "jordan_wigner",
    "majorana_to_fermion",
]
