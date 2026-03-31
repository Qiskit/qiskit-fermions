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
Operator Grouping
=================

.. currentmodule:: qiskit_fermions.operators.grouping

.. autosummary::
   :toctree: ../stubs/

   group_terms_by_electronic_structure
"""
# TODO: complete docstring

from qiskit_fermions._lib.operators.operators_grouping.electronic_structure import (
    group_terms_by_electronic_structure,
)

__all__ = [
    "group_terms_by_electronic_structure",
]
