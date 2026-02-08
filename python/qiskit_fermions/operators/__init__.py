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
========================
Operator Representations
========================

.. currentmodule:: qiskit_fermions.operators

This module provides various data structures for representing fermionic operators in different
bases.

Fermion Operator
----------------

This operator represents fermionic operators in terms of the second-quantization creation and
annihilation operators.

.. autosummary::
   :toctree: ../stubs/

   FermionAction
   FermionOperator
   cre
   ann

Majorana Operator
-----------------

This operator represents fermionic operators in terms of Majorana fermions.

.. autosummary::
   :toctree: ../stubs/

   MajoranaAction
   MajoranaOperator
   gamma
"""

from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
from qiskit_fermions._lib.operators.majorana_operator import MajoranaOperator

from .fermion_action import FermionAction, ann, cre
from .majorana_action import MajoranaAction, gamma

__all__ = [
    "FermionAction",
    "FermionOperator",
    "MajoranaAction",
    "MajoranaOperator",
    "ann",
    "cre",
    "gamma",
]
