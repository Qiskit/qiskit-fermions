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
r"""
=========
Protocols
=========

.. currentmodule:: qiskit_fermions.protocols

This module provides various protocols used throughout the package.

.. autosummary::
   :toctree: ../stubs/

   SupportsCommutators
   SupportsFermionOperator
   SupportsLinearOperator
   SupportsMajoranaOperator

"""

from __future__ import annotations

from .commutators import SupportsCommutators
from .fermion_operator import SupportsFermionOperator
from .linear_operator import SupportsLinearOperator
from .majorana_operator import SupportsMajoranaOperator

__all__ = [
    "SupportsCommutators",
    "SupportsFermionOperator",
    "SupportsLinearOperator",
    "SupportsMajoranaOperator",
]
