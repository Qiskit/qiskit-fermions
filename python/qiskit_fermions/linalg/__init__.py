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
Linear Algebra Utilities
========================

.. currentmodule:: qiskit_fermions.linalg

This module provides various linear algebra utilities.

.. autosummary::
   :toctree: ../stubs/

   givens_decomposition
   givens_decomposition_slater
   double_factorized_2body
   double_factorized_t2
   reconstruct_t2
   double_factorized_t2_alpha_beta
   reconstruct_t2_alpha_beta
"""

from qiskit_fermions._lib.linalg.double_factorized import (
    double_factorized_2body,
    double_factorized_t2,
    double_factorized_t2_alpha_beta,
    reconstruct_t2,
    reconstruct_t2_alpha_beta,
)
from qiskit_fermions._lib.linalg.givens import (
    givens_decomposition,
    givens_decomposition_slater,
)

__all__ = [
    "double_factorized_2body",
    "double_factorized_t2",
    "double_factorized_t2_alpha_beta",
    "givens_decomposition",
    "givens_decomposition_slater",
    "reconstruct_t2",
    "reconstruct_t2_alpha_beta",
]
