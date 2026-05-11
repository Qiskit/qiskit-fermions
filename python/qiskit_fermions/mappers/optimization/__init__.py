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
Mapper Optimizations
====================

.. currentmodule:: qiskit_fermions.mappers.optimization

This module provides various routines for optimizing fermion-to-qubit mappings.
Their specific nature can vary greatly, so refer to the docstrings of each individual function for
more details.

.. autosummary::
   :toctree: ../stubs/

   build_excitation_span_minimization_model
"""

from .minimize_excitation_spans import build_excitation_span_minimization_model

__all__ = [
    "build_excitation_span_minimization_model",
]
