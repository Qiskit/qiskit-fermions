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
Operator Terms
===============

.. currentmodule:: qiskit_fermions.operators.terms

This module collects routines that operate on the individual terms of an operator, partitioning
them based on their structure. The submodules below group related functionality.

Grouping
--------

Please refer to :ref:`grouping_explanation` for a detailed explanation of this module's
functionality.

Library
^^^^^^^

Rather than always relying on the user to provide the group indices themselves, this module provides
a collection of functions which determine the grouping information automatically.

.. autosummary::
   :toctree: ../stubs/

   group_terms_by_electronic_structure

Filtering
---------

This module provides convenience functions for removing terms from an operator that do not
contribute meaningfully to a downstream computation.

.. autosummary::
   :toctree: ../stubs/

   filter_diagonal_terms
   filter_terms_by_num_unique_modes
"""

from .filtering import filter_diagonal_terms, filter_terms_by_num_unique_modes
from .grouping import group_terms_by_electronic_structure

__all__ = [
    "filter_diagonal_terms",
    "filter_terms_by_num_unique_modes",
    "group_terms_by_electronic_structure",
]
