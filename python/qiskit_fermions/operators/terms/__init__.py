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
Operator Terms
==============

.. currentmodule:: qiskit_fermions.operators.terms

This module collects routines that operate on the individual terms of an operator, partitioning
them based on their structure. The submodules below group related functionality.

Grouping
--------

Please refer to :ref:`grouping_explanation` for a detailed explanation of this module's
functionality.

Assignment
^^^^^^^^^^

Rather than always relying on the user to provide the group indices themselves, this module provides
a collection of functions which determine the grouping information automatically.

.. autosummary::
   :toctree: ../stubs/

   group_terms_by_electronic_structure

Analysis
^^^^^^^^

Group indices carry no intrinsic meaning: the array only says which terms belong together, never
why. Accordingly, none of the functions below reports whether a grouping is "correct": each
answers one narrow, stated question about an existing grouping, leaving the interpretation to you.
They are provided so that an assumption a downstream consumer makes about a grouping can be checked
up front, rather than being paid for on every call.

.. autosummary::
   :toctree: ../stubs/

   group_coeff_means
   groups_are_hermitian
   groups_have_uniform_coeffs

Filtering
---------

This module provides convenience functions for removing terms from an operator that do not
contribute meaningfully to a downstream computation.

Members
^^^^^^^

.. autosummary::
   :toctree: ../stubs/

   filter_diagonal_terms

Ordering
--------

This module provides functions to reorder the terms of an operator. While the
resulting operators are mathematically identical, the term order can have
significant implications on algorithm behavior at runtime, due to the term
iteration order.

.. note::
   Unlike :func:`canonical_order`, :func:`order_terms` has no counterpart in the C API.

Members
^^^^^^^

.. autosummary::
   :toctree: ../stubs/

   canonical_order
   order_terms
"""

from .filtering import filter_diagonal_terms
from .grouping import (
    group_coeff_means,
    group_terms_by_electronic_structure,
    groups_are_hermitian,
    groups_have_uniform_coeffs,
)
from .ordering import canonical_order, order_terms

__all__ = [
    "canonical_order",
    "filter_diagonal_terms",
    "group_coeff_means",
    "group_terms_by_electronic_structure",
    "groups_are_hermitian",
    "groups_have_uniform_coeffs",
    "order_terms",
]
