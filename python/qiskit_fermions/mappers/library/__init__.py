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
==============
Mapper Library
==============

.. currentmodule:: qiskit_fermions.mappers.library

This module provides efficient implementations of commonly used operator representation mapper
routines.

Python wrappers
===============

The functions below are thin, type-agnostic convenience mappers. Each delegates to the appropriate
library implementation below under the hood, based on the type of the object it is given. They do
not implement any conversion logic themselves.

.. autosummary::
   :toctree: ../stubs/

   fermion_operator
   jordan_wigner
   majorana_operator
   ternary_tree

.. _jordan_wigner_mappers:

Library implementations
=======================

The functions below are the native, efficient mapper implementations backing this module.

.. autosummary::
   :toctree: ../stubs/

   fermion_jordan_wigner
   majorana_jordan_wigner
   edge_vertex_jordan_wigner
   transfer_vertex_jordan_wigner
   fermion_to_majorana
   majorana_to_fermion
   edge_vertex_to_fermion
   edge_vertex_to_majorana
   transfer_vertex_to_fermion
   transfer_vertex_to_majorana
   transfer_vertex_to_edge_vertex

.. _ternary_tree_mappers:

Ternary-tree encodings
======================

These map an operator through a ternary-tree encoding, which is built with
:class:`~qiskit_fermions.mappers.TernaryTree` and
:class:`~qiskit_fermions.mappers.TernaryTreeEncoding` from the
:mod:`~qiskit_fermions.mappers` framework module. Compile a tree once and reuse the encoding across
every operator mapped through it. The type-agnostic :func:`ternary_tree` wrapper dispatches to these.

.. autosummary::
   :toctree: ../stubs/

   fermion_ternary_tree
   majorana_ternary_tree
   edge_vertex_ternary_tree
   transfer_vertex_ternary_tree
"""

from __future__ import annotations

from qiskit_fermions._lib.mappers.mappers_library.edge_vertex import (
    edge_vertex_to_fermion,
    edge_vertex_to_majorana,
)
from qiskit_fermions._lib.mappers.mappers_library.jordan_wigner import (
    edge_vertex_jordan_wigner,
    fermion_jordan_wigner,
    majorana_jordan_wigner,
    transfer_vertex_jordan_wigner,
)
from qiskit_fermions._lib.mappers.mappers_library.majorana_fermion import (
    fermion_to_majorana,
    majorana_to_fermion,
)
from qiskit_fermions._lib.mappers.mappers_library.ternary_tree import (
    edge_vertex_ternary_tree,
    fermion_ternary_tree,
    majorana_ternary_tree,
    transfer_vertex_ternary_tree,
)
from qiskit_fermions._lib.mappers.mappers_library.transfer_vertex import (
    transfer_vertex_to_edge_vertex,
    transfer_vertex_to_fermion,
    transfer_vertex_to_majorana,
)

from .fermion_operator import fermion_operator
from .jordan_wigner import jordan_wigner
from .majorana_operator import majorana_operator
from .ternary_tree import ternary_tree

__all__ = [
    "edge_vertex_jordan_wigner",
    "edge_vertex_ternary_tree",
    "edge_vertex_to_fermion",
    "edge_vertex_to_majorana",
    "fermion_jordan_wigner",
    "fermion_operator",
    "fermion_ternary_tree",
    "fermion_to_majorana",
    "jordan_wigner",
    "majorana_jordan_wigner",
    "majorana_operator",
    "majorana_ternary_tree",
    "majorana_to_fermion",
    "ternary_tree",
    "transfer_vertex_jordan_wigner",
    "transfer_vertex_ternary_tree",
    "transfer_vertex_to_edge_vertex",
    "transfer_vertex_to_fermion",
    "transfer_vertex_to_majorana",
]
