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
======================
Representation Mappers
======================

.. currentmodule:: qiskit_fermions.mappers

This module provides a framework for implementing custom representation mapper routines.

Generator-based mappers
=======================

These build a mapper from the Pauli images of an algebra's generators.

.. note::
   These four functions do not have a counterpart in the C API.

.. autosummary::
   :toctree: ../stubs/

   map_fermion_action_generators
   map_majorana_action_generators
   map_edge_vertex_generators
   map_transfer_vertex_generators

Ternary trees
=============

A ternary tree on :math:`N` nodes defines a fermion-to-qubit encoding of :math:`N` modes onto
:math:`N` qubits, and the familiar encodings are particular tree shapes: a chain along ``"Z"`` is
Jordan-Wigner, a chain along ``"X"`` is the parity encoding, and the balanced tree attains the
optimal Pauli weight :math:`\lceil \log_3(2N+1) \rceil`. Arbitrary trees (including the irregular
ones grown from a device's coupling graph) are built by passing a parent-and-label specification to
:class:`.TernaryTree` directly, which is what makes this a framework rather than a fixed set of
encodings.

A tree is compiled into a :class:`.TernaryTreeEncoding` once and then passed to any of the
ternary-tree mappers in :mod:`~qiskit_fermions.mappers.library`. Properties derived from the encoding
itself, such as :meth:`.TernaryTreeEncoding.total_parity`, are methods on that class.

.. autosummary::
   :toctree: ../stubs/

   TernaryTree
   TernaryTreeEncoding
"""

from qiskit_fermions._lib.mappers.ternary_tree_structures import (
    TernaryTree,
    TernaryTreeEncoding,
)

from .edge_vertex_generators import map_edge_vertex_generators
from .fermion_generators import map_fermion_action_generators
from .majorana_generators import map_majorana_action_generators
from .transfer_vertex_generators import map_transfer_vertex_generators

__all__ = [
    "TernaryTree",
    "TernaryTreeEncoding",
    "map_edge_vertex_generators",
    "map_fermion_action_generators",
    "map_majorana_action_generators",
    "map_transfer_vertex_generators",
]
