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

A :class:`typing.Protocol` is an interface that a class can fulfill structurally, without
explicitly inheriting from it. For example, any class that implements a ``__len__`` method
satisfies :class:`~typing.Sized`, whether or not it says so anywhere in its class hierarchy.

The protocols in this module follow the same idea, but each one centers on a single, otherwise
unremarkable dunder-style method (for example ``_linear_operator_``). A class implementing that
one method thereby satisfies the corresponding protocol -- :class:`.SupportsLinearOperator` in
this example -- and becomes usable wherever that protocol is expected, without any explicit
registration. Note that "protocol" is sometimes also used loosely to refer to the type-agnostic
dispatch function built on top of a protocol (for example :func:`.linear_operator`) rather than
the interface itself; the distinction should be clear from context.

Most protocols below follow this pattern paired with a small, type-agnostic **helper function**
that dispatches to the protocol method on whatever object it is given:

.. list-table::
   :header-rows: 1

   * - Protocol
     - Method
     - Helper function
   * - :class:`.SupportsFermionOperator`
     - ``_fermion_operator_(self)``
     - :func:`.fermion_operator`
   * - :class:`.SupportsMajoranaOperator`
     - ``_majorana_operator_(self)``
     - :func:`.majorana_operator`
   * - :class:`.SupportsLinearOperator`
     - ``_linear_operator_(self, norb, nelec)``
     - :func:`.linear_operator`
   * - :class:`.SupportsCommutators`
     - ``_commutator_``, ``_anti_commutator_``, ``_double_commutator_`` (all ``@staticmethod``)
     - :func:`.commutator`, :func:`.anti_commutator`, :func:`.double_commutator`

.. autosummary::
   :toctree: ../stubs/
   :class: sd-d-none

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
