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
======================
Circuit Representation
======================

.. currentmodule:: qiskit_fermions.circuit

With this module you can build :class:`~qiskit.circuit.QuantumCircuit` objects that are
interpreted as acting on `fermionic` modes rather than `qubits`.

The following objects are simple type aliases to make this re-interpretation more clear:

.. autosummary::
   :toctree: ../stubs/

   Fermion
   FermionRegister
   FermionSpecifier

..
   TODO: revisit the summary above once qiskit-sphinx-theme updates furo (and therefore support
   Sphinx 9, which added a .. autotype:: directive). That may result in better (and actually
   customizable) docstrings for these type aliases.

For actually expressing your circuits, you should use the :class:`.FermionCircuit` class along with
the library of :class:`.FermionGate` implementations in :mod:`qiskit_fermions.circuit.library`.

.. autosummary::
   :toctree: ../stubs/

   FermionCircuit
   FermionGate
"""

from collections.abc import Sequence
from typing import TypeAlias

from qiskit.circuit import QuantumRegister, Qubit

Fermion: TypeAlias = Qubit

FermionRegister: TypeAlias = QuantumRegister

FermionSpecifier: TypeAlias = Fermion | FermionRegister | int | slice | Sequence[Fermion | int]
"""A type alias equivalent to Qiskit's ``QubitSpecifier`` but for fermionic modes."""

# NOTE: we must explicitly define the type aliases _before_ the following imports to ensure that
# they can actually use those type aliases themselves.
# ruff: noqa: E402
from .fermion_circuit import FermionCircuit
from .fermion_gate import FermionGate

__all__ = [
    "Fermion",
    "FermionCircuit",
    "FermionGate",
    "FermionRegister",
    "FermionSpecifier",
]
