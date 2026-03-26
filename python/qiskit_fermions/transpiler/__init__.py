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
==========
Transpiler
==========

.. currentmodule:: qiskit_fermions.transpiler

.. autosummary::
   :toctree: ../stubs/

   F2QLayout
"""

from __future__ import annotations

from typing import TypeAlias

from qiskit.circuit import QuantumRegister

F2QLayout: TypeAlias = dict[QuantumRegister, QuantumRegister]

__all__ = [
    "F2QLayout",
]
