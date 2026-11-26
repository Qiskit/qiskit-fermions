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

"""Helpers shared by the tests of the candidates and of the data they receive."""

from __future__ import annotations

import itertools

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import (
    Context,
    Problem,
    abstract_cost,
)

TIME = 0.37


def problem_of(op: SparsePauliOp) -> Problem:
    """The problem of ``op`` at :data:`TIME`."""
    return Problem.from_operator(op, TIME)


def context_of(num_qubits: int, *, device: bool = False) -> Context:
    """No connectivity, or a line of qubits placed one to one on a line device."""
    if not device:
        return Context(num_qubits)
    return Context(
        num_qubits,
        coupling_map=CouplingMap.from_line(num_qubits),
        physical_qubits=tuple(range(num_qubits)),
    )


def all_pairs(num_qubits: int) -> Context:
    """Every pair of qubits allowed, as user edges."""
    return Context(num_qubits, user_edges=tuple(itertools.combinations(range(num_qubits), 2)))


def cheapest(variants: dict[str, QuantumCircuit]) -> tuple[int, int]:
    """The lowest abstract cost among the variants."""
    return min(abstract_cost(c) for c in variants.values())
