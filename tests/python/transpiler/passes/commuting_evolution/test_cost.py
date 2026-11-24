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

"""Tests of the built-in cost functions."""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import (
    Context,
    abstract_cost,
    cost,
    device_cost,
)
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.cost import _two_qubit


class TestCost:
    def test_abstract_cost_counts_every_two_qubit_gate(self):
        """The layout candidates emit CZ, the others CX; both must count on the same scale."""
        cx, cz = QuantumCircuit(2), QuantumCircuit(2)
        cx.cx(0, 1)
        cz.cz(0, 1)
        assert abstract_cost(cx) == abstract_cost(cz) == (1, 1)

    def test_a_permutation_left_by_the_router_is_undone(self):
        """A gate between the ends of a line makes the router move a qubit; the scored circuit puts
        it back."""
        line = CouplingMap.from_line(3)
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 2)
        circuit.rz(0.3, 2)
        device = {"coupling_map": line, "basis_gates": list(cost.SCORING_BASIS)}
        routed = transpile(
            circuit,
            **device,
            initial_layout=[0, 1, 2],
            optimization_level=1,
            seed_transpiler=cost.SEED,
        )
        assert routed.layout.final_index_layout() != [0, 1, 2]
        restored = cost._undo_permutation(routed, line, device)
        assert restored.layout is None or restored.layout.final_index_layout() == [0, 1, 2]
        assert Operator(restored).equiv(Operator(circuit))
        context = Context(3, coupling_map=line, physical_qubits=(0, 1, 2))
        assert device_cost(circuit, context) == (
            restored.depth(_two_qubit),
            restored.num_nonlocal_gates(),
        )

    def test_the_target_basis_is_used(self):
        """With a target, the circuit is translated into the target's gates, here ECR instead of
        CZ."""
        backend = GenericBackendV2(
            3, basis_gates=["ecr", "rz", "sx", "x"], coupling_map=[[0, 1], [1, 2]], seed=1
        )
        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        context = Context(
            3, coupling_map=backend.coupling_map, target=backend.target, physical_qubits=(0, 1, 2)
        )
        assert device_cost(circuit, context) == (1, 1)

    def test_the_basis_argument_is_used(self):
        """Without a target, the ``basis`` argument decides the scoring basis, routed or not."""
        circuit = QuantumCircuit(2)
        circuit.swap(0, 1)
        with_swap = ["swap", *cost.SCORING_BASIS]
        # the default basis needs three CZ gates for a SWAP
        assert abstract_cost(circuit) == (3, 3)
        assert abstract_cost(circuit, basis=with_swap) == (1, 1)
        line = Context(2, coupling_map=CouplingMap.from_line(2), physical_qubits=(0, 1))
        assert device_cost(circuit, line) == (3, 3)
        assert device_cost(circuit, line, basis=with_swap) == (1, 1)
