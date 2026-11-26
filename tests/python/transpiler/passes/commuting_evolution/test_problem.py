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

"""Tests of the problem and of the context that the candidates receive."""

from __future__ import annotations

import pytest
from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import Context, Problem

from .helpers import TIME


class TestProblem:
    def test_the_terms_are_a_sparse_observable_next_to_their_arrays(self):
        op = SparseObservable.from_list([("XYZ", 0.5), ("ZZI", -0.25), ("XYZ", 0.25)])
        problem = Problem.from_operator(op, TIME)
        assert isinstance(problem.operator, SparseObservable)
        assert problem.operator.num_terms == 2  # simplified
        assert problem.paulis.to_labels() == ["XYZ", "ZZI"]
        assert problem.coeffs.tolist() == [0.75, -0.25]
        assert problem.x.tolist() == [[0, 1, 1], [0, 0, 0]] and problem.z.tolist() == [
            [1, 1, 0],
            [0, 1, 1],
        ]
        assert SparsePauliOp.from_sparse_observable(problem.operator).equiv(
            SparsePauliOp(["XYZ", "ZZI"], [0.75, -0.25])
        )

    def test_projectors_become_paulis(self):
        """``|0><0| = (I + Z) / 2``."""
        problem = Problem.from_operator(SparseObservable.from_list([("0", 1.0)]), TIME)
        assert sorted(problem.paulis.to_labels()) == ["I", "Z"]
        assert problem.coeffs.tolist() == [0.5, 0.5]


class TestContext:
    def test_nothing_known(self):
        context = Context(4)
        assert context.edges is None

    def test_device_edges_among_the_physical_qubits(self):
        """Qubits 0, 1, 2 of the gate on device qubits 4, 3, 1 of a line: only (4, 3) is a device
        edge."""
        context = Context(3, coupling_map=CouplingMap.from_line(6), physical_qubits=(4, 3, 1))
        assert context.edges == [(0, 1)]

    def test_user_edges_take_precedence(self):
        context = Context(
            3,
            coupling_map=CouplingMap.from_line(3),
            physical_qubits=(0, 1, 2),
            user_edges=((2, 0),),
        )
        assert context.edges == [(0, 2)]

    def test_everything_but_the_width_is_keyword_only(self):
        with pytest.raises(TypeError):
            Context(3, CouplingMap.from_line(3))  # type: ignore[misc]

    def test_a_coupling_map_without_physical_qubits_gives_no_edges(self):
        assert Context(4, coupling_map=CouplingMap.from_line(4)).edges is None
