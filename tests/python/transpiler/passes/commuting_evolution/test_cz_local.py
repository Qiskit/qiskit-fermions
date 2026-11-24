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

"""Tests of the CZ-then-local candidate."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import Context, CzLocal
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates import cz_local

from .groups import disjoint_windows, line_edges, same_unitary, zxz_windows, zz_interactions
from .helpers import TIME, all_pairs, context_of, problem_of


class TestCzLocal:
    def test_uses_only_allowed_pairs(self):
        op = zxz_windows(6)
        circuit = CzLocal().generate(problem_of(op), context_of(6, device=True))["diagonal"]
        cz_pairs = {
            tuple(sorted(circuit.find_bit(q).index for q in inst.qubits))
            for inst in circuit.data
            if inst.operation.name == "cz"
        }
        assert cz_pairs <= set(line_edges(6))

    def test_runs_on_user_edges_alone(self):
        """User edges give the candidate its pairs, without a device."""
        op = zxz_windows(6)
        context = Context(6, user_edges=tuple(line_edges(6)))
        variants = CzLocal().generate(problem_of(op), context)
        assert variants
        assert all(same_unitary(c, op, TIME) for c in variants.values())

    def test_copies_are_solved_one_by_one(self):
        """Two copies of a cell that each need a change of their single-qubit layer: the descent
        over the whole group could change only one at a time, and every layer in between is
        infeasible, so it found nothing."""
        cell = [("XZX", [0, 1, 2], 0.3), ("YZY", [0, 1, 2], 0.4)]
        copy = [(label, [q + 3 for q in qubits], c + 0.2) for label, qubits, c in cell]
        op = SparsePauliOp.from_sparse_list(cell + copy, num_qubits=6)
        variants = CzLocal().generate(problem_of(op), all_pairs(6))
        assert "diagonal" in variants
        assert same_unitary(variants["diagonal"], op, TIME)

    def test_copies_share_one_descent(self):
        """Three copies of a window set on a line: one descent, reused for the other two."""
        op = disjoint_windows(3, 4)
        problem = problem_of(op)
        cz_local._component_gates.cache_clear()
        assert cz_local.cz_local_gates(problem.x, problem.z, line_edges(12)) is not None
        assert cz_local._component_gates.cache_info().misses == 1

    def test_a_component_without_a_solution_ends_the_search(self):
        """No search over the whole group follows: it could not succeed either.

        Two copies of a column of the 2D guide's vertical flow set, which no ``L2 CZ L1`` puts on
        Z-strings: the first copy's descent answers for the group.
        """
        length = 4
        column = [("YXXY", [i, i + 1, length + i, length + i + 1], 0.5) for i in range(length - 1)]
        edges = [(i, i + 1) for i in range(length - 1)]
        edges += [(length + i, length + i + 1) for i in range(length - 1)]
        edges += [(i, length + i) for i in range(length)]
        shift = 2 * length
        copy = [(label, [q + shift for q in qubits], c) for label, qubits, c in column]
        op = SparsePauliOp.from_sparse_list(column + copy, num_qubits=2 * shift)
        problem = problem_of(op)
        both = edges + [(u + shift, v + shift) for u, v in edges]
        # the copies are joined at every site, which could help only a search over the whole group
        both += [(i, i + shift) for i in range(shift)]
        cz_local._component_gates.cache_clear()
        assert cz_local.cz_local_gates(problem.x, problem.z, sorted(both)) is None
        assert cz_local._component_gates.cache_info().misses == 1

    def test_z_strings_as_light_as_possible_need_no_descent(self, monkeypatch):
        """Eleven ``Z Z`` terms on seven qubits cannot all become weight-1 strings, of which there
        are seven: the empty Clifford is optimal, and the descent does not run."""

        def fail(*args, **kwargs):
            raise AssertionError("the descent ran")

        op = zz_interactions(7)
        problem = problem_of(op)
        cz_local._component_gates.cache_clear()
        monkeypatch.setattr(cz_local, "coordinate_descent", fail)
        assert cz_local.cz_local_gates(problem.x, problem.z, line_edges(7)) == []

    def test_a_lone_z_string_still_gets_lighter(self):
        """One ``Z Z`` term is not as light as it can be: ``H`` and a CZ turn it into ``Z``."""
        op = SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 0.5)], num_qubits=2)
        problem = problem_of(op)
        cz_local._component_gates.cache_clear()
        gates = cz_local.cz_local_gates(problem.x, problem.z, [(0, 1)])
        assert gates is not None
        assert ("cz", 0, 1) in gates
