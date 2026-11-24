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

"""Tests of the pair-block variants of the CZ-then-local candidate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, PauliList, SparsePauliOp
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import (
    DEFAULT_CANDIDATES,
    Context,
    CzLocal,
    Problem,
    _budget,
    _pair_blocks,
    abstract_cost,
)
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution._elimination import to_circuit
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution._pair_blocks import (
    fswap_gates,
    on_pairs,
)

from .groups import GROUPS, disjoint_windows, line_edges, same_unitary, zxz_windows
from .helpers import TIME, all_pairs, context_of, problem_of


def _two_qubit_pairs(circuit: QuantumCircuit) -> set[tuple[int, int]]:
    """The qubit pairs of the two-qubit gates of the decomposed circuit."""
    decomposed = circuit.decompose(reps=6)
    return {
        tuple(sorted(decomposed.find_bit(q).index for q in inst.qubits))  # type: ignore[misc]
        for inst in decomposed.data
        if len(inst.qubits) == 2
    }


def _column(length: int) -> tuple[SparsePauliOp, list[tuple[int, int]]]:
    """A vertical flow set of the 2D Fermi-Hubbard guide on one lattice column, and its ladder.

    The sites of the column are qubits ``0 .. L - 1`` and their ancillas ``L .. 2L - 1``; the hops
    are ``Y X X Y`` on two neighboring sites and their ancillas. The allowed pairs form a ladder:
    the sites in a line, the ancillas in a line, and a rung from every site to its ancilla.
    """
    terms = [("YXXY", [i, i + 1, length + i, length + i + 1], 0.5) for i in range(length - 1)]
    edges = [(i, i + 1) for i in range(length - 1)]
    edges += [(length + i, length + i + 1) for i in range(length - 1)]
    edges += [(i, length + i) for i in range(length)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=2 * length), sorted(edges)


def _cheapest_other(problem: Problem, context: Context) -> tuple[int, int]:
    """The cheapest circuit of every default candidate except the pair-block variants."""
    return min(
        abstract_cost(circuit)
        for candidate in DEFAULT_CANDIDATES
        for name, circuit in candidate.generate(problem, context).items()
        if not name.endswith("pair-blocks")
    )


class TestPairBlocks:
    def test_variant_names(self):
        variants = CzLocal().generate(problem_of(zxz_windows()), context_of(7, device=True))
        assert list(variants) == ["diagonal", "pair-blocks"]

    @pytest.mark.parametrize(
        ("name", "variant"),
        [
            ("vertical_bond", "pair-blocks"),
            ("horizontal_bond", "fswap-pair-blocks"),
            ("zxz_windows", "pair-blocks"),
        ],
    )
    def test_every_gate_is_on_an_allowed_pair(self, name, variant):
        """No routing is needed, so the circuit leaves the layout unchanged."""
        op = GROUPS[name]()
        circuit = CzLocal().generate(problem_of(op), context_of(op.num_qubits, device=True))[
            variant
        ]
        assert _two_qubit_pairs(circuit) <= set(line_edges(op.num_qubits))

    def test_one_layer_finds_the_vertical_bond(self):
        """One CZ decouples the ancilla, and one block holds ``{XX, YY, ZZ}``: cheaper than any
        diagonal circuit."""
        op = GROUPS["vertical_bond"]()
        problem, context = problem_of(op), context_of(3, device=True)
        variants = CzLocal().generate(problem, context)
        assert abstract_cost(variants["pair-blocks"]) < _cheapest_other(problem, context)

    def test_the_horizontal_bond_needs_fswap_moves(self):
        """One layer can only reach the non-edge ``(q_1, q_2)``; an fSWAP move puts the block onto
        an edge."""
        op = GROUPS["horizontal_bond"]()
        variants = CzLocal().generate(problem_of(op), context_of(3, device=True))
        assert "pair-blocks" not in variants
        assert "fswap-pair-blocks" in variants

    def test_fswap_moves_only_where_one_layer_does_not_suffice(self):
        """Where one layer solves every component, the fSWAP variant would be the same circuit."""
        op = GROUPS["vertical_bond"]()
        variants = CzLocal().generate(problem_of(op), context_of(3, device=True))
        assert "pair-blocks" in variants
        assert "fswap-pair-blocks" not in variants

    def test_the_hop_cell_with_all_pairs(self):
        """A single CZ splits the cell into two entangled pairs, which no diagonal circuit
        matches."""
        op = GROUPS["hop_cell"]()
        problem, context = problem_of(op), all_pairs(4)
        variants = CzLocal().generate(problem, context)
        assert same_unitary(variants["pair-blocks"], op, TIME)
        assert abstract_cost(variants["pair-blocks"]) < _cheapest_other(problem, context)

    def test_the_hop_cell_on_a_line_needs_an_fswap(self):
        """On a line the pairs ``(0, 2)`` and ``(1, 3)`` are not edges; an fSWAP on ``(1, 2)`` makes
        them ``(0, 1)`` and ``(2, 3)``, which beats every diagonal circuit."""
        op = GROUPS["hop_cell"]()
        problem, context = problem_of(op), context_of(4, device=True)
        variants = CzLocal().generate(problem, context)
        assert "pair-blocks" not in variants
        circuit = variants["fswap-pair-blocks"]
        assert _two_qubit_pairs(circuit) <= set(line_edges(4))
        assert abstract_cost(circuit) < _cheapest_other(problem, context)

    def test_copies_share_one_search(self):
        """Three copies of a window set on a line: one search, moved to the other two; the fSWAP
        variant asks the same memoized search instead of repeating it."""
        op = disjoint_windows(3, 4)
        problem = problem_of(op)
        edges = line_edges(op.num_qubits)
        _pair_blocks._one_layer_gates.cache_clear()
        gates = _pair_blocks.pair_block_one_layer(problem.x, problem.z, edges)
        assert gates is not None
        assert _pair_blocks._one_layer_gates.cache_info().misses == 1
        _pair_blocks.pair_block_fswap(problem.x, problem.z, edges)
        assert _pair_blocks._one_layer_gates.cache_info().misses == 1
        # the moved solutions are exact: test_every_variant_is_exact covers the two-copy group

    @pytest.mark.parametrize("length", [4, 8, 12, 40])
    def test_a_long_column_is_solved_within_the_budget(self, length):
        """The search finishes within the default node budget, for 24 and 80 qubits too.

        In qubit-index order the search met the sites long before their ancillas, so it could not
        reject a term until late, and it exhausted the budget from a column of seven sites on. The
        supports are Python integers, so a component may be wider than 64 qubits. The cheapest
        circuit has one CZ per site and a two-CX block per hop at every length.
        """
        op, edges = _column(length)
        problem = problem_of(op)
        solutions = list(
            _pair_blocks.one_layer_solutions(
                problem.x, problem.z, edges, _budget.PAIR_BLOCK_NODE_LIMIT
            )
        )
        key = solutions[-1][0]
        assert key[:2] == (2 * length + 2 * (length - 1), length)

    def test_a_column_is_exact(self):
        op, edges = _column(3)
        context = Context(op.num_qubits, user_edges=tuple(edges))
        circuit = CzLocal().generate(problem_of(op), context)["pair-blocks"]
        assert _two_qubit_pairs(circuit) <= set(edges)
        assert same_unitary(circuit, op, TIME)

    def test_the_cheapest_solution_survives_the_pruning(self):
        """Every solution yielded is cheaper than the ones before it, down to the cheapest one.

        The ``Z X Z`` windows on a line have many solutions of equal cost that differ only in the
        letters of settled terms. The search merges them, and still ends at the optimum that the
        exhaustive search found: three CZ gates and three two-CX blocks.
        """
        problem = problem_of(zxz_windows(7))
        solutions = list(
            _pair_blocks.one_layer_solutions(
                problem.x, problem.z, line_edges(7), _budget.PAIR_BLOCK_NODE_LIMIT
            )
        )
        keys = [key for key, _, _ in solutions]
        assert keys == sorted(set(keys), reverse=True)
        assert keys[-1] == (12, 3, 0)

    def test_fswap_gates(self):
        """The two-CX circuit is ``SWAP . CZ`` up to a global phase."""
        expected = QuantumCircuit(2)
        expected.swap(0, 1)
        expected.cz(0, 1)
        assert Operator(to_circuit(fswap_gates(0, 1), 2)).equiv(Operator(expected))

    def test_on_pairs(self):
        """Supports of one qubit or one allowed pair, the pairs disjoint."""
        allowed = {0b011, 0b110}

        def images(*labels):
            paulis = PauliList(labels)
            return paulis.x.astype(np.uint8), paulis.z.astype(np.uint8)

        assert on_pairs(*images("IXX", "IZZ", "ZII"), allowed)
        assert not on_pairs(*images("XIX"), allowed)  # (0, 2) is not allowed
        assert not on_pairs(*images("IXX", "XXI"), allowed)  # (0, 1) and (1, 2) overlap
        assert not on_pairs(*images("XXX"), allowed)  # three qubits
