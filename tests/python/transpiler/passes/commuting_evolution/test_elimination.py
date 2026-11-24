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

"""Tests of the candidates built on the GF(2) elimination: plain, majority, twin-fold and search."""

from __future__ import annotations

import pytest
from qiskit.quantum_info import Clifford, SparsePauliOp
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import (
    Context,
    Majority,
    Plain,
    Search,
    TwinFold,
)
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.clifford import (
    twin_fold_clifford,
)
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.search import (
    VARIANTS,
)

from .groups import GROUPS, line_edges, plaquettes, zxz_windows
from .helpers import cheapest, context_of, problem_of


class TestElimination:
    def test_plain_offers_both_cx_orders(self):
        variants = Plain().generate(problem_of(GROUPS["random"]()), context_of(6))
        assert list(variants) in (["elimination-order", "layered"], ["elimination-order"])

    def test_one_variant_when_the_orders_coincide(self):
        """A single two-qubit term needs one CX at most, so both orders are the same gates."""
        op = SparsePauliOp(["XX", "ZZ"], [0.4, 0.6])
        assert list(Majority().generate(problem_of(op), context_of(2))) == ["elimination-order"]


class TestTwinFold:
    def test_offers_nothing_without_twins(self):
        """A group with no two qubits that occur in the same terms has nothing to fold."""
        problem = problem_of(zxz_windows())
        assert twin_fold_clifford(problem) is None
        assert TwinFold().generate(problem, context_of(7)) == {}

    @pytest.mark.parametrize("sites", [2, 3, 4])
    def test_folds_the_plaquettes_to_weight_two(self, sites):
        """The rungs of the two-chain model are twins, and folding turns every weight-4 term into a
        weight-2 one."""
        op = plaquettes(sites)
        clifford = twin_fold_clifford(problem_of(op))
        assert clifford is not None
        images = op.paulis.evolve(Clifford(clifford), frame="s")
        assert not images.x.any()
        assert set(images.z.sum(axis=1)) == {2}

    def test_offers_both_orders_of_the_two_body_layers(self):
        op = plaquettes(3)
        assert list(TwinFold().generate(problem_of(op), context_of(op.num_qubits))) == [
            "forward",
            "reversed",
        ]

    def test_beats_plain_on_the_plaquettes(self):
        """Plain leaves the weight-4 terms as parity trees; twin-fold's circuit is shallower and
        smaller."""
        op = plaquettes(4)
        problem, context = problem_of(op), context_of(op.num_qubits)
        assert cheapest(TwinFold().generate(problem, context)) < cheapest(
            Plain().generate(problem, context)
        )


class TestSearch:
    def test_nothing_for_a_placed_gate(self):
        """Placed, the layout candidates do better; with only user edges the gate is not placed, and
        it runs."""
        problem = problem_of(GROUPS["random"]())
        assert Search().generate(problem, context_of(6, device=True)) == {}
        assert Search().generate(problem, Context(6, user_edges=tuple(line_edges(6))))

    def test_variants_are_shape_and_order(self):
        problem, context = problem_of(GROUPS["random"]()), context_of(6)
        assert list(Search().generate(problem, context)) == list(VARIANTS)
        star = Search(variants=["star-elimination-order", "star-layered"]).generate(
            problem, context
        )
        assert list(star) == ["star-layered", "star-elimination-order"]
        only = Search(variants=["star-elimination-order"]).generate(problem, context)
        assert list(only) == ["star-elimination-order"]

    def test_is_not_worse_than_plain_unrouted(self):
        """The descent starts at the identity basis, which is the plain Clifford, and only accepts
        improvements."""
        for name in ("zxz_windows", "yy_pairs", "random"):
            op = GROUPS[name]()
            problem, context = problem_of(op), context_of(op.num_qubits)
            plain = cheapest(Plain().generate(problem, context))
            assert cheapest(Search().generate(problem, context))[0] <= plain[0] + 2

    @pytest.mark.parametrize("variants", [[], ["star"], ["no-such-variant"]])
    def test_invalid_variants(self, variants):
        with pytest.raises(ValueError, match="variants"):
            Search(variants=variants)
