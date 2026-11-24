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

"""Tests of the local candidate."""

from __future__ import annotations

import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import Local

from .groups import zxz_windows
from .helpers import context_of, problem_of


class TestLocal:
    def test_variants_are_the_parity_shapes(self):
        variants = Local().generate(problem_of(zxz_windows()), context_of(7))
        assert list(variants) == ["tree", "ladder"]
        assert list(
            Local(variants=["ladder"]).generate(problem_of(zxz_windows()), context_of(7))
        ) == ["ladder"]

    def test_ladder_uses_nearest_neighbors_in_order(self):
        """The ladder joins sorted qubits one after another."""
        op = SparsePauliOp.from_sparse_list([("ZZZZ", [0, 1, 2, 3], 0.9)], num_qubits=4)
        ladder = Local(variants=["ladder"]).generate(problem_of(op), context_of(4))["ladder"]
        pairs = set()
        for inst in ladder.decompose(reps=6).data:
            if len(inst.qubits) == 2:
                a, b = (ladder.find_bit(q).index for q in inst.qubits)
                pairs.add((min(a, b), max(a, b)))
        assert pairs <= {(0, 1), (1, 2), (2, 3)}

    @pytest.mark.parametrize("variants", [[], ["no-such-variant"]])
    def test_invalid_variants(self, variants):
        with pytest.raises(ValueError, match="variants"):
            Local(variants=variants)
