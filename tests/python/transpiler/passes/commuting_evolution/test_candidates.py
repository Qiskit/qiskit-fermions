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

"""Tests that every built-in candidate must pass."""

from __future__ import annotations

import pytest
from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import (
    DEFAULT_CANDIDATES,
    CzLocal,
    Local,
    Majority,
    Plain,
    Search,
    TwinFold,
)

from .groups import GROUPS, plaquettes, same_unitary
from .helpers import TIME, context_of, problem_of

ALL = (
    Plain(),
    Majority(),
    TwinFold(),
    Local(),
    CzLocal(),
    Search(),
)
LAYOUT = {"cz-local"}


@pytest.mark.parametrize("name", sorted(GROUPS))
@pytest.mark.parametrize("candidate", ALL, ids=lambda c: c.name)
def test_every_variant_is_exact(candidate, name):
    """Every circuit a candidate offers equals the matrix exponential, including the global phase.

    The layout candidates are placed on a line, the others not (``search`` runs only for a gate that
    is not placed).
    """
    op = GROUPS[name]()
    context = context_of(op.num_qubits, device=candidate.name in LAYOUT)
    variants = candidate.generate(problem_of(op), context)
    if not variants:
        pytest.skip("the candidate has no circuit for this group")
    for circuit in variants.values():
        assert same_unitary(circuit, op, TIME)


@pytest.mark.parametrize("candidate", ALL, ids=lambda c: c.name)
def test_candidates_are_deterministic(candidate):
    """Two calls on the same problem give the same variants with the same gates."""
    op = GROUPS["zxz_windows"]()
    problem, context = problem_of(op), context_of(op.num_qubits, device=True)
    assert candidate.generate(problem, context) == candidate.generate(problem, context)


def test_names_are_unique():
    names = [c.name for c in ALL]
    assert len(names) == len(set(names))


def test_the_default_candidates_are_all_built_in_ones_with_search_last():
    """Ties go to the earlier candidate, so the most expensive one decides only where it is strictly
    better."""
    assert [c.name for c in DEFAULT_CANDIDATES] == [c.name for c in ALL]
    assert DEFAULT_CANDIDATES[-1].name == "search"


@pytest.mark.parametrize("candidate", ALL, ids=lambda c: c.name)
def test_only_the_layout_candidates_need_connectivity(candidate):
    """Without connectivity the layout candidates offer nothing; the others still do (``plaquettes``
    has twins)."""
    op = plaquettes(2)
    variants = candidate.generate(problem_of(op), context_of(op.num_qubits))
    assert bool(variants) == (candidate.name not in LAYOUT)
