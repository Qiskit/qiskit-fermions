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

"""Tests for ordering the terms of an operator by group index."""

from __future__ import annotations

import pytest
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
)
from qiskit_fermions.operators.terms.ordering import group_order, order_terms

# One grouped operator per supported type, so that the generic core routine is exercised through
# every code path it takes rather than only on ``FermionOperator``.
GROUPED_OPERATORS = [
    FermionOperator(
        [1.0, 1.0, 1.0], [True, False, True, False, True, False], [0, 1, 2, 3, 4, 5], [0, 2, 4, 6]
    ),
    MajoranaOperator([1.0, 1.0, 1.0], [0, 1, 2, 3, 4, 5], [0, 2, 4, 6]),
    EdgeVertexOperator([1.0, 1.0, 1.0], [0, 2, 4], [1, 3, 5], [0, 1, 2, 3]),
    TransferVertexOperator([1.0, 1.0, 1.0], [0, 2, 4], [1, 3, 5], [0, 1, 2, 3]),
]


def test_group_order_makes_groups_contiguous():
    op = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 2.0,
            ((True, 2), (False, 3)): 3.0,
        }
    )
    # Interleaved tags, the shape ``group_terms_by_electronic_structure`` produces: group 1's two
    # terms are separated by group 0's.
    op.groups = [1, 0, 1]

    ordered = group_order(op)

    # A new operator is returned; the input is untouched.
    assert ordered is not op
    assert op.groups == [1, 0, 1]
    assert op.equiv(ordered)
    # Each group is now one contiguous run and the indices are non-decreasing.
    assert ordered.groups == [0, 1, 1]


def test_group_order_preserves_the_term_group_association():
    op = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 2.0,
            ((True, 2), (False, 3)): 3.0,
        }
    )
    # ``from_dict`` does not preserve dict order, so tag by stored position and compare the
    # term<->tag association rather than the positions.
    op.groups = [2, 0, 1]
    before = {tuple(term): group for term, _, group in op.iter_terms_with_groups()}

    ordered = group_order(op)

    assert {tuple(term): group for term, _, group in ordered.iter_terms_with_groups()} == before


def test_group_order_is_stable_within_a_group():
    op = FermionOperator(
        [1.0, 2.0, 3.0],
        [True, False, True, False, True, False],
        [0, 1, 2, 3, 4, 5],
        [0, 2, 4, 6],
    )
    # The first two terms share a group, so nothing may reorder them relative to one another.
    op.groups = [0, 0, 1]

    ordered = group_order(op)

    assert ordered.groups == [0, 0, 1]
    # The group-0 terms keep their stored order, carrying their own coefficients. A non-stable sort
    # would be free to swap them.
    assert list(ordered.iter_terms()) == list(op.iter_terms())


def test_group_order_matches_order_terms():
    # ``order_terms`` keyed on the trailing group element already produces the correct group-ordered
    # result, just by round-tripping every term through Python. It is therefore the behavioral
    # oracle for the native implementation.
    op = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 2.0,
            ((True, 2), (False, 3)): 3.0,
            ((True, 3), (False, 2)): 4.0,
            ((True, 0), (False, 3)): 5.0,
        }
    )
    op.groups = [2, 0, 2, 1, 0]

    native = group_order(op)
    oracle = order_terms(op, key=lambda term: term[-1])

    assert list(native.iter_terms_with_groups()) == list(oracle.iter_terms_with_groups())


def test_group_order_ungrouped_returns_unchanged_copy():
    op = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 2.0})
    assert not op.has_groups()

    ordered = group_order(op)

    # Nothing to order by, so the operator comes back untouched rather than raising; this keeps the
    # function composable in a pipeline.
    assert ordered is not op
    assert ordered.groups is None
    assert list(ordered.iter_terms()) == list(op.iter_terms())


def test_group_order_already_ordered_is_a_no_op():
    op = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 2.0})
    op.groups = [0, 1]

    ordered = group_order(op)

    assert ordered.groups == [0, 1]
    assert list(ordered.iter_terms()) == list(op.iter_terms())


@pytest.mark.parametrize("op", GROUPED_OPERATORS, ids=lambda op: type(op).__name__)
def test_every_operator_type_is_supported(op):
    op.groups = [2, 0, 1]

    ordered = group_order(op)

    assert isinstance(ordered, type(op))
    assert op.equiv(ordered)
    assert ordered.groups == [0, 1, 2]


@pytest.mark.parametrize("op", GROUPED_OPERATORS, ids=lambda op: type(op).__name__)
def test_split_out_groups_agrees_before_and_after_ordering(op):
    # ``split_out_groups`` binary-searches the group boundaries when the group indices are sorted and
    # scans every term when they are not. Both must return the same operators. The request shapes
    # below cover everything the method documents: a single index, reversed and non-exhaustive
    # selection, a duplicated index (returned once per occurrence), an empty request, and an index no
    # term carries.
    op.groups = [1, 0, 1]
    ordered = group_order(op)

    for request in ([0], [1], [1, 0], [1, 1], [], [7], [0, 7, 1]):
        from_scan = op.split_out_groups(group_indices=request)
        from_search = ordered.split_out_groups(group_indices=request)

        assert len(from_scan) == len(request)
        assert len(from_search) == len(request)
        for scanned, searched in zip(from_scan, from_search, strict=True):
            assert scanned.equiv(searched), f"paths disagree for {request}"
            # Neither path tags the results with group indices.
            assert scanned.groups is None
            assert searched.groups is None


def test_group_order_rejects_unsupported_type():
    with pytest.raises(TypeError, match="expects a fermionic, Majorana, edge-vertex or transfer"):
        group_order("not an operator")
