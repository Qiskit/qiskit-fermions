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

"""Tests for ordering the terms of an operator into a canonical order."""

from __future__ import annotations

import pytest
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
)
from qiskit_fermions.operators.terms.ordering import canonical_order


def test_canonical_order_sorts_fermion_terms():
    # Terms stored out of canonical order: a†_1 a_0 before a†_0 a_1.
    op = FermionOperator.from_dict(
        {
            ((True, 1), (False, 0)): 1.0,
            ((True, 0), (False, 1)): 2.0,
        }
    )

    ordered = canonical_order(op)

    # A new operator is returned; the input is untouched.
    assert ordered is not op
    assert op.equiv(ordered)
    # The mode-0-leading term comes first, carrying its own coefficient.
    assert list(ordered.iter_terms()) == [
        ([(True, 0), (False, 1)], 2 + 0j),
        ([(True, 1), (False, 0)], 1 + 0j),
    ]


def test_canonical_order_sorts_majorana_terms():
    op = MajoranaOperator.from_dict({(2, 3): 1.0, (0, 1): 2.0})

    ordered = canonical_order(op)

    assert op.equiv(ordered)
    assert list(ordered.iter_terms()) == [([0, 1], 2 + 0j), ([2, 3], 1 + 0j)]


def test_canonical_order_preserves_groups():
    op = FermionOperator.from_dict(
        {
            ((True, 1), (False, 0)): 1.0,
            ((True, 0), (False, 1)): 2.0,
        }
    )
    # Give the two terms distinct group tags (term order out of from_dict is arbitrary, so tag by
    # position — the term<->tag association is what must survive reordering, not the positions).
    op.groups = [0, 1]
    before = {tuple(term): group for term, _, group in op.iter_terms_with_groups()}

    ordered = canonical_order(op)

    # Groups are preserved, and each term still carries the same group tag it had before.
    assert ordered.groups is not None
    after = {tuple(term): group for term, _, group in ordered.iter_terms_with_groups()}
    assert after == before


def test_canonical_order_supports_vertex_operators():
    edge = EdgeVertexOperator.from_dict({((1, 1),): 1.0, ((0, 0),): 2.0})
    ordered_edge = canonical_order(edge)
    assert isinstance(ordered_edge, EdgeVertexOperator)
    assert edge.equiv(ordered_edge)
    assert list(ordered_edge.iter_terms()) == [([(0, 0)], 2 + 0j), ([(1, 1)], 1 + 0j)]

    transfer = TransferVertexOperator.from_dict({((1, 1),): 1.0, ((0, 0),): 2.0})
    ordered_transfer = canonical_order(transfer)
    assert isinstance(ordered_transfer, TransferVertexOperator)
    assert transfer.equiv(ordered_transfer)


def test_canonical_order_rejects_unsupported_type():
    with pytest.raises(TypeError):
        canonical_order("not an operator")
