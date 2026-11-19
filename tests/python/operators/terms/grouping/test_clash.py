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

"""Tests for the term clash graph and the commuting grouping derived from it."""

import itertools

import pytest
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    TransferVertexOperator,
)
from qiskit_fermions.operators.terms.grouping import group_terms_by_non_commutation

# NOTE: every operator below is built from the sparse arrays rather than from a dictionary, because
# only the former fixes the term order that the edge and group indices refer to.


def _majorana_pairs(pairs):
    modes = [mode for pair in pairs for mode in pair]
    boundaries = [2 * i for i in range(len(pairs) + 1)]
    return MajoranaOperator([1.0] * len(pairs), modes, boundaries)


def _transfer(pairs):
    return TransferVertexOperator(
        [1.0] * len(pairs),
        [left for left, _ in pairs],
        [right for _, right in pairs],
        list(range(len(pairs) + 1)),
    )


def _edge(pairs):
    return EdgeVertexOperator(
        [1.0] * len(pairs),
        [left for left, _ in pairs],
        [right for _, right in pairs],
        list(range(len(pairs) + 1)),
    )


def _assert_proper_coloring(edges, groups):
    """No clash edge may lie inside a group: that is what the grouping means."""
    for left, right in edges:
        assert groups[left] != groups[right], (
            f"terms {left} and {right} do not commute yet share group {groups[left]}"
        )


def test_edges_are_sorted_and_unique():
    # Terms 0 and 1 share Majorana mode 1; term 2 is disjoint from both.
    op = _majorana_pairs([(0, 1), (1, 2), (3, 4)])
    edges = op.non_commutation_edges()

    assert edges == [(0, 1)]
    assert all(left < right for left, right in edges), "pairs must be reported as (i, j) with i < j"
    assert edges == sorted(set(edges)), "edges must be sorted and free of duplicates"


def test_grouping_is_a_proper_coloring():
    op = _majorana_pairs([(0, 1), (1, 2), (2, 3), (3, 0)])
    edges = op.non_commutation_edges()
    group_terms_by_non_commutation(op)

    assert op.groups is not None
    _assert_proper_coloring(edges, op.groups)
    # Dense indices: no group index is skipped.
    assert sorted(set(op.groups)) == list(range(op.num_groups()))


@pytest.mark.parametrize(
    "op",
    [
        _majorana_pairs([(0, 1), (1, 2), (2, 3)]),
        _edge([(0, 1), (1, 2), (0, 0)]),
        _transfer([(0, 1), (2, 1), (1, 2), (1, 1)]),
        FermionOperator(
            [1.0, 1.0, 1.0],
            [True, False, True, False, True, False],
            [0, 1, 1, 0, 2, 3],
            [0, 2, 4, 6],
        ),
    ],
)
def test_all_operator_types_are_supported(op):
    """Every operator type works, including `FermionOperator` whose terms are sums of monomials."""
    edges = op.non_commutation_edges()
    assert edges, "these fixtures are chosen to contain at least one clash"

    group_terms_by_non_commutation(op)
    _assert_proper_coloring(edges, op.groups)


def test_commuting_operator_needs_one_group():
    op = _majorana_pairs([(0, 1), (2, 3), (4, 5)])

    assert op.non_commutation_edges() == [], "disjoint supports commute"
    group_terms_by_non_commutation(op)
    assert op.num_groups() == 1


def test_head_to_tail_transfer_chain_commutes():
    """A chain of transfers shares a mode yet commutes, just as a flow set permits."""
    chain = _transfer([(0, 1), (1, 2)])
    assert chain.non_commutation_edges() == []

    # Two transfers into the *same* target do anticommute.
    into_one = _transfer([(0, 1), (2, 1)])
    assert into_one.non_commutation_edges() == [(0, 1)]


def test_empty_operator():
    op = MajoranaOperator([], [], [0])
    assert op.non_commutation_edges() == []
    group_terms_by_non_commutation(op)
    assert op.groups == []


def test_groups_replace_any_existing_ones():
    op = _majorana_pairs([(0, 1), (1, 2)])
    op.groups = [0, 0]  # the two terms clash, so this grouping is not a proper coloring
    group_terms_by_non_commutation(op)
    assert op.groups == [0, 1], "the clashing terms must be separated"


def test_result_is_independent_of_the_mode_span():
    """The relation is intrinsic, so padding the register with unused modes changes nothing."""
    near = _majorana_pairs([(0, 1), (1, 2), (3, 4)])
    far = _majorana_pairs([(0, 1), (1, 2), (80, 81)])
    assert near.non_commutation_edges() == far.non_commutation_edges()

    # The cost follows the indices in use, not their magnitude, so even the largest index a `u32` can
    # hold is cheap (a dense register up to it would need gigabytes per term).
    last = 2**32 - 1
    huge = _majorana_pairs([(0, 1), (1, 2), (last - 1, last)])
    assert huge.non_commutation_edges() == near.non_commutation_edges()
    ferm = FermionOperator([1.0, 1.0], [True, False, True, False], [0, last, last, 7], [0, 2, 4])
    assert ferm.non_commutation_edges() == [(0, 1)]


def test_fermion_relation_ignores_coefficient_scale():
    """Commutation of two terms does not depend on their coefficients, however small.

    Each term's Majorana expansion has contributions of a fixed size, set by the term degrees alone,
    so a tolerance on their sums would wrongly declare high-degree terms to commute. Two-body terms
    are where that first bites: their contributions are only ``2**-7``.
    """
    # a+_0 a+_1 a_2 a_3 against a+_3 a_0, which clash, plus a disjoint number operator on mode 4
    actions = [True, True, False, False, True, False, True, False]
    modes = [0, 1, 2, 3, 3, 0, 4, 4]
    boundaries = [0, 4, 6, 8]
    unit = FermionOperator([1.0] * 3, actions, modes, boundaries)
    tiny = FermionOperator([1e-12] * 3, actions, modes, boundaries)
    assert unit.non_commutation_edges() == [(0, 1)]
    assert tiny.non_commutation_edges() == unit.non_commutation_edges()

    group_terms_by_non_commutation(tiny)
    assert tiny.groups[0] != tiny.groups[1]


def test_zero_coefficient_terms_are_isolated():
    """An exactly zero term is the zero operator, so it clashes with nothing."""
    op = MajoranaOperator([0.0, 1.0, 1.0], [0, 1, 2], [0, 1, 2, 3])
    # Three distinct single generators would pairwise anticommute; the zero one drops out.
    assert op.non_commutation_edges() == [(1, 2)]
    group_terms_by_non_commutation(op)
    assert op.num_groups() == 2

    # Near-zero terms still count, which is what simplifying first is for.
    near = MajoranaOperator([1e-17, 1.0, 1.0], [0, 1, 2], [0, 1, 2, 3])
    assert len(near.non_commutation_edges()) == 3
    assert len(near.simplify().non_commutation_edges()) == 1


def test_fermion_degree_bound():
    """A fermionic term above degree 20 is refused rather than expanded into 2**21 monomials."""
    op = FermionOperator([1.0], [True] * 21, list(range(21)), [0, 21])
    with pytest.raises(ValueError, match="degree 21"):
        op.non_commutation_edges()
    with pytest.raises(ValueError, match="degree 21"):
        group_terms_by_non_commutation(op)
    assert op.groups is None, "a failed grouping must not assign groups"

    # A zero coefficient is never expanded, so its degree is no obstacle.
    zero = FermionOperator([0.0, 1.0], [True] * 22, list(range(22)), [0, 21, 22])
    assert zero.non_commutation_edges() == []
    group_terms_by_non_commutation(zero)
    assert zero.groups == [0, 0]


def test_edges_feed_a_graph_library():
    """The edge list is meant to be handed to a graph library; check it round-trips."""
    rx = pytest.importorskip("rustworkx")
    op = _majorana_pairs([(0, 1), (1, 2), (3, 4)])

    graph = rx.PyGraph()
    graph.add_nodes_from(range(len(op)))
    graph.add_edges_from_no_data(op.non_commutation_edges())

    assert graph.num_nodes() == len(op) == 3
    assert sorted(graph.edge_list()) == [(0, 1)]


def test_flow_sets_are_not_commuting_groups():
    """For Majorana bilinears, the grouping produced here is not the flow-set partition.

    A flow set forbids two arrows sharing a source or a target, but permits a head-to-tail chain -
    and two bilinears in such a chain share a generator, so they do not commute. The grouping
    guide's hand-written groups, which resemble line flow sets, therefore contain non-commuting
    pairs, while this routine's grouping never does. As transfers, by contrast, the same groups are
    clash-free.
    """
    edges = [(0, 1), (1, 2), (3, 0), (1, 4), (5, 2), (5, 4), (4, 3)]
    flow_sets = [0, 0, 1, 2, 1, 3, 3]  # verbatim from docs/guides/grouping.rst

    op = _majorana_pairs(edges)
    clashes = op.non_commutation_edges()

    # The head-to-tail chain (0,1) -> (1,2) shares mode 1, so terms 0 and 1 clash ...
    assert (0, 1) in clashes
    # ... yet the guide places them in the same flow set.
    assert flow_sets[0] == flow_sets[1]

    intra_group = [(left, right) for left, right in clashes if flow_sets[left] == flow_sets[right]]
    assert intra_group, (
        "the guide's flow sets are expected to contain non-commuting pairs; if this ever becomes "
        "empty, the two relations have stopped diverging and the docs must be revisited"
    )

    # Our own grouping, by contrast, is a proper coloring by construction.
    group_terms_by_non_commutation(op)
    _assert_proper_coloring(clashes, op.groups)

    # Written as transfers, the very same arrows clash exactly where a flow set forbids them to meet,
    # so the guide's flow sets become a proper coloring.
    transfers = _transfer(edges)
    conflicts = [
        (i, j)
        for i, j in itertools.combinations(range(len(edges)), 2)
        if edges[i][0] == edges[j][0] or edges[i][1] == edges[j][1]
    ]
    assert transfers.non_commutation_edges() == conflicts
    _assert_proper_coloring(conflicts, flow_sets)


def test_square_lattice_partition():
    """Both orientations of every bond of a 3x3 lattice, as transfer operators."""

    def index(row, col):
        return row * 3 + col

    bonds = []
    for row in range(3):
        for col in range(3):
            if col + 1 < 3:
                bonds.append((index(row, col), index(row, col + 1)))
            if row + 1 < 3:
                bonds.append((index(row, col), index(row + 1, col)))
    arrows = [arrow for j, k in bonds for arrow in ((j, k), (k, j))]
    assert len(arrows) == 24, "3x3 has 12 bonds, hence 24 directed arrows"

    op = _transfer(arrows)
    edges = op.non_commutation_edges()
    # The clash count follows from the commutation predicate alone, so it is fixed.
    assert len(edges) == 44

    group_terms_by_non_commutation(op)
    _assert_proper_coloring(edges, op.groups)
    # Greedy coloring gives an upper bound rather than a minimum, so assert a bound here and leave
    # the exact value pinned by the Rust characterization test.
    assert op.num_groups() <= 4


def test_agrees_with_commutator():
    """Ground truth: the edges must agree with the library's own commutator, term by term."""
    from qiskit_fermions.operators.library import commutator

    terms = [
        MajoranaOperator([1.0], list(combo), [0, len(combo)])
        for degree in (1, 2, 3)
        for combo in itertools.combinations(range(6), degree)
    ]
    modes = [mode for term in terms for mode in term.get_modes()]
    boundaries = [0]
    for term in terms:
        boundaries.append(boundaries[-1] + len(term.get_modes()))
    op = MajoranaOperator([1.0] * len(terms), modes, boundaries)

    clashes = set(op.non_commutation_edges())
    for i, j in itertools.combinations(range(len(terms)), 2):
        # `simplify` alone does not reorder generators within a term, so normal-order first: that
        # is what lets `g_a g_b` and `g_b g_a` cancel rather than survive as two terms.
        residual = commutator(terms[i], terms[j]).normal_ordered().simplify(atol=1e-12)
        commutes = all(abs(c) < 1e-12 for c in residual.get_coeffs())
        assert ((i, j) in clashes) is not commutes, (
            f"edges and commutator disagree for terms {i} and {j}"
        )


def test_grouping_rejects_unsupported_type():
    """The grouping is a free function over any operator type, so it must reject anything else."""
    with pytest.raises(TypeError, match="expects a fermionic, Majorana, edge-vertex"):
        group_terms_by_non_commutation("not an operator")
