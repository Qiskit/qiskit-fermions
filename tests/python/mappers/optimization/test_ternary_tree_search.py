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

"""Tests for the ternary-tree search algorithms."""

from __future__ import annotations

import pytest
from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
from qiskit_fermions.mappers.library import majorana_operator
from qiskit_fermions.mappers.optimization import hatt_tree
from qiskit_fermions.operators import EdgeVertexOperator, FermionOperator, MajoranaOperator


def hopping_hamiltonian(num_modes: int) -> FermionOperator:
    """Returns a nearest-neighbor hopping Hamiltonian on ``num_modes`` modes."""
    terms = {}
    for mode in range(num_modes - 1):
        terms[((True, mode), (False, mode + 1))] = 1.0
        terms[((True, mode + 1), (False, mode))] = 1.0
    return FermionOperator.from_dict(terms)


class TestHatt:
    def test_accepts_any_supports_majorana_operator(self):
        # The wrapper is not limited to fermions: an edge-vertex operator implements the same
        # protocol, and must give the tree of its Majorana image.
        hamiltonian = EdgeVertexOperator.from_dict({((0, 1),): 1.0, ((1, 2),): 0.5, ((0, 0),): 2.0})
        assert hatt_tree(hamiltonian, 3) == hatt_tree(majorana_operator(hamiltonian), 3)

    def test_num_modes_inferred(self):
        # Inferred from the operator's Majorana support, which for a 4-mode hopping term is
        # gamma_0 .. gamma_7.
        tree = hatt_tree(hopping_hamiltonian(4))
        assert tree.num_nodes == 4

    def test_num_modes_may_exceed_the_support(self):
        # An encoding may deliberately cover modes the Hamiltonian does not touch.
        tree = hatt_tree(hopping_hamiltonian(3), 6)
        assert tree.num_nodes == 6

    def test_ties_go_to_the_lowest_indices(self):
        # With no terms, the tie-break alone builds the tree. The first node takes legs 0, 1 and 2
        # and hosts mode 0; the second pairs it with leg 3 and hosts mode 1, so mode 1 is the root
        # and mode 0 hangs off its X link.
        tree = hatt_tree(FermionOperator.zero(), 2)
        assert tree == TernaryTree([(1, "X"), None])

    def test_weighted_reaches_the_search(self):
        """Three negligible terms and one dominant one: counting and weighting disagree."""
        hamiltonian = MajoranaOperator.from_dict(
            {(0, 3): 1e-3, (1, 2): 1e-3, (2, 5): 1e-3, (0, 4): 1.0}
        )
        assert hatt_tree(hamiltonian, 3) != hatt_tree(hamiltonian, 3, weighted=True)

    def test_beats_the_jordan_wigner_chain_on_total_weight(self):
        # All-to-all hopping is where a chain is at its worst: a hopping between distant modes
        # carries the whole Z string between them. Not a claim that HATT always wins (it is greedy),
        # but here it must.
        num_modes = 6
        terms = {}
        for left in range(num_modes):
            for right in range(left + 1, num_modes):
                terms[((True, left), (False, right))] = 1.0
                terms[((True, right), (False, left))] = 1.0
        hamiltonian = FermionOperator.from_dict(terms)
        hatt = total_pauli_weight(hamiltonian, hatt_tree(hamiltonian, num_modes))
        chain = total_pauli_weight(hamiltonian, TernaryTree.chain(num_modes, "Z"))
        assert hatt < chain

    def test_rejects_num_modes_zero(self):
        with pytest.raises(ValueError, match="at least one mode"):
            hatt_tree(hopping_hamiltonian(2), 0)

    def test_rejects_a_hamiltonian_wider_than_num_modes(self):
        with pytest.raises(ValueError, match="does not cover"):
            hatt_tree(hopping_hamiltonian(6), 2)

    def test_weighted_is_keyword_only(self):
        with pytest.raises(TypeError):
            hatt_tree(hopping_hamiltonian(2), 2, True)  # type: ignore[misc]

    @pytest.mark.parametrize("hamiltonian", [FermionOperator.zero(), FermionOperator.one()])
    def test_requires_num_modes_for_an_operator_on_no_mode(self, hamiltonian):
        # A constant has a term, but no mode to infer the count from.
        with pytest.raises(ValueError, match="num_modes must be given"):
            hatt_tree(hamiltonian)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_rejects_a_non_finite_coefficient(self, weighted):
        hamiltonian = MajoranaOperator.from_dict({(0, 1): 1.0, (1, 2): float("nan")})
        with pytest.raises(ValueError, match="non-finite coefficient"):
            hatt_tree(hamiltonian, 2, weighted=weighted)


def periodic_fermi_hubbard(rows: int, cols: int) -> FermionOperator:
    """Returns the Fermi-Hubbard model on a periodic ``rows x cols`` lattice.

    Spins are interleaved: site ``s`` hosts mode ``2 * s`` (up) and ``2 * s + 1`` (down). On the
    lattices tested below, this convention reproduces the reference's Jordan-Wigner column, which is
    what makes its HATT column comparable.
    """
    bonds = set()
    for row in range(rows):
        for col in range(cols):
            site = row * cols + col
            for right in (row * cols + (col + 1) % cols, ((row + 1) % rows) * cols + col):
                if right != site:
                    bonds.add(tuple(sorted((site, right))))
    terms = {}
    for left, right in bonds:
        for spin in (0, 1):
            terms[((True, 2 * left + spin), (False, 2 * right + spin))] = -1.0
            terms[((True, 2 * right + spin), (False, 2 * left + spin))] = -1.0
    for site in range(rows * cols):
        terms[
            ((True, 2 * site), (False, 2 * site), (True, 2 * site + 1), (False, 2 * site + 1))
        ] = 4.0
    return FermionOperator.from_dict(terms)


def total_pauli_weight(hamiltonian: FermionOperator, tree: TernaryTree) -> int:
    """Returns the summed Pauli weight of ``hamiltonian`` mapped through ``tree``."""
    from qiskit_fermions.mappers.library import fermion_ternary_tree

    observable = fermion_ternary_tree(hamiltonian, TernaryTreeEncoding(tree), tree.num_nodes)
    # Terms that cancel exactly can survive as floating-point residue, depending on summation order.
    observable = observable.simplify(1e-10)
    return sum(len(observable[i].indices) for i in range(len(observable)))


class TestHattReproducesTheReference:
    """Pauli weights from Table II of the HATT reference (Liu et al., arXiv:2409.02010).

    The rows are lattices on which the Hamiltonian convention above reproduces the reference's
    Jordan-Wigner column; on others, such as 2x3 and 3x4, the reference's Hamiltonian differs and
    so does its HATT column. HATT runs on the normal-ordered, simplified Majorana form, whose summed
    cost is the metric the table reports. With ties broken by the lowest index, the HATT column then
    matches on these rows.
    """

    @pytest.mark.parametrize(
        "rows, cols, jordan_wigner, hatt",
        [(2, 2, 80, 76), (2, 4, 304, 256), (3, 3, 492, 410), (4, 4, 1152, 760)],
    )
    def test_fermi_hubbard(self, rows, cols, jordan_wigner, hatt):
        hamiltonian = periodic_fermi_hubbard(rows, cols)
        num_modes = 2 * rows * cols
        # The Jordan-Wigner column pins the Hamiltonian convention, so that the HATT column tests
        # HATT.
        assert total_pauli_weight(hamiltonian, TernaryTree.chain(num_modes, "Z")) == jordan_wigner
        majorana = majorana_operator(hamiltonian).normal_ordered().simplify(1e-10)
        assert total_pauli_weight(hamiltonian, hatt_tree(majorana, num_modes)) == hatt
