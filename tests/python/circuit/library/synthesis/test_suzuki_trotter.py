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

"""Fermionic Suzuki-Trotter synthesis tests."""

from __future__ import annotations

import pytest
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.circuit.library.synthesis import FermionicSuzukiTrotter
from qiskit_fermions.circuit.library.synthesis.suzuki_trotter import _suzuki_schedule
from qiskit_fermions.operators import FermionOperator

ORDERS = (1, 2, 4, 6)
REPS = (1, 2, 4)
FACTOR_COUNTS = (1, 2, 3, 5)


def _hopping_chain(num_modes: int) -> FermionOperator:
    """Returns a Hermitian hopping chain grouped into even and odd bonds.

    Both groups are Hermitian (each bond is paired with its conjugate) but they do not commute with
    each other, so the product formula is genuinely approximate.
    """
    terms = []
    for site in range(num_modes - 1):
        group = site % 2
        terms.append((((True, site), (False, site + 1)), -1.0, group))
        terms.append((((True, site + 1), (False, site)), -1.0, group))
    return FermionOperator.from_terms_with_groups(terms)


# ----------------------------------------------------------------------------------------------
# the schedule: a pure function, testable without operators or circuits
# ----------------------------------------------------------------------------------------------
def test_schedule_scales_sum_to_one_per_factor(subtests):
    """Each factor must be evolved for the full time in total, or the formula is simply wrong."""
    for order in ORDERS:
        for reps in REPS:
            for num_factors in FACTOR_COUNTS:
                with subtests.test(f"order={order} reps={reps} num_factors={num_factors}"):
                    totals: dict[int, float] = {}
                    for index, scale in _suzuki_schedule(order, num_factors, reps):
                        totals[index] = totals.get(index, 0.0) + scale

                    assert set(totals) == set(range(num_factors))
                    for total in totals.values():
                        assert total == pytest.approx(1.0)


def test_schedule_factor_counts(subtests):
    """A first-order sweep visits each factor once; a second-order one is a palindrome of ``2n-1``."""
    for num_factors in FACTOR_COUNTS:
        with subtests.test(f"num_factors={num_factors}"):
            assert len(_suzuki_schedule(1, num_factors, 1)) == num_factors
            # a single factor short-circuits: the formula is exact at any order (see below)
            expected = 1 if num_factors == 1 else 2 * num_factors - 1
            assert len(_suzuki_schedule(2, num_factors, 1)) == expected


def test_schedule_repeats_scale_with_reps():
    """``reps`` repeats the sweep and divides every scale, leaving the total per factor at one."""
    single = _suzuki_schedule(2, 3, 1)
    doubled = _suzuki_schedule(2, 3, 2)

    assert len(doubled) == 2 * len(single)
    assert [index for index, _ in doubled] == [index for index, _ in single] * 2
    assert [scale for _, scale in doubled] == [scale / 2 for _, scale in single] * 2


def test_second_order_schedule_is_the_symmetric_sandwich():
    schedule = _suzuki_schedule(2, 3, 1)

    assert schedule == [(0, 0.5), (1, 0.5), (2, 1.0), (1, 0.5), (0, 0.5)]


def test_schedule_is_palindromic_for_even_orders(subtests):
    """Symmetry is what cancels the odd-order error terms, so it is worth pinning directly."""
    for order in (2, 4, 6):
        with subtests.test(f"order={order}"):
            schedule = _suzuki_schedule(order, 3, 1)

            assert schedule == list(reversed(schedule))


def test_single_factor_schedule_collapses_to_one_gate(subtests):
    """A lone factor commutes with itself, so splitting it further only adds depth."""
    for order in ORDERS:
        with subtests.test(f"order={order}"):
            assert _suzuki_schedule(order, 1, 1) == [(0, 1.0)]


# ----------------------------------------------------------------------------------------------
# the synthesis method
# ----------------------------------------------------------------------------------------------
def test_rejects_odd_orders_above_one():
    with pytest.raises(ValueError, match="order of 1 or an even order"):
        FermionicSuzukiTrotter(order=3)


def test_rejects_non_positive_reps():
    with pytest.raises(ValueError, match="must be positive"):
        FermionicSuzukiTrotter(reps=0)


def test_emitted_times_follow_the_schedule():
    hamil = _hopping_chain(4)
    time = 1.5
    evo = Evolution(4, hamil, time=time, synthesis=FermionicSuzukiTrotter(order=2))

    times = [instruction.operation.params[0] for instruction in evo.definition.data]

    # two groups: the first is halved and replayed around the second
    assert times == pytest.approx([time / 2, time, time / 2])


def test_emitted_factors_are_atomic():
    """Otherwise the factors would be split again, undoing the formula's own partition."""
    hamil = _hopping_chain(4)
    evo = Evolution(4, hamil, time=1.5, synthesis=FermionicSuzukiTrotter(order=2))

    for instruction in evo.definition.data:
        assert instruction.operation.atomic


def test_ungrouped_operator_is_split_term_by_term():
    """Without groups every term is its own factor, exactly as for Lie-Trotter."""
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 1.0,
            ((True, 2), (False, 3)): 1.0,
        }
    )
    evo = Evolution(4, hamil, time=1.0, synthesis=FermionicSuzukiTrotter(order=2))

    # three terms -> a 2*3-1 = 5 factor palindrome
    assert evo.definition.count_ops() == {"Evolution": 5}


def test_higher_order_emits_more_factors():
    hamil = _hopping_chain(6)
    counts = {}
    for order in (1, 2, 4):
        evo = Evolution(6, hamil, time=1.0, synthesis=FermionicSuzukiTrotter(order=order))
        counts[order] = evo.definition.count_ops()["Evolution"]

    assert counts[1] < counts[2] < counts[4]
