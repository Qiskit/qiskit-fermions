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

"""Numerical tests for the fermionic Suzuki-Trotter synthesis."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.circuit.library.synthesis import FermionicSuzukiTrotter
from qiskit_fermions.operators import FermionOperator

ffsim = pytest.importorskip("ffsim")

NORB = 4
NELEC = (1, 1)
TIME = 1.5


def _hopping_chain() -> FermionOperator:
    """Returns a hopping chain whose two groups are Hermitian but do not commute with each other."""
    terms = []
    for site in range(NORB - 1):
        group = site % 2
        terms.append((((True, site), (False, site + 1)), -1.0, group))
        terms.append((((True, site + 1), (False, site)), -1.0, group))
    return FermionOperator.from_terms_with_groups(terms)


def _errors(operator: FermionOperator, order: int, rep_counts: tuple[int, ...]) -> list[float]:
    """Returns the error of the synthesized evolution against the exact one, per repetition count."""
    initial = ffsim.slater_determinant(NORB, ([0], [0]))
    # the un-decomposed gate exponentiates the whole operator, i.e. carries no Trotter error
    exact = Evolution(NORB, operator, time=TIME)._apply_unitary_(initial, NORB, NELEC, copy=True)

    errors = []
    for reps in rep_counts:
        circ = FermionicCircuit(NORB)
        circ.append(
            Evolution(
                NORB,
                operator,
                time=TIME,
                synthesis=FermionicSuzukiTrotter(order=order, reps=reps),
            ),
            circ.modes,
        )
        evolved = circ.decompose()._apply_unitary_(initial, NORB, NELEC, copy=True)
        errors.append(float(np.linalg.norm(evolved - exact)))
    return errors


def test_synthesized_evolution_stays_unitary(subtests):
    """Every factor is Hermitian here, so the product must be unitary at any order and depth."""
    operator = _hopping_chain()
    initial = ffsim.slater_determinant(NORB, ([0], [0]))

    for order in (1, 2, 4):
        for reps in (1, 3):
            with subtests.test(f"order={order} reps={reps}"):
                circ = FermionicCircuit(NORB)
                circ.append(
                    Evolution(
                        NORB,
                        operator,
                        time=TIME,
                        synthesis=FermionicSuzukiTrotter(order=order, reps=reps),
                    ),
                    circ.modes,
                )
                evolved = circ.decompose()._apply_unitary_(initial, NORB, NELEC, copy=True)

                assert np.isclose(np.linalg.norm(evolved), 1.0)


def test_error_converges_at_the_expected_order(subtests):
    """The error must fall like ``reps**-order``, which is the whole point of a higher order.

    The ratios are asserted rather than absolute errors, so the test pins the convergence *rate*
    without becoming hostage to floating-point drift.
    """
    operator = _hopping_chain()
    rep_counts = (1, 2, 4, 8)

    for order, expected_ratio in ((1, 2.0), (2, 4.0)):
        with subtests.test(f"order={order}"):
            errors = _errors(operator, order, rep_counts)

            assert errors == sorted(errors, reverse=True), "error must decrease with more reps"
            # the asymptotic ratio is approached from above, so allow generous headroom
            for coarse, fine in itertools.pairwise(errors):
                assert coarse / fine == pytest.approx(expected_ratio, rel=0.25)


def test_second_order_beats_first_order_at_equal_repetitions():
    operator = _hopping_chain()

    first = _errors(operator, 1, (4,))[0]
    second = _errors(operator, 2, (4,))[0]

    assert second < first
