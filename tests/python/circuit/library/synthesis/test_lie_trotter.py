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

"""Fermionic Lie-Trotter synthesis tests."""

from __future__ import annotations

import pickle

import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.circuit.library.synthesis import (
    FermionicEvolutionSynthesis,
    FermionicLieTrotter,
)
from qiskit_fermions.operators import FermionOperator


def _hopping(*, grouped: bool) -> FermionOperator:
    """Returns a 4-mode Hermitian hopping operator, optionally carrying two conjugate-paired groups.

    The grouped variant is built with ``from_terms_with_groups`` rather than by assigning ``groups``
    to a ``from_dict`` operator: the latter reorders its terms, so a positional group assignment would
    not pair each hopping with its own conjugate partner.
    """
    terms = [
        (((True, 0), (False, 1)), 2.0, 0),
        (((True, 1), (False, 0)), 2.0, 0),
        (((True, 2), (False, 3)), -2.0, 1),
        (((True, 3), (False, 2)), -2.0, 1),
    ]
    if grouped:
        return FermionOperator.from_terms_with_groups(terms)
    return FermionOperator.from_terms([(actions, coeff) for actions, coeff, _ in terms])


class _SingleFactor(FermionicEvolutionSynthesis):
    """A minimal synthesis emitting the whole evolution as one full-width factor.

    Defined at module level so that a gate carrying it stays picklable.
    """

    def synthesize(self, gate: Evolution) -> FermionicCircuit:
        definition = FermionicCircuit(gate.num_modes)
        definition.append(
            Evolution(gate.num_modes, gate.operator, time=gate.params[0]), definition.modes
        )
        return definition


def test_default_synthesis_is_lie_trotter():
    evo = Evolution(4, _hopping(grouped=False), time=1.5)

    assert isinstance(evo.synthesis, FermionicLieTrotter)


def test_synthesis_is_read_only():
    """The definition is cached once built, so a later assignment could not take effect."""
    evo = Evolution(4, _hopping(grouped=False), time=1.5)

    with pytest.raises(AttributeError):
        evo.synthesis = FermionicLieTrotter()  # type: ignore[misc]


def test_default_synthesis_is_not_shared_between_instances():
    """The default must be constructed per instance: gate attributes are aliased across ``copy()``."""
    first = Evolution(4, _hopping(grouped=False), time=1.5)
    second = Evolution(4, _hopping(grouped=False), time=1.5)

    assert first.synthesis is not second.synthesis


def test_explicit_lie_trotter_matches_the_default(subtests):
    """Passing the default formula explicitly must not change the emitted definition."""
    for label, grouped, expected in (("ungrouped", False, 4), ("grouped", True, 2)):
        with subtests.test(label):
            hamil = _hopping(grouped=grouped)
            default = Evolution(4, hamil, time=1.5)
            explicit = Evolution(4, hamil, time=1.5, synthesis=FermionicLieTrotter())

            assert default.definition.count_ops() == {"Evolution": expected}
            assert explicit.definition.count_ops() == default.definition.count_ops()


def test_lie_trotter_emits_the_full_time_per_factor():
    """A first-order formula gives every factor the *whole* evolution time."""
    time = 1.5
    evo = Evolution(4, _hopping(grouped=True), time=time)

    times = [instruction.operation.params[0] for instruction in evo.definition.data]

    assert times == [time, time]


def test_lie_trotter_narrows_factors_to_their_support():
    """Each emitted factor acts only on the modes its operator actually touches."""
    evo = Evolution(4, _hopping(grouped=True), time=1.5)

    definition = evo.definition
    for instruction in definition.data:
        assert instruction.operation.num_modes == 2
        placed = [definition.find_bit(mode).index for mode in instruction.qubits]
        assert placed == sorted(placed)

    # the two groups act on modes {0, 1} and {2, 3} respectively
    supports = [
        [definition.find_bit(mode).index for mode in instruction.qubits]
        for instruction in definition.data
    ]
    assert supports == [[0, 1], [2, 3]]


def test_custom_synthesis_is_honored():
    circ = FermionicCircuit(4)
    circ.append(
        Evolution(4, _hopping(grouped=True), time=1.5, synthesis=_SingleFactor()), circ.modes
    )

    # the custom formula emits a single full-width factor rather than one gate per group
    assert circ.decompose().count_ops() == {"Evolution": 1}


def test_synthesis_survives_copy():
    synthesis = _SingleFactor()
    evo = Evolution(4, _hopping(grouped=True), time=1.5, synthesis=synthesis)

    assert evo.copy().synthesis is synthesis


def test_gate_with_custom_synthesis_is_picklable():
    """Regression guard: a circuit holding an ``Evolution`` must stay picklable (see #225)."""
    circ = FermionicCircuit(4)
    circ.append(
        Evolution(4, _hopping(grouped=True), time=1.5, synthesis=_SingleFactor()), circ.modes
    )

    reconstructed = pickle.loads(pickle.dumps(circ))

    assert reconstructed.count_ops() == circ.count_ops()
    assert isinstance(reconstructed._inner.data[0].operation.synthesis, _SingleFactor)


def test_the_interface_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        FermionicEvolutionSynthesis()  # type: ignore[abstract]
