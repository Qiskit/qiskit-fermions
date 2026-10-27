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

"""Tests for the fermionic synthesis of a PrepareSlaterDeterminant gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    Evolution,
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)
from qiskit_fermions.circuit.library.synthesis import (
    FermionicSlaterGivensDecomposition,
)

from ...utils import (
    SLATER_STRUCTURE_CASES,
    draw_multi_block_target,
    givens_2mode,
    multi_block_rotation,
    random_unitary,
)

ffsim = pytest.importorskip("ffsim")


def _factors(circ: FermionicCircuit):
    return [inst.operation for inst in circ._inner.data]


def _reference_modes(circ: FermionicCircuit) -> list[int]:
    reference = _factors(circ)[0]
    assert isinstance(reference, InitializeModes)
    return [int(mode) for mode in np.nonzero(reference.occupation)[0]]


def _prepared_state(circ: FermionicCircuit, norb: int, nelec: int) -> np.ndarray:
    """The state the circuit prepares when applied to the vacuum, in the ``nelec`` sector.

    The circuit holds an ``InitializeModes`` that sets the reference, which simulation only
    validates, so the reference is fed in as the state vector and the rest is applied to it.
    """
    occupied = _reference_modes(circ)
    assert len(occupied) == nelec
    vec = ffsim.slater_determinant(norb, occupied)
    rest = FermionicCircuit(norb)
    for inst in circ._inner.data[1:]:
        rest.append(
            inst.operation, [rest.modes[circ._inner.find_bit(q).index] for q in inst.qubits]
        )
    return ffsim.apply_unitary(vec, rest, norb=norb, nelec=nelec)


def _expected_state(occupation, rotation) -> np.ndarray:
    norb = len(occupation)
    occupied = list(np.nonzero(occupation)[0])
    return ffsim.slater_determinant(norb, occupied, orbital_rotation=rotation)


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)))


def test_default_synthesis_is_the_slater_givens_decomposition():
    gate = PrepareSlaterDeterminant([1, 1, 0], random_unitary(3, seed=1))
    assert isinstance(gate.synthesis, FermionicSlaterGivensDecomposition)


def test_synthesis_is_keyword_only_and_read_only():
    synthesis = FermionicSlaterGivensDecomposition(atol=1e-9)
    gate = PrepareSlaterDeterminant([1, 0], random_unitary(2, seed=1), synthesis=synthesis)
    assert gate.synthesis is synthesis
    with pytest.raises(AttributeError):
        gate.synthesis = FermionicSlaterGivensDecomposition()  # type: ignore[misc]
    with pytest.raises(TypeError):
        PrepareSlaterDeterminant([1, 0], random_unitary(2, seed=1), synthesis)  # type: ignore[misc]


def test_the_definition_of_the_gate_is_not_the_synthesis():
    """``_define`` stays the operator-equivalent ``InitializeModes`` + ``OrbitalRotation``."""
    gate = PrepareSlaterDeterminant([1, 1, 0, 0], random_unitary(4, seed=1))

    names = [inst.operation.name for inst in gate.definition.data]

    assert names == ["InitializeModes", "OrbitalRotation"]


def test_simulation_needs_no_decompose_and_keeps_the_fast_path():
    """A circuit mixing the gates is simulated as it is: no ``decompose()`` is required.

    Every one of the gates applies its whole unitary, so the circuit stays on the fast path of
    ffsim. Decomposing it is optional, and does not change the final state. This pins the answer to
    "do I need ``decompose()`` to simulate a ``PrepareSlaterDeterminant``?", which is no.
    """
    norb, nelec = 4, 2
    occupation = [True, True, False, False]
    circ = FermionicCircuit(norb)
    circ.append(PrepareSlaterDeterminant(occupation, random_unitary(norb, seed=1)), circ.modes)
    circ.append(OrbitalRotation(random_unitary(norb, seed=2)), circ.modes)
    circ.append(OrbitalRotation(random_unitary(norb, seed=3)), circ.modes)

    reference = ffsim.slater_determinant(norb, [0, 1])
    direct = ffsim.apply_unitary(reference, circ, norb=norb, nelec=nelec)

    # the circuit is simulated gate by gate, with nothing broken down
    assert dict(circ.count_ops()) == {"PrepareSlaterDeterminant": 1, "OrbitalRotation": 2}

    # decomposing only the Slater gate gives the same state, and keeps the rotations whole
    partial = circ.decompose("PrepareSlaterDeterminant")
    assert dict(partial.count_ops())["OrbitalRotation"] == 3
    np.testing.assert_allclose(
        ffsim.apply_unitary(reference, partial, norb=norb, nelec=nelec), direct, atol=1e-10
    )


def test_a_custom_synthesis_is_used_by_the_synthesis_not_the_definition():
    class Marker:
        calls = 0

        def synthesize(self, gate):
            Marker.calls += 1
            return FermionicCircuit(gate.num_modes)

    gate = PrepareSlaterDeterminant([1, 0], random_unitary(2, seed=1), synthesis=Marker())
    assert gate.definition is not None
    assert Marker.calls == 0

    gate.synthesis.synthesize(gate)
    assert Marker.calls == 1


def test_reference_comes_first_and_factors_are_atomic_single_group_evolutions():
    num_modes, nocc = 6, 3
    occupation = [i < nocc for i in range(num_modes)]
    gate = PrepareSlaterDeterminant(occupation, random_unitary(num_modes, seed=1))

    factors = _factors(gate.synthesis.synthesize(gate))

    assert isinstance(factors[0], InitializeModes)
    assert int(np.sum(factors[0].occupation)) == nocc
    for factor in factors[1:]:
        assert isinstance(factor, Evolution)
        assert factor.atomic
        assert factor.num_modes == 2
        assert factor.params[0] == 1.0
        assert set(factor.operator.groups) == {0}


def test_reduced_gate_count_and_no_phases():
    """At most ``m(n - m)`` two-mode factors, and no single-mode phase factors at all."""
    num_modes, nocc = 6, 3
    occupation = [i < nocc for i in range(num_modes)]
    gate = PrepareSlaterDeterminant(occupation, random_unitary(num_modes, seed=1))

    factors = _factors(gate.synthesis.synthesize(gate))[1:]

    assert 0 < len(factors) <= nocc * (num_modes - nocc)
    assert {factor.num_modes for factor in factors} == {2}


def test_full_occupation_needs_only_the_reference():
    gate = PrepareSlaterDeterminant([True] * 4, random_unitary(4, seed=2))

    factors = _factors(gate.synthesis.synthesize(gate))

    assert len(factors) == 1
    assert isinstance(factors[0], InitializeModes)
    assert all(factors[0].occupation)


def test_empty_occupation_needs_nothing():
    gate = PrepareSlaterDeterminant([False] * 4, random_unitary(4, seed=2))
    assert _factors(gate.synthesis.synthesize(gate)) == []


@pytest.mark.parametrize(
    ("label", "num_modes", "occupied_columns", "build_rotation"),
    SLATER_STRUCTURE_CASES,
    ids=[case[0] for case in SLATER_STRUCTURE_CASES],
)
def test_prepared_state_across_occupied_space_structures(
    label, num_modes, occupied_columns, build_rotation
):
    """The synthesis prepares the Slater determinant for every occupied-space structure.

    Covers disjoint and adjacent multi-mode mixing blocks, nested and interleaved windows (which
    must share a block, or a rotation crosses an occupied mode), a dense single block, a pure
    permutation and a basis vector strictly between a mixing pair's endpoints. The state is
    compared exactly (not just mapped to qubits), so it holds for every encoding.
    """
    rotation = build_rotation()
    occupation = [i in occupied_columns for i in range(num_modes)]
    gate = PrepareSlaterDeterminant(occupation, rotation)

    circ = gate.synthesis.synthesize(gate)

    prepared = _prepared_state(circ, num_modes, sum(occupation))
    np.testing.assert_allclose(
        _overlap(prepared, _expected_state(occupation, rotation)), 1.0, atol=1e-8
    )


def test_prepared_state_randomized_sweep():
    """A seeded sweep of multi-block targets, some with basis vectors between a cluster's modes."""
    rng = np.random.default_rng(20260724)
    trials = multi_block = 0
    for _ in range(300):
        drawn = draw_multi_block_target(rng)
        if drawn is None:
            continue
        num_modes, occupation, rotation, is_multi_block = drawn
        trials += 1
        multi_block += is_multi_block

        gate = PrepareSlaterDeterminant(occupation, rotation)
        prepared = _prepared_state(gate.synthesis.synthesize(gate), num_modes, sum(occupation))
        np.testing.assert_allclose(
            _overlap(prepared, _expected_state(occupation, rotation)), 1.0, atol=1e-8
        )

    # the seed genuinely stresses block-splitting, so the sweep guards it rather than passing on
    # structureless draws
    assert trials > 150
    assert multi_block > 20


def test_a_basis_vector_needs_no_rotation_wherever_it_sits():
    """An occupied orbital that no rotation reaches is seeded on its own mode.

    The rotation mixes only modes 0 and 1, and column 5 is the basis vector of mode 5, with empty
    modes in between. Column 1 mixes modes 0 and 1, which leaves one genuine rotation. The basis
    vector is a block of a single mode, so the leading-reference sweep never has to carry occupation
    across the gap to it.
    """
    occupation = [False, True, False, False, False, True]
    gate = PrepareSlaterDeterminant(occupation, givens_2mode(0.6, 0, 1, 6))

    circ = gate.synthesis.synthesize(gate)

    assert _reference_modes(circ) == [0, 5]
    assert len(_factors(circ)) - 1 == 1


def test_a_pure_permutation_needs_no_rotation():
    gate = PrepareSlaterDeterminant([False, True, False, True], np.eye(4, dtype=complex))

    factors = _factors(gate.synthesis.synthesize(gate))

    assert len(factors) == 1
    assert list(factors[0].occupation) == [False, True, False, True]


def test_block_splitting_drops_the_transport_between_clusters():
    """Two disjoint mixing clusters are synthesized on their own windows, not across the gap."""
    num_modes = 8
    rotation = multi_block_rotation(num_modes, [[0, 1], [5, 6]], seed=31)
    occupation = [i in (0, 3, 5) for i in range(num_modes)]
    gate = PrepareSlaterDeterminant(occupation, rotation)

    factors = _factors(gate.synthesis.synthesize(gate))

    # one rotation per cluster, between the adjacent modes of its own window
    assert len(factors) - 1 == 2


def test_every_rotation_is_between_adjacent_modes():
    """Each rotation is nearest-neighbor, so it is a hop between adjacent qubits under JW."""
    num_modes = 6
    rotation = givens_2mode(0.6, 0, num_modes - 1, num_modes)
    occupation = [i < num_modes - 1 for i in range(num_modes)]
    gate = PrepareSlaterDeterminant(occupation, rotation)

    circ = gate.synthesis.synthesize(gate)

    spans = [
        abs(np.subtract(*(circ._inner.find_bit(q).index for q in inst.qubits)))
        for inst in circ._inner.data[1:]
    ]
    assert spans
    assert set(spans) == {1}
