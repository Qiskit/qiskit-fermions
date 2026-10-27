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

"""Tests for the fermionic synthesis of an OrbitalRotation gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, OrbitalRotation
from qiskit_fermions.circuit.library.synthesis import (
    FermionicGivensDecomposition,
)

from ...utils import random_unitary

ffsim = pytest.importorskip("ffsim")


def _factors(circ: FermionicCircuit):
    return [inst.operation for inst in circ._inner.data]


def test_default_synthesis_is_the_givens_decomposition():
    gate = OrbitalRotation(random_unitary(3, seed=1))
    assert isinstance(gate.synthesis, FermionicGivensDecomposition)


def test_synthesis_is_keyword_only_and_read_only():
    synthesis = FermionicGivensDecomposition(atol=1e-9)
    gate = OrbitalRotation(random_unitary(3, seed=1), synthesis=synthesis)
    assert gate.synthesis is synthesis
    with pytest.raises(AttributeError):
        gate.synthesis = FermionicGivensDecomposition()  # type: ignore[misc]
    with pytest.raises(TypeError):
        OrbitalRotation(random_unitary(3, seed=1), synthesis)  # type: ignore[misc]


@pytest.mark.parametrize("num_modes", [2, 3, 4, 5])
def test_factors_are_atomic_single_group_evolutions(num_modes):
    gate = OrbitalRotation(random_unitary(num_modes, seed=num_modes))

    factors = _factors(gate.synthesis.synthesize(gate))

    # the mesh has n (n - 1) / 2 rotations between adjacent modes, plus one phase per mode
    assert len(factors) == num_modes * (num_modes - 1) // 2 + num_modes
    for factor in factors:
        assert isinstance(factor, Evolution)
        assert factor.atomic
        assert factor.params[0] == 1.0
        assert set(factor.operator.groups) == {0}
    assert sorted({factor.num_modes for factor in factors}) == [1, 2]


def test_rotations_come_before_phases():
    gate = OrbitalRotation(random_unitary(4, seed=4))
    sizes = [factor.num_modes for factor in _factors(gate.synthesis.synthesize(gate))]
    assert sizes == sorted(sizes, reverse=True)


def test_identity_rotation_has_no_factors():
    gate = OrbitalRotation(np.eye(4))
    assert _factors(gate.synthesis.synthesize(gate)) == []


def test_phase_only_rotation_has_no_two_mode_factors():
    gate = OrbitalRotation(np.diag(np.exp(1j * np.array([0.3, -1.1, 0.0, 2.0]))))

    factors = _factors(gate.synthesis.synthesize(gate))

    # the mode without a phase is dropped, and nothing mixes the modes
    assert [factor.num_modes for factor in factors] == [1, 1, 1]


def test_generators_are_exactly_hermitian():
    """``logm`` is anti-Hermitian only to round-off; the generator must not inherit that."""
    gate = OrbitalRotation(random_unitary(5, seed=11))
    for factor in _factors(gate.synthesis.synthesize(gate)):
        coeffs = {tuple(term): coeff for term, coeff in factor.operator.iter_terms()}
        for term, coeff in coeffs.items():
            adjoint = tuple((not is_cre, idx) for is_cre, idx in reversed(term))
            assert coeffs[adjoint] == np.conj(coeff)


def test_tolerance_drops_negligible_factors():
    near_identity = np.eye(3, dtype=complex)
    near_identity[0, 1], near_identity[1, 0] = 1e-9, -1e-9
    gate = OrbitalRotation(near_identity)

    assert len(_factors(FermionicGivensDecomposition(atol=1e-6).synthesize(gate))) == 0
    assert len(_factors(FermionicGivensDecomposition(atol=1e-12).synthesize(gate))) > 0


@pytest.mark.parametrize(("norb", "nelec"), [(2, 1), (3, 1), (3, 2), (4, 2), (5, 3), (6, 3)])
def test_factors_multiply_to_the_rotation(norb, nelec):
    """The product of the factors is the full rotation, with no truncation to speak of."""
    unitary = random_unitary(norb, seed=norb + nelec)
    circ = FermionicCircuit(norb)
    circ.append(OrbitalRotation(unitary), circ.modes)

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=1)
    expected = ffsim.apply_orbital_rotation(vec, unitary, norb=norb, nelec=nelec)
    decomposed = ffsim.apply_unitary(vec, circ.decompose(), norb=norb, nelec=nelec)

    np.testing.assert_allclose(decomposed, expected, atol=1e-10)


def test_two_successive_rotations_keep_their_order():
    """Factors of one substituted rotation must not interleave with the next one's."""
    norb, nelec = 3, 2
    u1, u2 = random_unitary(norb, seed=1), random_unitary(norb, seed=2)
    circ = FermionicCircuit(norb)
    circ.append(OrbitalRotation(u1), circ.modes)
    circ.append(OrbitalRotation(u2), circ.modes)

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=2)
    expected = ffsim.apply_orbital_rotation(
        ffsim.apply_orbital_rotation(vec, u1, norb=norb, nelec=nelec), u2, norb=norb, nelec=nelec
    )
    decomposed = ffsim.apply_unitary(vec, circ.decompose(), norb=norb, nelec=nelec)

    np.testing.assert_allclose(decomposed, expected, atol=1e-10)


def test_custom_synthesis_is_used_by_decompose():
    class Marker:
        calls = 0

        def synthesize(self, gate):
            Marker.calls += 1
            return FermionicCircuit(gate.num_modes)

    circ = FermionicCircuit(2)
    circ.append(OrbitalRotation(random_unitary(2, seed=1), synthesis=Marker()), circ.modes)
    circ.decompose()

    assert Marker.calls == 1


def test_inverse_is_the_adjoint_and_keeps_the_synthesis():
    synthesis = FermionicGivensDecomposition(atol=1e-9)
    unitary = random_unitary(3, seed=5)
    gate = OrbitalRotation(unitary, synthesis=synthesis)

    inverse = gate.inverse()

    assert isinstance(inverse, OrbitalRotation)
    np.testing.assert_allclose(inverse.rotation_unitary, unitary.conj().T)
    assert inverse.synthesis is synthesis


def test_equality_and_hash_follow_the_unitary():
    unitary = random_unitary(3, seed=5)
    a, b = OrbitalRotation(unitary), OrbitalRotation(unitary.copy())
    c = OrbitalRotation(random_unitary(3, seed=6))

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != "OrbitalRotation"
    assert len({a, b, c}) == 2


def test_simulation_does_not_go_through_the_synthesis():
    class Boom:
        def synthesize(self, gate):
            raise AssertionError("the simulation path must apply the whole unitary")

    norb, nelec = 3, 2
    unitary = random_unitary(norb, seed=3)
    circ = FermionicCircuit(norb)
    circ.append(OrbitalRotation(unitary, synthesis=Boom()), circ.modes)

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=3)
    out = ffsim.apply_unitary(vec, circ, norb=norb, nelec=nelec)

    np.testing.assert_allclose(
        out, ffsim.apply_orbital_rotation(vec, unitary, norb=norb, nelec=nelec), atol=1e-12
    )
