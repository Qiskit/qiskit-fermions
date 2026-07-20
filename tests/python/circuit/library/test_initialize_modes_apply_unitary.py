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

"""Tests for seeding an InitializeModes gate into an ffsim state vector (SupportsApplyUnitary)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import InitializeModes, OrbitalRotation

from ...utils import random_unitary

ffsim = pytest.importorskip("ffsim")


@pytest.mark.parametrize(
    "norb, nelec, occupation, ffsim_occ",
    [
        # spinful, block-spin occupation: alpha orbitals 0,1 (modes 0,1); beta orbital 0 (mode norb)
        (3, (2, 1), [True, True, False, True, False, False], ([0, 1], [0])),
        # spinless: orbitals 0 and 2 occupied
        (5, 2, [True, False, True, False, False], [0, 2]),
        # spinless, a different placement
        (4, 2, [True, False, False, True], [0, 3]),
        # spinful, higher ranks: alpha modes 1,3,5 | beta modes norb+0, norb+4 (orbitals 0,4)
        (
            6,
            (3, 2),
            [False, True, False, True, False, True, True, False, False, False, True, False],
            ([1, 3, 5], [0, 4]),
        ),
        # spinless, higher rank
        (6, 3, [False, True, False, True, False, True], [1, 3, 5]),
    ],
)
def test_initialize_modes_apply_unitary_seed_matches_ffsim(norb, nelec, occupation, ffsim_occ):
    """Seeding a vec=None state produces the same determinant as ffsim.slater_determinant.

    Covers the spinful and spinless mode conventions across several ranks and placements; the native
    ``slater_determinant_statevector`` seed must match ffsim's determinant bit-for-bit in every case.
    """
    result = InitializeModes(occupation)._apply_unitary_(None, norb, nelec, copy=True)
    expected = ffsim.slater_determinant(norb, ffsim_occ)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_initialize_modes_apply_unitary_agreeing_vec_logs_and_seeds(caplog):
    """A non-None vec that agrees with the occupation's sector is accepted, warned, and overwritten."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    expected = ffsim.slater_determinant(norb, ([0, 1], [0]))

    # a real, correctly sized (but different) vector in the same sector
    rng = np.random.default_rng(0)
    incoming = rng.standard_normal(len(expected)) + 1j * rng.standard_normal(len(expected))
    incoming_before = incoming.copy()

    with caplog.at_level(
        logging.WARNING, logger="qiskit_fermions.circuit.library.initialize_modes"
    ):
        result = InitializeModes(occupation)._apply_unitary_(incoming, norb, nelec, copy=True)

    np.testing.assert_allclose(result, expected, atol=1e-12)
    # the incoming amplitudes are replaced by the determinant, and the input is left untouched
    np.testing.assert_array_equal(incoming, incoming_before)
    # discarding accumulated state is surprising enough to warrant a warning, not just an info log
    assert any(rec.levelname == "WARNING" for rec in caplog.records)
    assert any("discarding the incoming state vector" in rec.message for rec in caplog.records)


def test_initialize_modes_apply_unitary_through_circuit_with_placement():
    """A subset-placed InitializeModes seeds onto its global modes (straddling alpha and beta)."""
    norb = 3
    nelec = (1, 1)

    # a single gate placed on a subset of the 2*norb register: its two local modes map onto global
    # alpha mode 2 and global beta mode (norb + 1), so the seeded determinant is ([2], [1])
    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes([True, True]), [circ.modes[2], circ.modes[norb + 1]])

    result = circ._apply_unitary_(None, norb, nelec, copy=True)
    expected = ffsim.slater_determinant(norb, ([2], [1]))

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_initialize_modes_apply_unitary_end_to_end_full_circuit():
    """Seeding via InitializeModes then rotating matches the external-seed flow."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    rot = random_unitary(norb, seed=5)

    # our flow: InitializeModes seeds the state, then OrbitalRotation acts on it
    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])
    result = circ._apply_unitary_(None, norb, nelec, copy=True)

    # reference: the current external-seed flow (build the determinant, then apply the same rotation)
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))
    expected = ffsim.apply_orbital_rotation(vec0.copy(), (rot, None), norb=norb, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_initialize_modes_apply_unitary_via_ffsim_apply_unitary_with_real_vec():
    """ffsim.apply_unitary (which always passes a real array) drives a circuit starting with the seed."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    rot = random_unitary(norb, seed=5)

    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])

    # a correctly sized (arbitrary) vector in the target sector; InitializeModes overwrites it
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))
    result = ffsim.apply_unitary(vec0, circ, norb=norb, nelec=nelec)

    expected = ffsim.apply_orbital_rotation(vec0.copy(), (rot, None), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_initialize_modes_apply_unitary_rejects_sector_mismatch():
    """An occupation whose electron counts disagree with nelec is rejected."""
    # spinful: occupation sets 3 alpha but nelec asks for 2
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, True, True, False, False]  # 3 alpha, 1 beta
    with pytest.raises(ValueError, match="do not match the requested nelec"):
        InitializeModes(occupation)._apply_unitary_(None, norb, nelec, copy=False)

    # spinless: total count disagrees
    with pytest.raises(ValueError, match="do not match the requested nelec"):
        InitializeModes([True, True, True])._apply_unitary_(None, 3, 2, copy=False)


def test_initialize_modes_apply_unitary_rejects_wrong_length_vec():
    """A non-None vec whose length disagrees with the occupation's sector is rejected."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    wrong = np.zeros(5, dtype=complex)  # sector dim is C(3,2)*C(3,1) = 9
    with pytest.raises(ValueError, match="does not match the dimension"):
        InitializeModes(occupation)._apply_unitary_(wrong, norb, nelec, copy=True)
