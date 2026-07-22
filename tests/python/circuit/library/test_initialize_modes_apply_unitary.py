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

"""Tests for the InitializeModes assert-only validator (SupportsApplyUnitary)."""

from __future__ import annotations

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
def test_initialize_modes_matching_determinant_passes_unchanged(norb, nelec, occupation, ffsim_occ):
    """A full-determinant occupation accepts its own determinant and returns it unchanged.

    The gate is a validator: given the determinant its occupation describes, the amplitude is
    confined to the occupation's subspace, so the check passes and the very same vector is returned
    (identity). Covers the spinful (full both-sector) and spinless conventions across several ranks.
    """
    vec = ffsim.slater_determinant(norb, ffsim_occ)
    result = InitializeModes(occupation)._apply_unitary_(vec, norb, nelec, copy=True)
    np.testing.assert_array_equal(result, vec)


def test_initialize_modes_parallel_spinful_hartree_fock():
    """Two parallel per-sector gates each certify their own axis of a spinful HF state.

    An alpha-only gate fixes the alpha axis (a set of full rows) and a beta-only gate fixes the beta
    axis (full columns); the Hartree-Fock determinant lies in both, so both pass and the state is
    returned unchanged. This is the parallel placement the old producer could not express.
    """
    norb = 3
    nelec = (2, 1)
    vec = ffsim.slater_determinant(norb, ([0, 1], [0]))

    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes([True, True, False]), [circ.modes[i] for i in range(norb)])
    circ.append(InitializeModes([True, False, False]), [circ.modes[norb + i] for i in range(norb)])

    result = circ._apply_unitary_(vec, norb, nelec, copy=True)
    np.testing.assert_allclose(result, vec, atol=1e-12)


def test_initialize_modes_parallel_disjoint_fragments_within_one_sector():
    """Three disjoint single-orbital gates within one spin sector compose (each fixes one row family).

    Each gate fixes one alpha orbital's occupation and leaves the rest free; their intersection pins
    the alpha determinant. A beta gate on the other sector rounds out a full spinful HF state.
    """
    norb = 4
    nelec = (3, 1)
    vec = ffsim.slater_determinant(norb, ([0, 1, 2], [0]))

    circ = FermionicCircuit(2 * norb)
    # three disjoint alpha fragments: orbitals 0, 1, 2 each occupied (orbital 3 left free)
    circ.append(InitializeModes([True]), [circ.modes[0]])
    circ.append(InitializeModes([True]), [circ.modes[1]])
    circ.append(InitializeModes([True]), [circ.modes[2]])
    # the beta sector
    circ.append(
        InitializeModes([True, False, False, False]), [circ.modes[norb + i] for i in range(norb)]
    )

    result = circ._apply_unitary_(vec, norb, nelec, copy=True)
    np.testing.assert_allclose(result, vec, atol=1e-12)


def test_initialize_modes_partial_fragment_accepts_spread_over_free_axis():
    """A partial (single-orbital) gate accepts a state spread across the orbitals it leaves free.

    Fixing only that alpha orbital 0 is occupied, the gate must accept any superposition of
    determinants that all occupy orbital 0 -- the free orbitals may carry genuine multi-address
    amplitude -- and reject a state with weight on a determinant that leaves orbital 0 empty.
    """
    norb = 4
    nelec = 2  # spinless

    circ = FermionicCircuit(norb)
    circ.append(InitializeModes([True]), [circ.modes[0]])  # orbital 0 occupied; 1,2,3 free

    # a genuine superposition, both terms occupying orbital 0
    good = ffsim.slater_determinant(norb, [0, 1]) + ffsim.slater_determinant(norb, [0, 2])
    result = circ._apply_unitary_(good, norb, nelec, copy=True)
    np.testing.assert_allclose(result, good, atol=1e-12)

    # a determinant leaving orbital 0 empty must be rejected
    bad = ffsim.slater_determinant(norb, [1, 2])
    with pytest.raises(ValueError, match="amplitude outside the subspace"):
        circ._apply_unitary_(bad, norb, nelec, copy=True)


def test_initialize_modes_global_phase_and_magnitude_tolerated():
    """The check is on confinement, not equality: a phased / rescaled determinant still passes."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    vec = ffsim.slater_determinant(norb, ([0, 1], [0]))

    scaled = 0.5 * np.exp(1j * 0.9) * vec
    result = InitializeModes(occupation)._apply_unitary_(scaled, norb, nelec, copy=True)
    np.testing.assert_array_equal(result, scaled)


def test_initialize_modes_rejects_partial_straddle():
    """A spinful gate constraining orbitals of *both* sectors without pinning a full determinant."""
    norb = 3
    nelec = (1, 1)
    vec = ffsim.slater_determinant(norb, ([0], [0]))

    # two local modes placed onto alpha orbital 0 and beta orbital 1 -- a partial straddle
    with pytest.raises(ValueError, match="straddles both spin sectors"):
        InitializeModes([True, True])._apply_unitary_placed_(vec, norb, nelec, False, [0, norb + 1])


def test_initialize_modes_rejects_amplitude_outside_subspace():
    """A state whose amplitude falls outside the occupation's subspace is rejected."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]  # alpha {0,1}, beta {0}
    # a determinant with the beta electron on orbital 1 instead of 0
    bad = ffsim.slater_determinant(norb, ([0, 1], [1]))
    with pytest.raises(ValueError, match="outside the beta-sector subspace"):
        InitializeModes(occupation)._apply_unitary_(bad, norb, nelec, copy=False)


def test_initialize_modes_rejects_unsatisfiable_occupation():
    """An occupation forcing more electrons into a sector than nelec allows is rejected.

    With a full-register occupation, forcing three alpha orbitals occupied is impossible for a
    two-alpha sector; the subspace mask is unsatisfiable, surfaced as a ValueError.
    """
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, True, True, False, False]  # 3 alpha occupied, sector has 2
    vec = ffsim.slater_determinant(norb, ([0, 1], [0]))
    with pytest.raises(ValueError):
        InitializeModes(occupation)._apply_unitary_(vec, norb, nelec, copy=False)


def test_initialize_modes_rejects_wrong_length_vec():
    """A vector whose length disagrees with the (norb, nelec) sector dimension is rejected."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    wrong = np.zeros(5, dtype=complex)  # sector dim is C(3,2)*C(3,1) = 9
    with pytest.raises(ValueError, match="does not match the dimension"):
        InitializeModes(occupation)._apply_unitary_(wrong, norb, nelec, copy=True)


def test_initialize_modes_through_circuit_with_full_placement():
    """A single spinful gate placed on the full 2*norb register pins a full determinant and passes."""
    norb = 3
    nelec = (1, 1)

    # local modes map onto global alpha mode 2 and global beta mode (norb + 1): a full straddle only
    # if the whole register is covered, so place a full-register occupation pinning ([2], [1]).
    occupation = [False, False, True, False, True, False]  # alpha orbital 2, beta orbital 1
    vec = ffsim.slater_determinant(norb, ([2], [1]))

    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes(occupation), circ.modes)

    result = circ._apply_unitary_(vec, norb, nelec, copy=True)
    np.testing.assert_array_equal(result, vec)


def test_initialize_modes_end_to_end_seed_then_rotate():
    """Certifying a reference then rotating it matches the external-seed flow."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    rot = random_unitary(norb, seed=5)

    # our flow: InitializeModes certifies the prepared reference, then OrbitalRotation acts on it
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))
    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])
    result = circ._apply_unitary_(vec0, norb, nelec, copy=True)

    # reference: apply the same rotation directly to the same prepared determinant
    expected = ffsim.apply_orbital_rotation(vec0.copy(), (rot, None), norb=norb, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.parametrize(
    "norb, nelec, expected_occ, ffsim_occ",
    [
        (3, (2, 1), [True, True, False, True, False, False], ([0, 1], [0])),
        (4, (1, 1), [True, False, False, False, True, False, False, False], ([0], [0])),
        (5, 2, [True, True, False, False, False], [0, 1]),  # spinless
    ],
)
def test_initialize_modes_from_hartree_fock(norb, nelec, expected_occ, ffsim_occ):
    """``from_hartree_fock`` builds the HF occupation and round-trips through the validator."""
    gate = InitializeModes.from_hartree_fock(norb, nelec)
    assert list(gate.occupation) == expected_occ

    vec = ffsim.slater_determinant(norb, ffsim_occ)
    result = gate._apply_unitary_(vec, norb, nelec, copy=True)
    np.testing.assert_array_equal(result, vec)


def test_initialize_modes_from_hartree_fock_rejects_overfull_sector():
    """``from_hartree_fock`` rejects an electron count exceeding the available orbitals."""
    with pytest.raises(ValueError, match="exceeds the norb"):
        InitializeModes.from_hartree_fock(2, 3)  # spinless: 3 electrons, 2 orbitals
    with pytest.raises(ValueError, match="exceeding the norb"):
        InitializeModes.from_hartree_fock(2, (3, 1))  # spinful: 3 alpha, 2 orbitals
