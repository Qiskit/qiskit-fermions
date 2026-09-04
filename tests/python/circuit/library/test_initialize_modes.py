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

"""Mode initialization tests."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import InitializeModes


def _test_initialize_modes(occupation):
    num_modes = len(occupation)
    init = InitializeModes(occupation)
    circ = FermionicCircuit(num_modes)
    circ.append(init, circ.modes)

    ops = circ.count_ops()
    assert ops == {"InitializeModes": 1}


def test_initialize_modes_gate(subtests):
    with subtests.test("from list"):
        _test_initialize_modes([True, False, True, False])

    with subtests.test("from tuple"):
        _test_initialize_modes((True, False, True, False))

    with subtests.test("from np.ndarray"):
        _test_initialize_modes(np.asarray([True, False, True, False], dtype=bool))


def test_apply_unitary_rejects_out_of_range_mode():
    """A gate placed onto a mode outside ``[0, num_modes)`` is rejected before touching ``vec``.

    Uses a raw ``_apply_unitary_placed_`` call with an out-of-range global mode index; a determinant
    vector is not needed since the range check precedes any amplitude inspection.
    """
    norb = 3
    nelec = 2  # spinless -> num_modes == norb == 3
    vec = np.zeros(comb(norb, nelec), dtype=complex)
    # place the single local mode onto global mode 3, which is >= num_modes == 3
    with pytest.raises(ValueError, match="places a mode outside the range"):
        InitializeModes([True])._apply_unitary_placed_(vec, norb, nelec, False, [norb])


def test_apply_unitary_rejects_wrong_length_vec_spinless():
    """A spinless vector whose length disagrees with ``C(norb, nelec)`` is rejected."""
    pytest.importorskip("ffsim")  # the subspace check enumerates determinants via ffsim

    norb = 4
    nelec = 2  # sector dimension is C(4, 2) == 6
    occupation = [True, True, False, False]  # a satisfiable 2-electron occupation
    wrong = np.zeros(5, dtype=complex)
    with pytest.raises(ValueError, match="does not match the dimension"):
        InitializeModes(occupation)._apply_unitary_(wrong, norb, nelec, copy=True)


def test_apply_unitary_rejects_amplitude_outside_alpha_subspace():
    """A spinful state with amplitude outside the alpha-sector subspace is rejected.

    An all-ones vector necessarily carries weight on alpha determinants that leave orbital 0 empty,
    so an alpha-only gate fixing orbital 0 occupied must reject it (the alpha-axis branch).
    """
    pytest.importorskip("ffsim")  # the subspace check enumerates determinants via ffsim

    norb = 3
    nelec = (2, 1)
    dim = comb(norb, nelec[0]) * comb(norb, nelec[1])
    bad = np.ones(dim, dtype=complex)
    # a partial alpha gate: orbital 0 occupied, orbitals 1 and 2 left free (beta sector untouched)
    with pytest.raises(ValueError, match="outside the alpha-sector subspace"):
        InitializeModes([True])._apply_unitary_placed_(bad, norb, nelec, False, [0])
