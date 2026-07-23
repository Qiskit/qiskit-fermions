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

"""Tests for the OrbitalRotation gate."""

from __future__ import annotations

import numpy as np
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import OrbitalRotation
from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

from ...utils import random_unitary


def test_orbital_rotation_coerces_real_dtype():
    """A real-dtype rotation matrix is coerced to complex on construction.

    The synthesis path (``givens_decomposition``) requires a complex matrix; without coercion a
    real-valued unitary reaches the Rust binding and raises an opaque cast error. Constructing from a
    real array must store a complex ``rotation_unitary`` and synthesize without error.
    """
    num_modes = 4
    # a real-valued orthogonal matrix (real special case of a unitary)
    real_rotation = np.linalg.qr(random_unitary(num_modes, seed=3).real)[0]
    assert real_rotation.dtype.kind == "f"

    gate = OrbitalRotation(real_rotation)
    assert gate.rotation_unitary.dtype == np.complex128

    circ = FermionicCircuit(num_modes)
    circ.append(gate, circ.modes)
    # lowering must not raise (previously: "'ndarray' object cannot be cast as 'ndarray'")
    generate_preset_jw_pass_manager().run(circ)
