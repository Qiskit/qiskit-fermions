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

"""Tests for applying a FermionicCircuit to an ffsim state vector (SupportsApplyUnitary)."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.fermionic_gate import FermionicGate
from qiskit_fermions.circuit.library import InitializeModes

ffsim = pytest.importorskip("ffsim")


class _PlainProtocolGate(FermionicGate):
    """A gate implementing only ffsim's plain ``_apply_unitary_`` (no ``_apply_unitary_placed_``).

    Stands in for any third-party gate that supports only the base protocol. It scales the vector by
    a fixed factor so that whether (and how) it was applied is observable in the returned state,
    without relying on gate-object identity (a circuit copies the instructions it stores).
    """

    SCALE = 2.0

    def __init__(self, num_modes: int = 2):
        super().__init__("plain", num_modes)

    def _apply_unitary_(self, vec, norb, nelec, copy):
        return self.SCALE * vec


def test_apply_unitary_raises_when_instruction_lacks_protocol():
    """A bare FermionicGate cannot be applied to a state vector and raises a clear error.

    A bare :class:`.FermionicGate` (a type marker) does not implement ``_apply_unitary_placed_``. It
    does inherit the base ``_apply_unitary_`` -- the identity-placement delegator -- but that method
    refuses, with a ``NotImplementedError`` naming the offending gate, rather than failing obscurely
    when it finds nothing to delegate to.
    """
    norb = 2
    nelec = (1, 1)

    circ = FermionicCircuit(2 * norb)
    circ.append(FermionicGate("dummy", 2), [circ.modes[0], circ.modes[1]])

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    with pytest.raises(NotImplementedError, match="does not implement '_apply_unitary_placed_'"):
        circ._apply_unitary_(vec0, norb, nelec, copy=True)


def test_bare_fermionic_gate_apply_unitary_raises_directly():
    """Calling the inherited ``_apply_unitary_`` on a bare gate refuses with a clear error.

    The base :meth:`.FermionicGate._apply_unitary_` is the identity-placement delegator shared by all
    concrete gates; on a bare gate (no ``_apply_unitary_placed_`` to delegate to) it must raise a
    ``NotImplementedError`` naming the gate rather than an obscure ``AttributeError``.
    """
    norb = 2
    vec0 = ffsim.slater_determinant(norb, ([0], [0]))
    with pytest.raises(NotImplementedError, match="'FermionicGate' does not implement"):
        FermionicGate("dummy", 2 * norb)._apply_unitary_(vec0, norb, (1, 1), copy=True)


def test_apply_unitary_raises_when_instruction_declines():
    """A circuit instruction returning NotImplemented raises ValueError."""

    class _DecliningGate(FermionicGate):
        """A gate that implements the protocol but declines to act."""

        def __init__(self):
            super().__init__("declines", 2)

        def _apply_unitary_(self, vec, norb, nelec, copy):
            return NotImplemented

    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    circ.append(_DecliningGate(), [circ.modes[0], circ.modes[1]])

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    with pytest.raises(ValueError, match="declined to apply"):
        circ._apply_unitary_(vec0, norb, nelec, copy=True)


def test_apply_unitary_rejects_plain_protocol_gate_on_non_identity_placement():
    """A plain-``_apply_unitary_`` gate placed on a non-identity subset raises rather than misapplies.

    ffsim's base protocol has no mode argument, so the gate acts on modes ``0..k`` of the vector and
    cannot honor a subset placement. Placing it on ``[1, 2]`` of a larger register would silently act
    on the wrong modes, so the walk rejects it instead.
    """
    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    circ.append(_PlainProtocolGate(2), [circ.modes[1], circ.modes[2]])  # non-identity placement

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    with pytest.raises(ValueError, match="no mode-placement argument"):
        circ._apply_unitary_(vec0, norb, nelec, copy=True)


def test_apply_unitary_empty_circuit_with_none_vec_raises():
    """An empty circuit applied to ``vec=None`` raises rather than returning ``None``.

    With no instruction to seed a state and no incoming vector, there is nothing to return; the
    protocol requires an array, so this is rejected instead of silently yielding ``None``.
    """
    circ = FermionicCircuit(4)
    with pytest.raises(ValueError, match="empty circuit"):
        circ._apply_unitary_(None, 2, (1, 1), copy=True)


def test_apply_unitary_empty_circuit_with_vec_returns_it():
    """An empty circuit applied to a real vector returns that vector unchanged (identity)."""
    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    result = circ._apply_unitary_(vec0, norb, nelec, copy=True)
    np.testing.assert_array_equal(result, vec0)


def test_apply_unitary_seeds_from_none_vec_via_producer_first_instruction():
    """A circuit whose first instruction is a state producer seeds from a None vec with copy=True.

    Regression: the DAG walk must let a producer (:class:`.InitializeModes`) seed the state from a
    ``None`` incoming vector without tripping the ``copy`` handling.
    """
    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes([True, False, True, False]), circ.modes)

    result = circ._apply_unitary_(None, norb, nelec, copy=True)
    expected = ffsim.slater_determinant(norb, ([0], [0]))
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_apply_unitary_normalizes_numpy_int_nelec_in_dag_walk():
    """A numpy-integer spinless nelec is normalized inside the circuit's DAG walk, matching ``int``.

    The DAG walk calls each instruction's ``_apply_unitary_placed_`` directly, bypassing the
    gate-level ``FermionicGate._apply_unitary_`` choke-point that normalizes ``nelec``. It therefore
    normalizes the ``nelec`` itself (:meth:`.FermionicCircuit._apply_unitary_placed_`); without that a
    ``np.int64`` would reach the instructions un-normalized and be misclassified as spinful. The
    ``np.int64`` result must match the plain-``int`` spinless path.
    """
    norb = 5
    circ = FermionicCircuit(norb)
    circ.append(InitializeModes([True, False, True, False, False]), circ.modes)

    expected = circ._apply_unitary_(None, norb, 2, copy=True)
    result = circ._apply_unitary_(None, norb, np.int64(2), copy=True)
    np.testing.assert_array_equal(result, expected)


def test_apply_unitary_transform_first_gate_with_none_vec_raises_clean_error():
    """A transform-only first instruction fed ``vec=None`` raises a clear ValueError, not an opaque one.

    A ``None`` incoming vector is only meaningful for a state-*producing* first instruction. A
    transform-only gate has nothing to act on and, left unguarded, fails deep inside its numerics
    with an opaque ``AttributeError`` on the ``None``. The walk must instead raise a ``ValueError``
    naming the offending instruction, as the ``_apply_unitary_`` docstring promises.
    """

    class _TransformOnlyGate(FermionicGate):
        """A transform-only gate that touches ``vec.shape`` -- the exact failure mode of the real
        transform gates (Evolution/OrbitalRotation), which raise ``AttributeError`` on a None vec."""

        def __init__(self, num_modes):
            super().__init__("transform", num_modes)

        def _apply_unitary_(self, vec, norb, nelec, copy):
            return np.zeros(vec.shape, dtype=complex)  # AttributeError when vec is None

    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    circ.append(_TransformOnlyGate(2 * norb), circ.modes)  # identity placement, transform-only

    with pytest.raises(ValueError, match="cannot seed a state from a None vector"):
        circ._apply_unitary_(None, norb, nelec, copy=True)


def test_apply_unitary_accepts_plain_protocol_gate_on_identity_placement():
    """A plain-``_apply_unitary_`` gate on the identity placement ``[0, 1, ...]`` is applied as-is.

    The gate scales the vector by a known factor, so the observable output confirms the fallback
    path ran the gate (rather than being wrongly rejected).
    """
    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    circ.append(_PlainProtocolGate(2 * norb), circ.modes)  # identity placement [0, 1, 2, 3]

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    result = circ._apply_unitary_(vec0, norb, nelec, copy=True)
    np.testing.assert_array_equal(result, _PlainProtocolGate.SCALE * vec0)
