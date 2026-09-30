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
from qiskit_fermions.circuit.library import InitializeModes, OrbitalRotation

from ..utils import random_unitary

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


def test_apply_unitary_empty_circuit_with_vec_returns_it():
    """An empty circuit applied to a real vector returns that vector unchanged (identity)."""
    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    result = circ._apply_unitary_(vec0, norb, nelec, copy=True)
    np.testing.assert_array_equal(result, vec0)


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
    # a validator gate that only accepts the spinless interpretation of nelec: without normalizing
    # the np.int64 it would be routed to the spinful path and behave differently.
    circ.append(InitializeModes([True, False, True, False, False]), circ.modes)
    vec = ffsim.slater_determinant(norb, [0, 2])

    expected = circ._apply_unitary_(vec, norb, 2, copy=True)
    result = circ._apply_unitary_(vec, norb, np.int64(2), copy=True)
    np.testing.assert_array_equal(result, expected)


def test_apply_unitary_parallel_seed_then_rotate():
    """Two parallel per-sector InitializeModes gates then an OrbitalRotation, driven with a real vec.

    The seed gates validate the incoming Hartree-Fock reference per spin sector (the placement the old
    producer could not express); a following alpha-sector rotation transforms it. The result must
    match applying the same rotation directly to the reference.
    """
    norb = 3
    nelec = (2, 1)
    rot = random_unitary(norb, seed=7)
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes([True, True, False]), [circ.modes[i] for i in range(norb)])
    circ.append(InitializeModes([True, False, False]), [circ.modes[norb + i] for i in range(norb)])
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])

    result = circ._apply_unitary_(vec0, norb, nelec, copy=True)
    expected = ffsim.apply_orbital_rotation(vec0.copy(), (rot, None), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_apply_unitary_skips_barrier():
    """A barrier is unitarily the identity, so it must not change the resulting state vector.

    Barriers carry no unitary content and are skipped by the DAG walk rather than dispatched through
    ffsim's protocol (which they do not implement). Comparing against the same circuit without any
    barriers proves both that the walk does not reject them and that it does not apply them as
    something other than the identity.
    """
    norb = 3
    nelec = (2, 1)
    rot = random_unitary(norb, seed=11)
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    def build(with_barriers: bool) -> FermionicCircuit:
        circ = FermionicCircuit(2 * norb)
        alpha = [circ.modes[i] for i in range(norb)]
        if with_barriers:
            circ.barrier()
        circ.append(InitializeModes([True, True, False]), alpha)
        circ.append(
            InitializeModes([True, False, False]), [circ.modes[norb + i] for i in range(norb)]
        )
        if with_barriers:
            # both a full-width and a partial barrier, since they take different qargs paths
            circ.barrier()
            circ.barrier(circ.register[0], circ.register[1])
        circ.append(OrbitalRotation(rot), alpha)
        if with_barriers:
            circ.barrier()
        return circ

    assert build(True).count_ops()["barrier"] == 4

    result = build(True)._apply_unitary_(vec0, norb, nelec, copy=True)
    expected = build(False)._apply_unitary_(vec0, norb, nelec, copy=True)
    np.testing.assert_allclose(result, expected, atol=1e-10)


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


def test_public_ffsim_apply_unitary_drives_a_fermionic_circuit():
    """The public ``ffsim.apply_unitary`` entry point drives a FermionicCircuit end to end.

    ``ffsim.apply_unitary`` calls the object's ``_apply_unitary_`` with ``norb``/``nelec``/``copy`` as
    *keyword* arguments (and honors a ``NotImplemented`` return). Driving a real (seed-then-rotate)
    circuit through it -- rather than calling ``circ._apply_unitary_`` positionally as the other tests
    do -- is the one place that guards this public-protocol contract: a reordered signature would pass
    the positional tests yet break actual ffsim usage. One such check suffices for all gates, since
    ffsim's dispatch is gate-agnostic; the per-gate correctness lives in the gate apply-unitary tests.
    """
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    rot = random_unitary(norb, seed=5)

    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])

    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))
    # public entry point: ffsim.apply_unitary invokes circ._apply_unitary_ with keyword args
    result = ffsim.apply_unitary(vec0, circ, norb=norb, nelec=nelec)

    expected = ffsim.apply_orbital_rotation(vec0, (rot, None), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_repeat_apply_unitary_matches_applying_twice():
    """A repeated circuit simulates as applying the original that many times.

    This is the regression lock on the flattening decision: ``repeat`` reuses the gates it already
    holds, so the ffsim path keeps working with no extra protocol code. A wrapping implementation
    would need its own ``_apply_unitary_placed_`` or raise here.
    """
    norb = 3
    nelec = (2, 1)
    rot = random_unitary(norb, seed=11)

    circ = FermionicCircuit(2 * norb)
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])

    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    once = circ._apply_unitary_(vec0, norb, nelec, True)
    twice = circ._apply_unitary_(once, norb, nelec, True)
    repeated = circ.repeat(2)._apply_unitary_(vec0, norb, nelec, True)

    np.testing.assert_allclose(repeated, twice, atol=1e-10)


def test_repeat_merges_trotter_step_boundaries():
    """The motivating use case: a naive symmetric step plus ``repeat`` equals hand-fused boundaries.

    A second-order Trotter step ends and begins with half-duration orbital rotations. Writing the step
    naively and repeating it leaves two rotations per step; ``MergeOrbitalRotations`` then fuses each
    interior pair, giving ``reps + 1`` rotations: exactly what fusing the boundaries by hand
    produces, and the same state vector. This only works because ``repeat`` flattens: the pass cannot
    see across an opaque per-repetition container.
    """
    import scipy.linalg
    from qiskit_fermions.circuit.library import Evolution
    from qiskit_fermions.operators import FermionOperator, ann, cre
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.passes import MergeOrbitalRotations

    norb = 2
    nelec = (1, 1)
    num_modes = 2 * norb
    occupation = [True, False, True, False]
    reps = 3
    time = 1.0
    dt = time / reps

    hopping = np.zeros((norb, norb))
    hopping[0, 1] = hopping[1, 0] = -1.0
    onsite = FermionOperator.from_dict(
        {(cre(p), ann(p), cre(p + norb), ann(p + norb)): 2.0 for p in range(norb)}
    )

    def rotation(duration):
        propagator = scipy.linalg.expm(-1j * duration * hopping)
        return scipy.linalg.block_diag(propagator, propagator)

    # the manual construction this feature replaces: the trailing half-step of one repetition is
    # folded into the leading half-step of the next by hand
    manual = FermionicCircuit(num_modes)
    manual.append(InitializeModes(occupation), manual.modes)
    manual.append(OrbitalRotation(rotation(dt / 2)), manual.modes)
    for step in range(reps):
        manual.append(Evolution(num_modes, onsite, dt, atomic=True), manual.modes)
        manual.append(OrbitalRotation(rotation(dt if step < reps - 1 else dt / 2)), manual.modes)

    # the same circuit, written naively and repeated
    trotter_step = FermionicCircuit(num_modes)
    trotter_step.append(OrbitalRotation(rotation(dt / 2)), trotter_step.modes)
    trotter_step.append(Evolution(num_modes, onsite, dt, atomic=True), trotter_step.modes)
    trotter_step.append(OrbitalRotation(rotation(dt / 2)), trotter_step.modes)

    circuit = FermionicCircuit(num_modes)
    circuit.append(InitializeModes(occupation), circuit.modes)
    circuit.compose(trotter_step.repeat(reps), inplace=True)

    # two rotations per repetition before the merge, one per repetition plus one after it
    assert circuit.count_ops()["OrbitalRotation"] == 2 * reps
    merged = FermionicPassManager([MergeOrbitalRotations()]).run(circuit)
    assert merged.count_ops()["OrbitalRotation"] == reps + 1
    assert merged.count_ops() == manual.count_ops()

    vec0 = ffsim.hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        merged._apply_unitary_(vec0, norb, nelec, True),
        manual._apply_unitary_(vec0, norb, nelec, True),
        atol=1e-10,
    )


def test_repeat_insert_barriers_blocks_the_merge():
    """``insert_barriers`` is the escape hatch from the boundary fusion ``repeat`` otherwise invites.

    Complements :func:`test_repeat_merges_trotter_step_boundaries`: the same repeated step keeps all
    of its rotations when the repetitions are separated by barriers, since
    :class:`.MergeOrbitalRotations` does not fuse across one. The state vector is unchanged either
    way, because a barrier carries no unitary effect and the fusion it blocks is exact.
    """
    import scipy.linalg
    from qiskit_fermions.circuit.library import Evolution
    from qiskit_fermions.operators import FermionOperator, ann, cre
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.passes import MergeOrbitalRotations

    norb = 2
    nelec = (1, 1)
    num_modes = 2 * norb
    reps = 3
    dt = 1.0 / reps

    hopping = np.zeros((norb, norb))
    hopping[0, 1] = hopping[1, 0] = -1.0
    interaction = FermionOperator.from_dict(
        {(cre(p), ann(p), cre(p + norb), ann(p + norb)): 2.0 for p in range(norb)}
    )

    def rotation(duration):
        propagator = scipy.linalg.expm(-1j * duration * hopping)
        return scipy.linalg.block_diag(propagator, propagator)

    step = FermionicCircuit(num_modes)
    step.append(OrbitalRotation(rotation(dt / 2)), step.modes)
    step.append(Evolution(num_modes, interaction, dt, atomic=True), step.modes)
    step.append(OrbitalRotation(rotation(dt / 2)), step.modes)

    pass_manager = FermionicPassManager([MergeOrbitalRotations()])
    fused = pass_manager.run(step.repeat(reps))
    kept = pass_manager.run(step.repeat(reps, insert_barriers=True))

    assert fused.count_ops()["OrbitalRotation"] == reps + 1
    assert kept.count_ops()["OrbitalRotation"] == 2 * reps

    vec0 = ffsim.hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(
        kept._apply_unitary_(vec0, norb, nelec, True),
        fused._apply_unitary_(vec0, norb, nelec, True),
        atol=1e-10,
    )


def test_apply_unitary_rejects_measurement():
    """A measured circuit cannot be applied as a unitary, and says so specifically.

    A barrier is skipped because it is unitarily the identity; a measurement is not, so skipping it
    would silently simulate a different circuit. The check is that the error names the cause rather
    than falling through to the generic "does not implement the protocol" message.
    """
    norb, nelec = 2, (1, 1)
    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes([1, 0, 1, 0]), circ.modes)
    circ.measure_all()

    vec = ffsim.hartree_fock_state(norb, nelec)
    with pytest.raises(TypeError, match="measurement is not a unitary operation"):
        circ._apply_unitary_(vec, norb, nelec, True)
