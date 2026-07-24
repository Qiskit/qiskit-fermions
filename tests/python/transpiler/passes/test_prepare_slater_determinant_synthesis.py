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

"""Slater determinant preparation synthesis tests."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XGate, XXPlusYYGate
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import Statevector
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)
from qiskit_fermions.linalg import givens_decomposition_slater
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    GivensDecompositionOrbitalRotationSynthesis,
    GivensDecompositionSlaterDeterminantSynthesis,
    TrivialF2QLayout,
    TrivialOccupationInitializeModesSynthesis,
)
from qiskit_fermions.transpiler.passes.synthesis.prepare_slater_determinant import (
    _peel_disjoint_unit_orbitals,
    _split_mixing_blocks,
)

from ...utils import random_unitary


def _synthesize(circ: FermionicCircuit, synth: F2QSynthesis):
    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    return pm.run(circ)


def _slater_synth() -> F2QSynthesis:
    # most tests below assert the peeling (minimize-2q-gate-count) behavior, which is *not* the
    # default; request it explicitly. The peel-free default path is covered by its own tests.
    synth = F2QSynthesis()
    synth.methods["PrepareSlaterDeterminant"] = GivensDecompositionSlaterDeterminantSynthesis(
        minimize_2q_gate_count=True
    )
    return synth


def _reference_synth() -> F2QSynthesis:
    """Synthesizes the equivalent InitializeModes + OrbitalRotation via the trusted square path."""
    synth = F2QSynthesis()
    synth.methods["InitializeModes"] = TrivialOccupationInitializeModesSynthesis()
    synth.methods["OrbitalRotation"] = GivensDecompositionOrbitalRotationSynthesis()
    return synth


def _assert_equal_up_to_global_phase(qc_a, qc_b):
    sv_a = Statevector(qc_a).data
    sv_b = Statevector(qc_b).data
    # align the global phase on the largest-magnitude amplitude, then compare
    k = int(np.argmax(np.abs(sv_b)))
    phase = sv_a[k] / sv_b[k]
    np.testing.assert_allclose(sv_a, phase * sv_b, atol=1e-10)


def _reference_circuit(num_modes, occupation, rotation):
    """The faithful InitializeModes + OrbitalRotation synthesis (never peels or block-splits)."""
    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    return _synthesize(reference, _reference_synth())


def _assert_matches_reference(num_modes, occupation, rotation):
    """Synthesize via the Slater plugin and assert it matches the faithful reference path.

    Returns the synthesized Slater circuit so callers can additionally inspect its gate counts.
    """
    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    qc_slater = _synthesize(slater, _slater_synth())
    _assert_equal_up_to_global_phase(qc_slater, _reference_circuit(num_modes, occupation, rotation))
    return qc_slater


def test_prepare_slater_determinant_synthesis_matches_reference():
    """The reduced Slater synthesis prepares the same state as the full InitializeModes+rotation."""
    num_modes = 6
    occupation = [True, True, True, False, False, False]
    rotation = random_unitary(num_modes, seed=42)
    _assert_matches_reference(num_modes, occupation, rotation)


def test_prepare_slater_determinant_synthesis_scattered_occupation():
    """A non-leading, non-contiguous occupation still prepares the correct state.

    The reduced decomposition seeds a *leading*-``m`` reference (X gates on the first ``m`` modes)
    regardless of where the occupied modes actually sit; ``occupation``'s positions enter only via
    the column selection ``rotation_unitary[:, occupied]``. This checks that this leading-reference
    convention is nonetheless correct for an occupation whose ``True`` entries are neither leading
    nor contiguous.
    """
    num_modes = 6
    occupation = [False, True, False, True, True, False]
    rotation = random_unitary(num_modes, seed=7)
    _assert_matches_reference(num_modes, occupation, rotation)


def test_prepare_slater_determinant_synthesis_reduced_gate_count():
    """The Slater synthesis uses fewer gates than the full orbital rotation and emits no phases."""
    num_modes = 6
    nocc = 3
    occupation = [i < nocc for i in range(num_modes)]
    rotation = random_unitary(num_modes, seed=1)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    ops = _synthesize(slater, _slater_synth()).count_ops()

    # at most m(n-m) XXPlusYYGates, m X gates, and crucially no phase gates
    assert ops.get("xx_plus_yy", 0) <= nocc * (num_modes - nocc)
    assert ops.get("x", 0) == nocc
    assert "p" not in ops

    # the full orbital-rotation synthesis of the same rotation uses the n(n-1)/2 brick-wall + phases
    ref_ops = _reference_circuit(num_modes, occupation, rotation).count_ops()
    assert ops.get("xx_plus_yy", 0) < ref_ops.get("xx_plus_yy", 0)


def _givens_2mode(theta: float, p: int, q: int, num_modes: int) -> np.ndarray:
    """A real orbital rotation mixing only modes ``p`` and ``q`` (identity elsewhere).

    Used to build occupied spaces that factor into disjoint unit-orbitals plus a genuinely-mixing
    block, exercising the peel-off fast path.
    """
    rot = np.eye(num_modes, dtype=complex)
    rot[p, p] = np.cos(theta)
    rot[p, q] = -np.sin(theta)
    rot[q, p] = np.sin(theta)
    rot[q, q] = np.cos(theta)
    return rot


def _multi_block_rotation(num_modes: int, blocks: list[list[int]], *, seed: int) -> np.ndarray:
    """Identity except for an independent random unitary on each mode group in ``blocks``.

    Each group may span arbitrary -- possibly non-contiguous -- modes. Placing several genuinely-mixing
    clusters on disjoint mode windows (with empty modes between them) builds the targets that
    block-splitting exists to handle: each cluster synthesizes on its own contiguous window instead of
    being transported across the whole line by a single leading sweep. Passing a single non-contiguous
    group (e.g. ``[[0, 2]]``) instead exercises the Jordan-Wigner Z-string compensation: a peeled
    unit-orbital occupied on a mode strictly between two of the mixing modes drives a sweep
    ``XXPlusYYGate`` onto physically non-adjacent qubits.
    """
    rotation = np.eye(num_modes, dtype=complex)
    for offset, modes in enumerate(blocks):
        block = random_unitary(len(modes), seed=seed + offset)
        for a, mode_a in enumerate(modes):
            for b, mode_b in enumerate(modes):
                rotation[mode_a, mode_b] = block[a, b]
    return rotation


def _two_qubit_count(qc) -> int:
    return qc.count_ops().get("xx_plus_yy", 0)


def test_prepare_slater_determinant_synthesis_peels_disjoint_unit_orbital():
    """A disjoint unit-orbital is placed directly, not transported out with bare SWAPs.

    The rotation mixes only modes 0 and 1; occupying column 3 (a bare basis vector on mode 3) and
    column 1 (a superposition of modes 0 and 1) factors the occupied space into the disjoint
    unit-orbital ``e_3`` plus one genuine two-mode rotation. The reduced sweep therefore needs a
    single ``XXPlusYYGate`` -- the two transport SWAPs the leading-reference sweep would emit for
    ``e_3`` are peeled away -- while the prepared state is unchanged.
    """
    num_modes = 4
    occupation = [False, True, False, True]
    rotation = _givens_2mode(0.6, 0, 1, num_modes)

    qc_slater = _assert_matches_reference(num_modes, occupation, rotation)
    ops = qc_slater.count_ops()

    # only the genuine 0-1 mixing survives; no bare-SWAP transport of the e_3 orbital
    assert ops.get("xx_plus_yy", 0) == 1
    assert ops.get("x", 0) == 2


def test_prepare_slater_determinant_synthesis_permutation_no_two_qubit_gates():
    """A pure permutation rotation (here the identity) emits only X gates, no ``XXPlusYYGate``.

    With ``rotation`` the identity and a non-leading, non-contiguous occupation the entire occupied
    space is a signed permutation, so every occupied orbital peels: the state is prepared with X
    gates placed directly on the occupied modes and zero two-qubit gates.
    """
    num_modes = 4
    occupation = [False, True, False, True]
    rotation = np.eye(num_modes, dtype=complex)

    qc_slater = _assert_matches_reference(num_modes, occupation, rotation)
    ops = qc_slater.count_ops()

    assert ops.get("xx_plus_yy", 0) == 0
    assert ops.get("x", 0) == 2
    assert "p" not in ops


def test_prepare_slater_determinant_synthesis_multiple_disjoint_units():
    """Several disjoint unit-orbitals peel independently around a single mixing block.

    The rotation mixes only the adjacent modes 0 and 1; occupying modes 0, 4, 5 gives two disjoint
    unit-orbitals (``e_4``, ``e_5``) plus a single occupied orbital spanning the adjacent pair {0, 1}.
    (Occupying *both* mixed columns 0 and 1 would fill the whole {0, 1} subspace and reduce to a plain
    permutation -- selecting only column 0 keeps a genuine two-mode rotation.) The two disjoint units
    peel to direct X gates, leaving a single ``XXPlusYYGate`` on the adjacent mixing pair.
    """
    num_modes = 6
    occupation = [True, False, False, False, True, True]
    rotation = _givens_2mode(0.5, 0, 1, num_modes)

    qc_slater = _assert_matches_reference(num_modes, occupation, rotation)
    ops = qc_slater.count_ops()

    assert ops.get("xx_plus_yy", 0) == 1
    assert ops.get("x", 0) == 3


def test_prepare_slater_determinant_synthesis_generic_rotation_not_peeled():
    """A generic entangled rotation has nothing to peel; the sweep is unchanged.

    This guards the fast path against regressing the common case: a fully-mixing occupied space must
    still be synthesized by the ordinary reduced Givens sweep (its ``m(n-m)`` bound intact) and
    prepare the correct state.
    """
    num_modes = 6
    nocc = 3
    occupation = [i < nocc for i in range(num_modes)]
    rotation = random_unitary(num_modes, seed=42)

    qc_slater = _assert_matches_reference(num_modes, occupation, rotation)
    ops = qc_slater.count_ops()

    # nothing peels: the full leading-reference sweep runs (up to m(n-m) two-qubit gates, m X gates)
    assert ops.get("x", 0) == nocc
    assert 0 < ops.get("xx_plus_yy", 0) <= nocc * (num_modes - nocc)


def test_prepare_slater_determinant_synthesis_full_occupation():
    """A fully occupied reference (m == n) needs no Givens rotations, only the X gates."""
    num_modes = 4
    occupation = [True] * num_modes
    rotation = random_unitary(num_modes, seed=2)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    ops = _synthesize(slater, _slater_synth()).count_ops()

    # m == n: the occupied space is the whole space, so any rotation within it is discarded
    assert ops.get("x", 0) == num_modes
    assert ops.get("xx_plus_yy", 0) == 0


# --- correctness across occupied-space structures --------------------------------------------------
#
# Each case is (label, num_modes, occupied_columns, rotation-builder). ``occupied_columns`` selects
# the rotation columns that become the occupied orbitals; the builder controls how the rotation mixes
# modes. Together they exercise: disjoint multi-mode mixing blocks (with an empty gap between them --
# the block-split payoff), adjacent multi-mode blocks, a nested-window mixing pair, an
# interleaved-window mixing pair, a dense single mixing block, a pure permutation (everything peels),
# and the #206 peel-bug shape (a peeled unit strictly between a mixing pair's physical endpoints).

_STRUCTURE_CASES = [
    # disjoint multi-mode blocks {0,1} and {5,6} on an 8-line, one orbital occupied per block
    (
        "disjoint_multimode_blocks",
        8,
        [0, 5],
        lambda: _multi_block_rotation(8, [[0, 1], [5, 6]], seed=11),
    ),
    # two adjacent multi-mode blocks {0,1} and {2,3} (windows touch but do not overlap)
    (
        "adjacent_multimode_blocks",
        6,
        [0, 2],
        lambda: _multi_block_rotation(6, [[0, 1], [2, 3]], seed=13),
    ),
    # nested windows: mixing {0,3} and mixing {1,2} -- window [0,3] spans modes 1,2 so they MUST merge
    ("nested_windows", 4, [0, 1], lambda: _multi_block_rotation(4, [[0, 3], [1, 2]], seed=17)),
    # interleaved windows: mixing {0,4} and mixing {1,5} -- windows overlap, must merge into one block
    ("interleaved_windows", 6, [0, 1], lambda: _multi_block_rotation(6, [[0, 4], [1, 5]], seed=19)),
    # a single dense mixing block spanning every mode (one contiguous window, reproduces leading)
    ("dense_single_block", 5, [0, 1, 2], lambda: random_unitary(5, seed=23)),
    # a pure permutation: identity rotation, scattered occupation -- every orbital peels
    ("pure_permutation", 6, [1, 3, 4], lambda: np.eye(6, dtype=complex)),
    # the #206 peel-bug shape: mixing {0,2} with a peeled unit on the interior mode 1
    ("peel_bug_interior_unit", 3, [0, 1], lambda: _multi_block_rotation(3, [[0, 2]], seed=3)),
]


@pytest.mark.parametrize(
    "label,num_modes,occupied_columns,build_rotation",
    _STRUCTURE_CASES,
    ids=[case[0] for case in _STRUCTURE_CASES],
)
def test_prepare_slater_determinant_synthesis_structures(
    label, num_modes, occupied_columns, build_rotation
):
    """Peel + block-split prepares the correct state across occupied-space structures.

    Covers disjoint and adjacent multi-mode mixing blocks, nested/interleaved windows (which must
    share a block or a rotation crosses an occupied mode), a dense single block, a pure permutation,
    and the #206 peel-bug shape. Every case is checked against the faithful ``InitializeModes +
    OrbitalRotation`` reference up to a global phase.
    """
    rotation = build_rotation()
    occupation = [i in occupied_columns for i in range(num_modes)]
    _assert_matches_reference(num_modes, occupation, rotation)


def test_prepare_slater_determinant_synthesis_interleaved_peel_odd_parity():
    """A single peeled unit-orbital between two mixing modes still prepares the correct state.

    This is the regression case for the Jordan-Wigner Z-string bug: the rotation mixes the
    *non-adjacent* modes 0 and 2 (occupying column 0 keeps a genuine two-mode rotation), while mode 1
    is an occupied disjoint unit-orbital that peels. The reduced sweep emits an ``XXPlusYYGate`` on the
    physically non-adjacent qubits (0, 2); the occupied peeled mode 1 sits strictly between them, so
    its Jordan-Wigner Z-string (odd parity) must be folded in as a ``pi`` phase shift. Without that
    compensation the prepared state is wrong (overlap ~0.078 vs. the faithful reference).
    """
    num_modes = 3
    occupation = [True, True, False]
    rotation = _multi_block_rotation(num_modes, [[0, 2]], seed=3)

    qc_slater = _assert_matches_reference(num_modes, occupation, rotation)
    ops = qc_slater.count_ops()

    # mode 1 peels; the genuine 0-2 mixing emits a single XXPlusYYGate on non-adjacent qubits
    assert ops.get("xx_plus_yy", 0) == 1
    assert ops.get("x", 0) == 2


def test_prepare_slater_determinant_synthesis_interleaved_peel_even_parity():
    """Two peeled unit-orbitals between the mixing endpoints leave the sweep uncompensated.

    Guards against a naive "flip whenever any peel is between the endpoints" fix: the rotation mixes
    the non-adjacent modes 0 and 3, with occupied unit-orbitals on *both* modes 1 and 2 (even parity).
    Their two Jordan-Wigner Z-strings cancel, so the ``XXPlusYYGate`` must be emitted with its base
    phase -- the compensation must stay inert. The prepared state must match the faithful reference.
    """
    num_modes = 4
    occupation = [True, True, True, False]
    rotation = _multi_block_rotation(num_modes, [[0, 3]], seed=5)

    qc_slater = _assert_matches_reference(num_modes, occupation, rotation)
    ops = qc_slater.count_ops()

    # modes 1 and 2 peel; a single XXPlusYYGate on the non-adjacent mixing pair (0, 3)
    assert ops.get("xx_plus_yy", 0) == 1
    assert ops.get("x", 0) == 3


def test_prepare_slater_determinant_synthesis_interleaved_peel_no_overhead():
    """A peeled unit sitting mid-window does not raise the two-qubit-gate count.

    Directly answers the maintainer's question: when a peeled unit-orbital lands strictly between a
    mixing rotation's physical endpoints, peeling still peels it out (one free ``XGate``) rather than
    absorbing it into a denser block. So the emitted two-qubit-gate count equals the count of the same
    mixing block synthesized *alone* -- the interior peel adds no transport SWAPs. Checked for a small
    mixing pair with one interior peel and for a wider block with three interior peels.
    """
    for mixing_modes, unit_modes in (([0, 2], [1]), ([0, 4], [1, 2, 3])):
        num_modes = max(*mixing_modes, *unit_modes) + 1
        rotation = _multi_block_rotation(num_modes, [mixing_modes], seed=29)

        # occupy the first mixing column (a genuine mixing orbital) plus the interior units
        occupied_columns = [mixing_modes[0], *unit_modes]
        occupation = [i in occupied_columns for i in range(num_modes)]
        qc_interleaved = _assert_matches_reference(num_modes, occupation, rotation)

        # the same mixing orbital with the interior units *removed* (no peel between the endpoints):
        # its two-qubit-gate count is the floor the interleaved case must not exceed
        occupation_alone = [i == mixing_modes[0] for i in range(num_modes)]
        slater_alone = FermionicCircuit(num_modes)
        slater_alone.append(
            PrepareSlaterDeterminant(occupation_alone, rotation), slater_alone.modes
        )
        qc_alone = _synthesize(slater_alone, _slater_synth())

        assert _two_qubit_count(qc_interleaved) <= _two_qubit_count(qc_alone)


def test_prepare_slater_determinant_synthesis_intra_block_gates_mode_adjacent():
    """Every emitted two-qubit gate is either mode-adjacent or a compensated interior-peel gate.

    Block-split guarantees intra-block gates act on physically adjacent qubits; the only non-adjacent
    gate the plugin ever emits spans a peeled interior mode, and then only with the Jordan-Wigner
    Z-string folded into its phase. Assert exactly that: each ``XXPlusYYGate`` has ``|i - j| == 1`` or
    an odd number of peeled occupied modes strictly between its endpoints.
    """
    num_modes = 8
    # disjoint mixing clusters {0,1} and {5,6} plus a peeled interior unit on mode 3
    rotation = _multi_block_rotation(num_modes, [[0, 1], [5, 6]], seed=31)
    occupied_columns = [0, 3, 5]
    occupation = [i in occupied_columns for i in range(num_modes)]

    orbital_coeffs = rotation[:, np.nonzero(occupation)[0]].T
    peel_modes, _, _ = _peel_disjoint_unit_orbitals(orbital_coeffs)

    qc_slater = _assert_matches_reference(num_modes, occupation, rotation)

    for instruction in qc_slater.data:
        if instruction.operation.name != "xx_plus_yy":
            continue
        i, j = (qc_slater.find_bit(qubit).index for qubit in instruction.qubits)
        lo, hi = min(i, j), max(i, j)
        peels_between = sum(1 for p in peel_modes if lo < p < hi)
        assert hi - lo == 1 or peels_between % 2 == 1


def test_prepare_slater_determinant_synthesis_block_split_reduces_two_qubit_gates():
    """Block-split emits strictly fewer two-qubit gates than a single leading sweep would.

    On an 8-line with two disjoint mixing clusters {0,1} and {5,6} plus a peeled unit on mode 3, a
    single leading-reference sweep transports both clusters across the whole line; block-splitting
    synthesizes each cluster on its own contiguous window. This asserts the actual payoff: the
    plugin's two-qubit-gate count is strictly below what one leading sweep over all kept modes emits.
    """
    num_modes = 8
    rotation = _multi_block_rotation(num_modes, [[0, 1], [5, 6]], seed=31)
    occupied_columns = [0, 3, 5]
    occupation = [i in occupied_columns for i in range(num_modes)]

    qc_slater = _assert_matches_reference(num_modes, occupation, rotation)

    # what a single leading sweep over the mixing remainder (no block split) would emit: peel the unit,
    # then hand the whole kept-mode remainder to one givens_decomposition_slater call
    orbital_coeffs = rotation[:, np.nonzero(occupation)[0]].T
    _, _, reduced = _peel_disjoint_unit_orbitals(orbital_coeffs)
    single_sweep = sum(
        1 for c, _, _, _ in givens_decomposition_slater(reduced) if not np.isclose(np.acos(c), 0.0)
    )

    assert _two_qubit_count(qc_slater) < single_sweep


# --- seeded randomized sweep + negative controls ---------------------------------------------------


def _draw_multi_block_target(rng):
    """Draw a random multi-block occupied-space target that may include interleaved peels.

    Returns ``(num_modes, occupation, rotation, is_multi_block, is_interleaved_peel)`` where the two
    flags record whether the drawn target genuinely stresses block-splitting (more than one mixing
    cluster) and the #206 interior-peel compensation (a peeled occupied mode strictly inside the kept
    mixing span). ``None`` is returned for a degenerate draw (empty or full occupation).
    """
    num_modes = int(rng.integers(4, 9))

    # lay one or two mixing clusters out left-to-right on *contiguous* mode ranges separated by
    # random gaps: this keeps their physical windows genuinely disjoint (exercising block-splitting)
    # while the gap modes host scattered unit-orbitals -- and occupying fewer than all of a cluster's
    # columns lets a peeled unit land *inside* a cluster's window (an interleaved peel).
    num_blocks = int(rng.integers(1, 3))
    blocks: list[list[int]] = []
    rest: list[int] = []
    cursor = 0
    for block in range(num_blocks):
        remaining_blocks = num_blocks - block
        # leave room for the remaining clusters (>=2 modes each) plus optional gaps
        if num_modes - cursor < 2 * remaining_blocks:
            break
        # a random leading gap whose modes become candidate unit-orbitals
        max_gap = num_modes - cursor - 2 * remaining_blocks
        gap = int(rng.integers(0, max_gap + 1))
        rest += list(range(cursor, cursor + gap))
        cursor += gap
        size = int(rng.integers(2, min(4, num_modes - cursor - 2 * (remaining_blocks - 1)) + 1))
        blocks.append(list(range(cursor, cursor + size)))
        cursor += size
    rest += list(range(cursor, num_modes))

    rotation = np.eye(num_modes, dtype=complex)
    for offset, block_modes in enumerate(blocks):
        sub = random_unitary(len(block_modes), seed=int(rng.integers(0, 10**6)) + offset)
        for a, mode_a in enumerate(block_modes):
            for b, mode_b in enumerate(block_modes):
                rotation[mode_a, mode_b] = sub[a, b]

    # occupy fewer than all columns of each mixing block (occupying all would fill the subspace and
    # collapse to a permutation), plus some isolated units -- interior ones drive interleaved peels
    occupied_columns: list[int] = []
    for block_modes in blocks:
        num_occ = int(rng.integers(1, len(block_modes)))
        occupied_columns += block_modes[:num_occ]
    occupied_columns += [mode for mode in rest if rng.random() < 0.5]

    occupation = [i in occupied_columns for i in range(num_modes)]
    if not any(occupation) or all(occupation):
        return None

    orbital_coeffs = rotation[:, np.nonzero(occupation)[0]].T
    peel_modes, kept_modes, reduced = _peel_disjoint_unit_orbitals(orbital_coeffs)

    is_multi_block = False
    if reduced.shape[0] > 0:
        windows = []
        for row in range(reduced.shape[0]):
            support = [kept_modes[col] for col in np.nonzero(np.abs(reduced[row]) > 1e-10)[0]]
            windows.append((min(support), max(support)))
        order = sorted(range(len(windows)), key=lambda r: windows[r][0])
        running_hi = windows[order[0]][1]
        for r in order[1:]:
            lo, hi = windows[r]
            if lo > running_hi:
                is_multi_block = True
            running_hi = max(running_hi, hi)

    is_interleaved_peel = bool(
        reduced.shape[0] > 0
        and peel_modes
        and any(min(kept_modes) < p < max(kept_modes) for p in peel_modes)
    )
    return num_modes, occupation, rotation, is_multi_block, is_interleaved_peel


def test_prepare_slater_determinant_synthesis_randomized_sweep():
    """Randomized stress test of peel + block-split against the faithful reference.

    Draws many multi-block occupied spaces -- mixing single- and multi-mode clusters on disjoint
    windows, plus disjoint unit-orbitals, some of which land *between* a mixing cluster's endpoints.
    Each synthesized state is checked against the faithful ``InitializeModes + OrbitalRotation``
    reference (which never peels or block-splits). The seed is fixed for determinism and asserted to
    genuinely exercise both load-bearing regimes -- multiple mixing blocks (block-splitting) and
    interleaved peels (the #206 Z-string compensation) -- so the test cannot pass trivially.
    """
    rng = np.random.default_rng(20260724)
    trials = 0
    multi_block = 0
    interleaved_peel = 0
    for _ in range(300):
        drawn = _draw_multi_block_target(rng)
        if drawn is None:
            continue
        num_modes, occupation, rotation, is_multi_block, is_interleaved_peel = drawn
        trials += 1
        multi_block += is_multi_block
        interleaved_peel += is_interleaved_peel

        _assert_matches_reference(num_modes, occupation, rotation)

    # the seed genuinely stresses both load-bearing regimes, so the sweep guards them rather than
    # trivially passing on structureless draws
    assert trials > 150
    assert multi_block > 20
    assert interleaved_peel > 10


def _synthesize_with_support_grouping(num_modes, occupation, rotation):
    """Negative control: synthesize grouping mixing orbitals by raw SUPPORT overlap, not window.

    This deliberately-wrong variant reproduces the plugin's peel + emission recipe exactly but groups
    the mixing remainder by whether orbital *supports* intersect instead of whether their physical
    *windows* overlap. Nested/interleaved windows (disjoint support, overlapping window) then split
    into separate blocks and emit a rotation across an occupied mode -- preparing the wrong state.
    """
    occupied = np.nonzero(occupation)[0]
    orbital_coeffs = rotation[:, occupied].T
    peel_modes, kept_modes, reduced = _peel_disjoint_unit_orbitals(orbital_coeffs)

    qc = QuantumCircuit(num_modes)
    for mode in peel_modes:
        qc.append(XGate(), [mode])
    if reduced.shape[0] == 0:
        return qc

    # group by raw support overlap (WRONG): union-find over shared kept-mode columns
    supports = [
        set(np.nonzero(np.abs(reduced[row]) > 1e-10)[0].tolist()) for row in range(reduced.shape[0])
    ]
    parent = list(range(len(supports)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(supports)):
        for b in range(a + 1, len(supports)):
            if supports[a] & supports[b]:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for row in range(len(supports)):
        groups.setdefault(find(row), []).append(row)

    for block_rows in groups.values():
        cols = sorted(set().union(*(supports[row] for row in block_rows)))
        block_modes = [kept_modes[col] for col in cols]
        sub = np.ascontiguousarray(reduced[np.ix_(block_rows, cols)])
        for i in range(len(block_rows)):
            qc.append(XGate(), [block_modes[i]])
        for c, s, i, j in givens_decomposition_slater(sub):
            c_angle = np.acos(c)
            if not np.isclose(c_angle, 0.0):
                mode_i, mode_j = block_modes[i], block_modes[j]
                lo, hi = min(mode_i, mode_j), max(mode_i, mode_j)
                sign = -1 if sum(1 for p in peel_modes if lo < p < hi) % 2 == 0 else 1
                qc.append(
                    XXPlusYYGate(2 * c_angle, np.angle(s) + sign * 0.5 * np.pi), [mode_i, mode_j]
                )
    return qc


def _synthesize_without_z_string(num_modes, occupation, rotation):
    """Negative control: reproduce the plugin exactly but DROP the #206 Z-string sign compensation.

    Groups by physical windows (correct) but always emits the base ``angle(s) - pi/2`` phase, ignoring
    peeled modes between a rotation's endpoints. An interleaved peel with odd parity then prepares the
    wrong state.
    """
    occupied = np.nonzero(occupation)[0]
    orbital_coeffs = rotation[:, occupied].T
    peel_modes, kept_modes, reduced = _peel_disjoint_unit_orbitals(orbital_coeffs)

    qc = QuantumCircuit(num_modes)
    for mode in peel_modes:
        qc.append(XGate(), [mode])
    if reduced.shape[0] == 0:
        return qc

    for block_rows, block_modes, sub in _split_mixing_blocks(reduced, kept_modes):
        for i in range(len(block_rows)):
            qc.append(XGate(), [block_modes[i]])
        for c, s, i, j in givens_decomposition_slater(sub):
            c_angle = np.acos(c)
            if not np.isclose(c_angle, 0.0):
                mode_i, mode_j = block_modes[i], block_modes[j]
                # DROP the Z-string sign: always the base phase
                qc.append(XXPlusYYGate(2 * c_angle, np.angle(s) - 0.5 * np.pi), [mode_i, mode_j])
    return qc


def _overlap(qc, num_modes, occupation, rotation) -> float:
    qc_reference = _reference_circuit(num_modes, occupation, rotation)
    return float(abs(Statevector(qc).inner(Statevector(qc_reference))) ** 2)


def test_prepare_slater_determinant_synthesis_negative_control_support_grouping():
    """A support-overlap grouping (instead of window overlap) breaks on nested/interleaved windows.

    Guards the correct-path sweep against a too-gentle formulation: grouping mixing orbitals by raw
    support overlap rather than physical-window overlap splits a nested-window pair into separate
    blocks and emits a rotation across an occupied mode -- preparing the wrong state. The target is
    the canonical nested-window case (mixing ``{0,3}`` and mixing ``{1,2}``: disjoint support, but
    window ``[0,3]`` spans the occupied modes 1, 2), constructed so the pathology is hit by design.
    The correct window grouping (the plugin) prepares this same target correctly -- verified by the
    ``nested_windows`` structure case above.
    """
    num_modes = 4
    rotation = _multi_block_rotation(num_modes, [[0, 3], [1, 2]], seed=17)
    occupation = [i in (0, 1) for i in range(num_modes)]

    qc_wrong = _synthesize_with_support_grouping(num_modes, occupation, rotation)
    assert not np.isclose(_overlap(qc_wrong, num_modes, occupation, rotation), 1.0, atol=1e-8)


def test_prepare_slater_determinant_synthesis_negative_control_no_z_string():
    """Dropping the #206 Z-string sign compensation breaks on an odd-parity interleaved peel.

    Guards the correct-path sweep's other load-bearing assumption: emitting the base ``XXPlusYYGate``
    phase without folding in the Jordan-Wigner Z-string of an odd number of peeled interior modes
    prepares the wrong state. The target is the #206 peel-bug shape (mixing the non-adjacent modes
    ``{0,2}`` with a peeled unit on the interior mode 1, odd parity), constructed so the compensation
    is load-bearing. The compensated plugin prepares this same target correctly -- verified by
    ``test_..._interleaved_peel_odd_parity`` above.
    """
    num_modes = 3
    rotation = _multi_block_rotation(num_modes, [[0, 2]], seed=3)
    occupation = [True, True, False]

    qc_wrong = _synthesize_without_z_string(num_modes, occupation, rotation)
    assert not np.isclose(_overlap(qc_wrong, num_modes, occupation, rotation), 1.0, atol=1e-8)


# --- minimize_2q_gate_count toggle -----------------------------------------------------------------
#
# The default synthesis uses the full leading-reference sweep (block splitting retained): every
# emitted rotation is nearest-neighbor along the mode order, minimizing the routed two-qubit *depth*.
# Setting ``minimize_2q_gate_count=True`` peels disjoint unit-orbitals instead, minimizing the
# two-qubit gate *count* but potentially stranding a residual mixing orbital across a wide window (a
# long-range gate that inflates the routed depth). The two are a non-dominated trade-off; both prepare
# the same state (up to a global phase). ``_slater_synth`` above requests the peeling (count-min) path
# explicitly; the plain default plugin below exercises the depth-min path.


def _slater_synth_default() -> F2QSynthesis:
    synth = F2QSynthesis()
    synth.methods["PrepareSlaterDeterminant"] = GivensDecompositionSlaterDeterminantSynthesis()
    return synth


def test_prepare_slater_determinant_minimize_2q_gate_count_keyword_only():
    """``minimize_2q_gate_count`` is keyword-only and defaults to ``False`` (peeling off)."""
    assert GivensDecompositionSlaterDeterminantSynthesis().minimize_2q_gate_count is False
    assert (
        GivensDecompositionSlaterDeterminantSynthesis(
            minimize_2q_gate_count=True
        ).minimize_2q_gate_count
        is True
    )
    with pytest.raises(TypeError):
        GivensDecompositionSlaterDeterminantSynthesis(True)  # type: ignore[misc]


@pytest.mark.parametrize(
    "label,num_modes,occupied_columns,build_rotation",
    _STRUCTURE_CASES,
    ids=[case[0] for case in _STRUCTURE_CASES],
)
def test_prepare_slater_determinant_default_matches_reference(
    label, num_modes, occupied_columns, build_rotation
):
    """The default (peel-free) synthesis prepares the correct state for every occupied-space structure.

    Runs the same structural cases as the peeling path against the faithful ``InitializeModes +
    OrbitalRotation`` reference. The ``minimize_2q_gate_count`` setting only changes *how* the state is
    synthesized (full leading-reference sweep vs peel + direct X gates), never *which* state -- so the
    reference comparison must hold for both settings.
    """
    rotation = build_rotation()
    occupation = [i in occupied_columns for i in range(num_modes)]

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    qc_default = _synthesize(slater, _slater_synth_default())
    _assert_equal_up_to_global_phase(
        qc_default, _reference_circuit(num_modes, occupation, rotation)
    )


def _xx_plus_yy_spans(qc) -> list[int]:
    """The physical qubit-index spans ``|i - j|`` of every emitted ``XXPlusYYGate``."""
    spans = []
    for instruction in qc.data:
        if instruction.operation.name != "xx_plus_yy":
            continue
        i, j = (qc.find_bit(qubit).index for qubit in instruction.qubits)
        spans.append(abs(i - j))
    return spans


def test_prepare_slater_determinant_default_avoids_long_range_gate():
    """The default (depth-minimizing) synthesis emits a nearest-neighbor chain where peeling strands.

    This is the routing motivation for the default: a dense occupation with one disjoint mixing orbital
    spanning the whole window peels its interior modes to X gates and strands the residual as a single
    long-range ``XXPlusYYGate`` on the block extremes (``minimize_2q_gate_count=True``). The default
    instead emits the Clements brickwork -- every rotation mode-adjacent (``|i - j| == 1``) -- while
    preparing the identical state.

    The rotation mixes only the block *extremes* (modes 0 and ``num_modes - 1``) and is the identity on
    the interior; occupying columns ``0 .. num_modes - 2`` (all but the last mode) makes the interior
    modes 1..num_modes-2 disjoint unit-orbitals that peel, leaving one mixing orbital on ``{0,
    num_modes - 1}``. This is exactly the N2 LUCJ HF state-prep shape (all-but-one mode occupied) that
    motivated the toggle.
    """
    num_modes = 6
    rotation = _givens_2mode(0.6, 0, num_modes - 1, num_modes)
    occupation = [i < num_modes - 1 for i in range(num_modes)]

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)

    qc_count_min = _synthesize(slater, _slater_synth())  # minimize_2q_gate_count=True
    qc_default = _synthesize(slater, _slater_synth_default())

    # peeling (count-min) strands a single long-range gate on the block extremes; the default keeps
    # every rotation nearest-neighbor
    assert max(_xx_plus_yy_spans(qc_count_min)) == num_modes - 1
    default_spans = _xx_plus_yy_spans(qc_default)
    assert default_spans and max(default_spans) == 1

    # ... and both prepare the same state (up to a global phase)
    _assert_equal_up_to_global_phase(qc_count_min, qc_default)
