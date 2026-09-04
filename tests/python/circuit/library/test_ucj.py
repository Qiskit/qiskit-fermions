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

"""Structural tests for the UCJ ansatz gate.

The ansatz tensors and their construction belong to ffsim, so the tests here cover only what this
package adds: reading the spin variant and mode count off the ffsim operator type, the gate sequence
the definition expands into, and the diagonal Coulomb term grouping that lets :class:`.Evolution`
synthesize one parallel layer per group.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import UCJ, OrbitalRotation

ffsim = pytest.importorskip("ffsim")


def _balanced_op(norb, n_reps, *, seed, with_final=False):
    """Returns a random spin-balanced ffsim UCJ operator."""
    return ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=n_reps, with_final_orbital_rotation=with_final, seed=seed
    )


def _spinless_op(norb, n_reps, *, seed):
    """Returns a random spinless ffsim UCJ operator."""
    return ffsim.random.random_ucj_op_spinless(norb, n_reps=n_reps, seed=seed)


def test_ucj_balanced_reads_norb_and_modes_off_the_operator():
    """A spin-balanced operator gives a ``2 * norb``-mode gate."""
    op = _balanced_op(3, 2, seed=1234)
    gate = UCJ(op)
    assert gate.ucj_op is op
    assert gate.norb == 3
    assert gate.n_reps == 2
    assert gate.num_modes == 6
    assert gate.ucj_op.final_orbital_rotation is None


def test_ucj_unbalanced_reads_norb_and_modes_off_the_operator():
    """A spin-unbalanced operator gives a ``2 * norb``-mode gate with per-spin rotations."""
    op = ffsim.random.random_ucj_op_spin_unbalanced(
        3, n_reps=2, with_final_orbital_rotation=True, seed=1234
    )
    gate = UCJ(op)
    assert gate.norb == 3
    assert gate.num_modes == 6
    assert gate.ucj_op.orbital_rotations.shape == (2, 2, 3, 3)
    assert gate.ucj_op.final_orbital_rotation.shape == (2, 3, 3)


def test_ucj_spinless_uses_a_single_register():
    """A spinless operator gives a ``norb``-mode gate, not a two-register one."""
    gate = UCJ(_spinless_op(3, 2, seed=1234))
    assert gate.norb == 3
    assert gate.num_modes == 3


def test_ucj_rejects_a_non_ffsim_operator():
    """Anything other than one of ffsim's three UCJ operators is rejected up front."""
    with pytest.raises(TypeError, match="requires one of ffsim's UCJ operators"):
        UCJ(np.zeros((1, 2, 3, 3)))


def test_ucj_zero_reps_is_the_identity_ansatz():
    """A zero-repetition operator yields an empty (but valid) gate."""
    op = ffsim.UCJOpSpinBalanced(np.empty((0, 2, 3, 3)), np.empty((0, 3, 3)))
    gate = UCJ(op)
    assert gate.n_reps == 0
    assert gate.num_modes == 6
    circ = FermionicCircuit(gate.num_modes)
    circ.append(gate, circ.modes)
    assert circ.decompose().count_ops() == {}


def test_ucj_define_gate_sequence():
    """The gate definition is per-rep (rotation, evolution, rotation) + optional final rotation.

    UCJ is a pure unitary carrying no reference state, so its definition contains only the ansatz
    layers, with no opening :class:`.InitializeModes`.
    """
    n_reps = 2
    gate = UCJ(_balanced_op(3, n_reps, seed=7, with_final=True))

    circ = FermionicCircuit(gate.num_modes)
    circ.append(gate, circ.modes)
    ops = circ.decompose().count_ops()

    assert "InitializeModes" not in ops
    assert ops["Evolution"] == n_reps
    # per rep: U^dagger (2 local) + U (2 local) = 4; plus a final rotation (2 local)
    assert ops["OrbitalRotation"] == 4 * n_reps + 2


def test_ucj_spinless_definition_places_one_rotation_per_layer():
    """A spinless gate places a single rotation across all norb modes."""
    n_reps = 1
    gate = UCJ(_spinless_op(3, n_reps, seed=8))
    circ = FermionicCircuit(gate.num_modes)
    circ.append(gate, circ.modes)
    ops = circ.decompose().count_ops()
    assert "InitializeModes" not in ops
    assert ops["Evolution"] == n_reps
    # spinless places a single rotation on all norb modes: U^dagger + U = 2 per rep
    assert ops["OrbitalRotation"] == 2 * n_reps


def test_orbital_rotation_from_t1_amplitudes_is_unitary():
    """OrbitalRotation.from_t1_amplitudes builds a unitary of the right size."""
    t1 = np.array([[0.1, 0.2], [0.3, -0.1]])  # 2 occ, 2 virt
    gate = OrbitalRotation.from_t1_amplitudes(t1)
    assert isinstance(gate, OrbitalRotation)
    assert gate.num_modes == 4
    u = gate.rotation_unitary
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-12)


def _term_supports(operator):
    """Returns each term's support (its set of modes) in iteration order."""
    return [{mode for _, mode in term} for term, _ in operator.iter_terms()]


def _reference_diag_coulomb_terms(gate, diag_coulomb_mat):
    """The ungrouped ``{term: coeff}`` reference for one repetition's diagonal Coulomb operator."""
    norb = gate.norb
    mat_aa, mat_ab, mat_bb = gate._resolve_diag_coulomb_blocks(diag_coulomb_mat)
    if gate._spinless:
        blocks = {(0, 0): mat_aa}
    else:
        blocks = {(0, 0): mat_aa, (0, 1): mat_ab, (1, 0): mat_ab.T, (1, 1): mat_bb}
    terms: dict[tuple, complex] = {}
    for (sigma, tau), block in blocks.items():
        for i in range(norb):
            for j in range(norb):
                coeff = 0.5 * block[i, j]
                if coeff == 0.0:
                    continue
                mode_i, mode_j = sigma * norb + i, tau * norb + j
                key = ((True, mode_i), (False, mode_i), (True, mode_j), (False, mode_j))
                terms[key] = terms.get(key, 0.0) + coeff
    return terms


@pytest.mark.parametrize("norb", [1, 2, 3, 4, 5])
def test_diag_coulomb_grouping_preserves_operator(norb):
    """Grouping the diagonal Coulomb terms leaves the operator itself unchanged (balanced)."""
    from qiskit_fermions.operators import FermionOperator

    gate = UCJ(_balanced_op(norb, 1, seed=100 + norb))
    operator = gate._diag_coulomb_operator(gate.ucj_op.diag_coulomb_mats[0])

    reference = FermionOperator.from_dict(
        _reference_diag_coulomb_terms(gate, gate.ucj_op.diag_coulomb_mats[0])
    )
    assert operator.equiv(reference, 1e-12)


@pytest.mark.parametrize(
    ("spinless", "norb"),
    [
        (False, 4),
        (True, 5),
    ],
)
def test_diag_coulomb_groups_have_disjoint_support(spinless, norb):
    """Every diagonal Coulomb group's terms act on mutually disjoint modes (one parallel layer)."""
    op = _spinless_op(norb, 1, seed=7) if spinless else _balanced_op(norb, 1, seed=7)
    gate = UCJ(op)

    operator = gate._diag_coulomb_operator(gate.ucj_op.diag_coulomb_mats[0])
    assert operator.groups is not None
    for group in operator.split_out_groups():
        seen: set[int] = set()
        for support in _term_supports(group):
            assert not (support & seen), "group has overlapping term supports"
            seen |= support


@pytest.mark.parametrize(
    ("spinless", "norb"),
    [
        (False, 3),  # spinful sector: even num_modes = 2*norb
        (False, 4),
        (False, 5),
        (True, 3),  # odd num_modes = norb (odd)
        (True, 4),  # even num_modes = norb (even)
        (True, 5),  # odd num_modes = norb (odd)
    ],
)
def test_diag_coulomb_group_count_is_optimal(spinless, norb):
    """The group (layer) count matches the provable optimum of the closed-form circle coloring.

    Every term sharing a mode must land in a distinct group, so the count is bounded below by the
    maximum mode degree. For an even mode count that lower bound is achievable and hit exactly; for
    an odd mode count (an odd complete graph) the chromatic index is one higher and unavoidable. The
    circle-method coloring attains this optimum in both cases.
    """
    op = _spinless_op(norb, 1, seed=7) if spinless else _balanced_op(norb, 1, seed=7)
    gate = UCJ(op)
    operator = gate._diag_coulomb_operator(gate.ucj_op.diag_coulomb_mats[0])

    degree: dict[int, int] = {}
    for support in _term_supports(operator):
        for mode in support:
            degree[mode] = degree.get(mode, 0) + 1
    max_degree = max(degree.values())

    # even mode count -> optimum == max degree; odd -> the unavoidable max degree + 1
    optimum = max_degree if gate.num_modes % 2 == 0 else max_degree + 1
    assert operator.num_groups() == optimum
