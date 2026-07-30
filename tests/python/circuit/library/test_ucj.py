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

"""Structural tests for the UCJ ansatz gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import UCJ, OrbitalRotation

from ...utils import random_unitary


def _balanced_tensors(norb, n_reps, *, seed):
    """Returns random ``(diag_coulomb_mats, orbital_rotations)`` for the spin-balanced variant."""
    rng = np.random.default_rng(seed)
    mats = rng.standard_normal((n_reps, 2, norb, norb))
    mats = mats + mats.transpose(0, 1, 3, 2)  # symmetric aa/ab
    rotations = np.stack([random_unitary(norb, seed=seed + k) for k in range(n_reps)])
    return mats, rotations


def test_ucj_balanced_variant():
    """A ``(L, 2, norb, norb)`` diag-Coulomb tensor is accepted for the balanced variant."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 2, seed=0)
    gate = UCJ(norb, "balanced", mats, rotations)
    assert gate.num_modes == 2 * norb
    assert gate._variant is UCJ.Variant.BALANCED


def test_ucj_unbalanced_variant():
    """A ``(L, 3, norb, norb)`` diag-Coulomb tensor with paired rotations is accepted as unbalanced."""
    norb = 3
    n_reps = 2
    rng = np.random.default_rng(1)
    mats = rng.standard_normal((n_reps, 3, norb, norb))
    rotations = np.stack(
        [
            np.stack([random_unitary(norb, seed=10 + k), random_unitary(norb, seed=20 + k)])
            for k in range(n_reps)
        ]
    )
    gate = UCJ(norb, "unbalanced", mats, rotations)
    assert gate.num_modes == 2 * norb
    assert gate._variant is UCJ.Variant.UNBALANCED


def test_ucj_spinless_variant():
    """A ``(L, norb, norb)`` diag-Coulomb tensor is accepted for the spinless variant (norb modes)."""
    norb = 3
    n_reps = 2
    rng = np.random.default_rng(2)
    mats = rng.standard_normal((n_reps, norb, norb))
    rotations = np.stack([random_unitary(norb, seed=30 + k) for k in range(n_reps)])
    spinless = UCJ(norb, "spinless", mats, rotations)
    assert spinless.num_modes == norb
    assert spinless._variant is UCJ.Variant.SPINLESS


def test_ucj_rejects_unknown_variant():
    """An unrecognized variant string raises a ValueError naming the accepted variants."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 2, seed=0)
    with pytest.raises(ValueError, match="Unknown UCJ variant"):
        UCJ(norb, "nonsense", mats, rotations)


def test_ucj_rejects_inconsistent_shapes():
    """Tensor shapes inconsistent with norb or with each other raise ValueError."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 2, seed=3)
    with pytest.raises(ValueError, match="Inconsistent"):
        UCJ(norb + 1, "balanced", mats, rotations)  # norb mismatch


def test_ucj_rejects_balanced_shaped_tensors_for_spinless_variant():
    """Balanced-shaped tensors passed with variant='spinless' are rejected as inconsistent."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 2, seed=4)
    with pytest.raises(ValueError, match="Inconsistent"):
        UCJ(norb, "spinless", mats, rotations)


def test_ucj_define_gate_sequence():
    """The gate definition is per-rep (rotation, evolution, rotation) + optional final rotation.

    UCJ is a pure unitary carrying no reference state, so its definition contains only the ansatz
    layers -- no opening :class:`.InitializeModes`.
    """
    norb = 3
    n_reps = 2
    mats, rotations = _balanced_tensors(norb, n_reps, seed=7)
    final = random_unitary(norb, seed=99)
    gate = UCJ(norb, "balanced", mats, rotations, final_orbital_rotation=final)

    circ = FermionicCircuit(gate.num_modes)
    circ.append(gate, circ.modes)
    ops = circ.decompose().count_ops()

    assert "InitializeModes" not in ops
    assert ops["Evolution"] == n_reps
    # per rep: U^dagger (2 local) + U (2 local) = 4; plus a final rotation (2 local)
    assert ops["OrbitalRotation"] == 4 * n_reps + 2


def test_ucj_spinless_uses_norb_modes():
    """A spinless variant acts on norb modes with only same-spin terms."""
    norb = 3
    n_reps = 1
    rng = np.random.default_rng(8)
    mats = rng.standard_normal((n_reps, norb, norb))
    mats = mats + mats.transpose(0, 2, 1)
    rotations = np.stack([random_unitary(norb, seed=40)])
    gate = UCJ(norb, "spinless", mats, rotations)
    circ = FermionicCircuit(gate.num_modes)
    circ.append(gate, circ.modes)
    ops = circ.decompose().count_ops()
    assert "InitializeModes" not in ops
    assert ops["Evolution"] == n_reps
    # spinless places a single rotation on all norb modes: U^dagger + U = 2 per rep
    assert ops["OrbitalRotation"] == 2 * n_reps


def test_ucj_accepts_complex_diag_coulomb_with_zero_imaginary_part():
    """Complex-typed diag-Coulomb mats with a negligible imaginary part are coerced to real."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 1, seed=11)
    gate = UCJ(norb, "balanced", mats.astype(complex), rotations)
    assert gate.diag_coulomb_mats.dtype == np.float64
    np.testing.assert_allclose(gate.diag_coulomb_mats, mats)


def test_ucj_rejects_complex_diag_coulomb_with_nonzero_imaginary_part():
    """A genuinely complex diag-Coulomb matrix is rejected rather than silently truncated."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 1, seed=12)
    complex_mats = mats.astype(complex)
    complex_mats[0, 0, 0, 1] += 0.5j
    with pytest.raises(ValueError, match="imaginary part"):
        UCJ(norb, "balanced", complex_mats, rotations)


def test_ucj_from_t_amplitudes_rejects_tuple_n_reps_for_balanced():
    """A tuple n_reps is only valid for the unbalanced variant; balanced/spinless reject it."""
    nocc, nvrt = 1, 2
    t2 = np.zeros((nocc, nocc, nvrt, nvrt))
    with pytest.raises(ValueError, match="only valid for the 'unbalanced' variant"):
        UCJ.from_t_amplitudes((1, 1), t2, variant="balanced", n_reps=(2, 3))


def test_ucj_from_t_amplitudes_empty_interaction_pairs_zeros_layer():
    """An empty-list interaction_pairs zeros the whole diagonal-Coulomb block (unlike None)."""
    nocc, nvrt = 1, 2
    rng = np.random.default_rng(13)
    t2 = rng.standard_normal((nocc, nocc, nvrt, nvrt))
    t2 = t2 + t2.transpose(1, 0, 3, 2)  # a nontrivial symmetric t2 so the factorization is nonzero
    gate = UCJ.from_t_amplitudes(2, t2, variant="spinless", interaction_pairs=[])
    # [] allows no interactions -> every diagonal-Coulomb matrix is zeroed
    np.testing.assert_array_equal(gate.diag_coulomb_mats, 0.0)
    # None (no restriction) leaves the genuine factorization terms in place
    gate_none = UCJ.from_t_amplitudes(2, t2, variant="spinless", interaction_pairs=None)
    assert np.any(gate_none.diag_coulomb_mats != 0.0)


def _unbalanced_t2(nocc, nvrt, *, seed):
    """Returns nontrivial ``(t2aa, t2ab, t2bb)`` amplitudes for the unbalanced variant.

    The same-spin blocks are symmetrized so their (aa, bb) double factorizations are non-empty,
    which exercises the same-spin assembly path.
    """
    rng = np.random.default_rng(seed)

    def _sym(a):
        return a + a.transpose(1, 0, 3, 2)

    t2aa = _sym(rng.standard_normal((nocc, nocc, nvrt, nvrt)))
    t2ab = rng.standard_normal((nocc, nocc, nvrt, nvrt))
    t2bb = _sym(rng.standard_normal((nocc, nocc, nvrt, nvrt)))
    return t2aa, t2ab, t2bb


def test_ucj_from_t_amplitudes_rejects_unknown_variant():
    """An unrecognized variant string raises a ValueError naming the accepted variants."""
    t2 = np.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="Unknown UCJ variant"):
        UCJ.from_t_amplitudes((1, 1), t2, variant="nonsense")


def test_ucj_from_t_amplitudes_unbalanced_tuple_n_reps_and_interaction_pairs():
    """A tuple ``n_reps`` and per-block interaction_pairs drive the unbalanced factorization path.

    This exercises the ``(n_reps_ab, n_reps_same)`` tuple split, the per-block masking (symmetric
    aa/bb, non-symmetric ab), and the same-spin (aa, bb) assembly.
    """
    nocc, nvrt = 1, 2
    t2 = _unbalanced_t2(nocc, nvrt, seed=21)
    gate = UCJ.from_t_amplitudes(
        (1, 1),
        t2,
        variant="unbalanced",
        n_reps=(2, 1),
        interaction_pairs=([(0, 0)], [(0, 0)], [(0, 0)]),
    )
    assert gate._variant is UCJ.Variant.UNBALANCED
    norb = nocc + nvrt
    assert gate.diag_coulomb_mats.shape == (3, 3, norb, norb)
    # every diagonal-Coulomb block keeps only the whitelisted (0, 0) entry
    for block in gate.diag_coulomb_mats.reshape(-1, norb, norb):
        off_diagonal = block.copy()
        off_diagonal[0, 0] = 0.0
        np.testing.assert_array_equal(off_diagonal, 0.0)


def test_ucj_from_t_amplitudes_unbalanced_int_n_reps_truncates():
    """An integer ``n_reps`` smaller than the factorization truncates the unbalanced tensors."""
    nocc, nvrt = 1, 2
    t2 = _unbalanced_t2(nocc, nvrt, seed=22)
    full = UCJ.from_t_amplitudes((1, 1), t2, variant="unbalanced")
    assert full.diag_coulomb_mats.shape[0] >= 2
    truncated = UCJ.from_t_amplitudes((1, 1), t2, variant="unbalanced", n_reps=1)
    assert truncated.diag_coulomb_mats.shape[0] == 1
    assert truncated.orbital_rotations.shape[0] == 1


def test_ucj_from_t_amplitudes_unbalanced_int_n_reps_pads():
    """An integer ``n_reps`` larger than the factorization pads with identity/zero layers."""
    nocc, nvrt = 1, 2
    t2 = _unbalanced_t2(nocc, nvrt, seed=23)
    full = UCJ.from_t_amplitudes((1, 1), t2, variant="unbalanced")
    n_have = full.diag_coulomb_mats.shape[0]
    padded = UCJ.from_t_amplitudes((1, 1), t2, variant="unbalanced", n_reps=n_have + 2)
    assert padded.diag_coulomb_mats.shape[0] == n_have + 2
    # the padding layers are zero diagonal-Coulomb matrices and identity rotations
    np.testing.assert_array_equal(padded.diag_coulomb_mats[n_have:], 0.0)
    norb = nocc + nvrt
    for rot_pair in padded.orbital_rotations[n_have:]:
        for rot in rot_pair:
            np.testing.assert_allclose(rot, np.eye(norb), atol=1e-12)


def test_ucj_rejects_mismatched_repetition_counts():
    """Differing repetition counts between the diag-Coulomb and rotation tensors are rejected."""
    norb = 3
    mats = np.zeros((2, 2, norb, norb))  # 2 reps
    rotations = np.stack([random_unitary(norb, seed=50), random_unitary(norb, seed=51)])[
        :1
    ]  # 1 rep
    with pytest.raises(ValueError, match="same number of repetitions"):
        UCJ(norb, "balanced", mats, rotations)


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
    if gate._variant is UCJ.Variant.SPINLESS:
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
    """Grouping the diagonal Coulomb terms leaves the operator itself unchanged (balanced variant)."""
    from qiskit_fermions.operators import FermionOperator

    mats, rotations = _balanced_tensors(norb, 1, seed=100 + norb)
    gate = UCJ(norb, "balanced", mats, rotations)
    operator = gate._diag_coulomb_operator(gate.diag_coulomb_mats[0])

    reference = FermionOperator.from_dict(
        _reference_diag_coulomb_terms(gate, gate.diag_coulomb_mats[0])
    )
    assert operator.equiv(reference, 1e-12)


@pytest.mark.parametrize(
    ("variant", "norb"),
    [
        ("balanced", 4),
        ("spinless", 5),
    ],
)
def test_diag_coulomb_groups_have_disjoint_support(variant, norb):
    """Every diagonal Coulomb group's terms act on mutually disjoint modes (one parallel layer)."""
    rng = np.random.default_rng(hash((variant, norb)) % (2**32))
    if variant == "balanced":
        mats = rng.standard_normal((1, 2, norb, norb))
        mats = mats + mats.transpose(0, 1, 3, 2)
    else:
        mats = rng.standard_normal((1, norb, norb))
        mats = mats + mats.transpose(0, 2, 1)
    rotations = np.stack([random_unitary(norb, seed=7)])
    gate = UCJ(norb, variant, mats, rotations)

    operator = gate._diag_coulomb_operator(gate.diag_coulomb_mats[0])
    assert operator.groups is not None
    for group in operator.split_out_groups():
        seen: set[int] = set()
        for support in _term_supports(group):
            assert not (support & seen), "group has overlapping term supports"
            seen |= support


@pytest.mark.parametrize(
    ("variant", "norb"),
    [
        ("balanced", 3),  # spinful sector: even num_modes = 2*norb
        ("balanced", 4),
        ("balanced", 5),
        ("spinless", 3),  # odd num_modes = norb (odd)
        ("spinless", 4),  # even num_modes = norb (even)
        ("spinless", 5),  # odd num_modes = norb (odd)
    ],
)
def test_diag_coulomb_group_count_is_optimal(variant, norb):
    """The group (layer) count matches the provable optimum of the closed-form circle coloring.

    Every term sharing a mode must land in a distinct group, so the count is bounded below by the
    maximum mode degree. For an even mode count that lower bound is achievable and hit exactly; for
    an odd mode count (an odd complete graph) the chromatic index is one higher and unavoidable. The
    circle-method coloring attains this optimum in both cases.
    """
    rng = np.random.default_rng(hash((variant, norb)) % (2**32))
    if variant == "balanced":
        mats = rng.standard_normal((1, 2, norb, norb))
        mats = mats + mats.transpose(0, 1, 3, 2)
    else:
        mats = rng.standard_normal((1, norb, norb))
        mats = mats + mats.transpose(0, 2, 1)
    rotations = np.stack([random_unitary(norb, seed=7)])
    gate = UCJ(norb, variant, mats, rotations)
    operator = gate._diag_coulomb_operator(gate.diag_coulomb_mats[0])

    degree: dict[int, int] = {}
    for support in _term_supports(operator):
        for mode in support:
            degree[mode] = degree.get(mode, 0) + 1
    max_degree = max(degree.values())

    # even mode count -> optimum == max degree; odd -> the unavoidable max degree + 1
    optimum = max_degree if gate.num_modes % 2 == 0 else max_degree + 1
    assert operator.num_groups() == optimum
