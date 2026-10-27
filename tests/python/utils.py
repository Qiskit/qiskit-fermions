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

"""Testing utilities."""

import itertools

import numpy as np


def random_unitary(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    z += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def random_two_body_tensor(dim: int, *, rank: int | None = None, seed=None) -> np.ndarray:
    """Generates a random real two-body tensor with the standard two-body symmetries.

    Mirrors ``ffsim.random.random_two_body_tensor`` with ``dtype=float``: it builds the tensor as a
    sum of outer products of symmetric "Cholesky" matrices, which makes the reshaped
    ``(dim**2, dim**2)`` matrix real symmetric positive semidefinite and hence exactly
    double-factorizable by both the Cholesky and eigendecomposition paths.
    """
    rng = np.random.default_rng(seed)
    if rank is None:
        rank = dim * (dim + 1) // 2
    cholesky_vecs = rng.standard_normal((rank, dim, dim))
    cholesky_vecs += cholesky_vecs.transpose((0, 2, 1))
    return np.einsum("ipr,iqs->prqs", cholesky_vecs, cholesky_vecs)


def antisymmetrized_t2(t2: np.ndarray) -> np.ndarray:
    """Returns a same-spin ``t2`` block antisymmetrized in the occupied and in the virtual indices.

    Newer ffsim versions validate this when constructing an unrestricted UCCSD operator. It implies
    the simultaneous-exchange symmetry ``t2[i, j, a, b] == t2[j, i, b, a]`` the cluster operator
    sees, so the result is valid for older ffsim versions too.
    """
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    return 0.5 * (t2 - t2.transpose(0, 1, 3, 2))


def random_t2_amplitudes(norb: int, nocc: int, *, seed=None) -> np.ndarray:
    """Generates random spin-restricted ``t2`` amplitudes.

    Mirrors ``ffsim.random.random_t2_amplitudes`` with ``dtype=float``: the amplitudes satisfy the
    restricted-CCSD symmetry ``t2[i, j, a, b] == t2[j, i, b, a]``, which makes the corresponding
    reshaped matrix real symmetric and exactly representable by the explicit factorization.
    """
    rng = np.random.default_rng(seed)
    nvrt = norb - nocc
    t2 = np.zeros((nocc, nocc, nvrt, nvrt))
    pairs = itertools.product(range(nocc), range(nocc, norb))
    for (i, a), (j, b) in itertools.combinations_with_replacement(pairs, 2):
        val = rng.standard_normal()
        t2[i, j, a - nocc, b - nocc] = val
        t2[j, i, b - nocc, a - nocc] = val
    return t2


def reference_pass_manager(passes=(), *, slater_synthesis=None, **kwargs):
    """The Jordan-Wigner preset with its ``optimization`` stage overwritten.

    The stage holds the given ``passes`` followed by the decomposition of the orbital rotations and
    Slater determinant preparations, which every such gate needs to reach the qubit stage. With no
    ``passes`` nothing is fused, so an ``InitializeModes`` followed by an ``OrbitalRotation`` stays
    two gates: the trusted square path to compare a Slater-determinant preparation against.
    Everything else is the preset's own, so the reference cannot drift away from it.

    ``slater_synthesis`` is applied to every ``PrepareSlaterDeterminant`` in place of the synthesis
    of the gate itself.
    """
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.passes import F2FSynthesis
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    pm = generate_preset_jw_pass_manager(**kwargs)
    config = {"OrbitalRotation": "Givens", "PrepareSlaterDeterminant": slater_synthesis or "Givens"}
    pm.optimization = FermionicPassManager([*passes, F2FSynthesis(config)])
    return pm


# --- Slater determinant targets --------------------------------------------------


def givens_2mode(theta: float, p: int, q: int, num_modes: int) -> np.ndarray:
    """A real orbital rotation mixing only modes ``p`` and ``q`` (identity elsewhere).

    Used to build occupied spaces that factor into disjoint unit-orbitals plus a genuinely-mixing
    block.
    """
    rot = np.eye(num_modes, dtype=complex)
    rot[p, p] = np.cos(theta)
    rot[p, q] = -np.sin(theta)
    rot[q, p] = np.sin(theta)
    rot[q, q] = np.cos(theta)
    return rot


def multi_block_rotation(num_modes: int, blocks: list[list[int]], *, seed: int) -> np.ndarray:
    """Identity except for an independent random unitary on each mode group in ``blocks``.

    Each group may span arbitrary -- possibly non-contiguous -- modes. Placing several genuinely-mixing
    clusters on disjoint mode windows (with empty modes between them) builds the targets that
    block-splitting exists to handle: each cluster synthesizes on its own contiguous window instead of
    being transported across the whole line by a single leading sweep. A single non-contiguous group
    (e.g. ``[[0, 2]]``) mixes two modes with another mode in between, which a rotation between them
    has to pass.
    """
    rotation = np.eye(num_modes, dtype=complex)
    for offset, modes in enumerate(blocks):
        block = random_unitary(len(modes), seed=seed + offset)
        for a, mode_a in enumerate(modes):
            for b, mode_b in enumerate(modes):
                rotation[mode_a, mode_b] = block[a, b]
    return rotation


SLATER_STRUCTURE_CASES = [
    # disjoint multi-mode blocks {0,1} and {5,6} on an 8-line, one orbital occupied per block
    (
        "disjoint_multimode_blocks",
        8,
        [0, 5],
        lambda: multi_block_rotation(8, [[0, 1], [5, 6]], seed=11),
    ),
    # two adjacent multi-mode blocks {0,1} and {2,3} (windows touch but do not overlap)
    (
        "adjacent_multimode_blocks",
        6,
        [0, 2],
        lambda: multi_block_rotation(6, [[0, 1], [2, 3]], seed=13),
    ),
    # nested windows: mixing {0,3} and mixing {1,2} -- window [0,3] spans modes 1,2 so they MUST merge
    ("nested_windows", 4, [0, 1], lambda: multi_block_rotation(4, [[0, 3], [1, 2]], seed=17)),
    # interleaved windows: mixing {0,4} and mixing {1,5} -- windows overlap, must merge into one block
    ("interleaved_windows", 6, [0, 1], lambda: multi_block_rotation(6, [[0, 4], [1, 5]], seed=19)),
    # a single dense mixing block spanning every mode (one contiguous window, reproduces leading)
    ("dense_single_block", 5, [0, 1, 2], lambda: random_unitary(5, seed=23)),
    # a pure permutation: identity rotation, scattered occupation -- every orbital is a basis vector
    ("pure_permutation", 6, [1, 3, 4], lambda: np.eye(6, dtype=complex)),
    # a basis vector on the interior mode 1 of a mixing pair {0,2}: the rotation passes an occupied mode
    ("interior_unit_orbital", 3, [0, 1], lambda: multi_block_rotation(3, [[0, 2]], seed=3)),
]


def draw_multi_block_target(rng):
    """Draw a random multi-block occupied-space target.

    Returns ``(num_modes, occupation, rotation, is_multi_block)`` where the flag records whether the
    drawn target genuinely stresses block-splitting (more than one cluster of occupied orbitals on
    disjoint windows). ``None`` is returned for a degenerate draw (empty or full occupation).
    """
    from qiskit_fermions.circuit.library.synthesis.slater_givens_decomposition import (
        _split_mixing_blocks,
    )

    num_modes = int(rng.integers(4, 9))

    # lay one or two mixing clusters out left-to-right on *contiguous* mode ranges separated by
    # random gaps: this keeps their windows genuinely disjoint (exercising block-splitting) while the
    # gap modes host scattered unit-orbitals, which an occupied orbital of a cluster can then sit
    # between.
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
    # collapse to a permutation), plus some isolated units
    occupied_columns: list[int] = []
    for block_modes in blocks:
        num_occ = int(rng.integers(1, len(block_modes)))
        occupied_columns += block_modes[:num_occ]
    occupied_columns += [mode for mode in rest if rng.random() < 0.5]

    occupation = [i in occupied_columns for i in range(num_modes)]
    if not any(occupation) or all(occupation):
        return None

    orbital_coeffs = rotation[:, np.nonzero(occupation)[0]].T
    num_clusters = len(_split_mixing_blocks(orbital_coeffs))
    return num_modes, occupation, rotation, num_clusters > 1
