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

"""Linear algebra over GF(2) for the ``cz-local`` candidate.

For fixed single-qubit layers the CZ graph that diagonalizes a group is the solution of a linear
system over GF(2); the hardware-tailored diagonalization of Miller et al. (arXiv:2203.03646). The
functions here solve that system and pick, among its solutions, the one with the lightest images.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._symplectic import apply_cz

CHUNK = 32
"""Nullspace vectors scored together in one numpy call by :func:`descend_nullspace`."""


def unique_rows_any_order(matrix: np.ndarray) -> np.ndarray:
    """The distinct rows of a 0/1 matrix, in order of first occurrence.

    The reduced row echelon form of a system does not depend on the order of its rows, so duplicates
    can be dropped.
    """
    if matrix.shape[0] <= 1:
        return matrix
    packed = np.ascontiguousarray(np.packbits(matrix, axis=1))
    void = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, idx = np.unique(void, return_index=True)
    return matrix[np.sort(idx)]


def solve_packed(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Solves ``a v = b`` over GF(2) with the rows packed into 64-bit words.

    Gauss-Jordan elimination reaches the unique reduced row echelon form, so the result does not
    depend on the order of the rows.

    Args:
        a: the coefficient matrix (0/1).
        b: the right-hand side (0/1).

    Returns:
        ``(particular solution, nullspace)`` with the nullspace as a 2D array (one basis vector per
        free column), or ``None`` if the system is inconsistent.
    """
    rows, cols = a.shape
    aug = np.concatenate([a, b[:, None]], axis=1).astype(np.uint8)
    nbytes = ((cols + 1 + 63) // 64) * 8
    packed = np.packbits(aug, axis=1, bitorder="little")
    pad = np.zeros((rows, nbytes), np.uint8)
    pad[:, : packed.shape[1]] = packed
    words = pad.view(np.uint64).copy()
    one = np.uint64(1)
    pivots: list[int] = []
    r = 0
    for c in np.nonzero(aug[:, :cols].any(axis=0))[
        0
    ]:  # a column without any 1 can never be a pivot
        if r >= rows:
            break
        w, s = int(c) >> 6, np.uint64(int(c) & 63)
        hit = np.flatnonzero((words[r:, w] >> s) & one)
        if hit.size == 0:
            continue
        p = r + int(hit[0])
        if p != r:
            words[[r, p]] = words[[p, r]]
        mask = ((words[:, w] >> s) & one).astype(bool)
        mask[r] = False
        if mask.any():
            words[mask] ^= words[r]
        pivots.append(int(c))
        r += 1
    if ((words[r:, cols >> 6] >> np.uint64(cols & 63)) & one).any():
        return None
    reduced = np.unpackbits(words[:r].view(np.uint8), axis=1, bitorder="little")[:, : cols + 1]
    sol = np.zeros(cols, np.uint8)
    piv = np.array(pivots, int)
    sol[piv] = reduced[:, cols]
    is_pivot = np.zeros(cols, bool)
    is_pivot[piv] = True
    free = np.flatnonzero(~is_pivot)
    null = np.zeros((free.size, cols), np.uint8)
    null[np.arange(free.size), free] = 1
    if r:
        null[:, piv] = reduced[:, free].T
    return sol, null


def descend_nullspace(
    sol: np.ndarray,
    null: np.ndarray,
    x1: np.ndarray,
    z1: np.ndarray,
    edges: Sequence[tuple[int, int]],
    num_edges: int,
) -> np.ndarray:
    """First-improvement descent over the nullspace, with candidates scored in numpy chunks.

    The objective is lexicographic: (largest image weight, total image weight, number of CZ gates).
    Candidates are visited in the order of the nullspace basis and the first improvement is taken.
    The image change caused by one nullspace vector is precomputed once, so a candidate costs one
    XOR instead of a rebuild of the whole image.

    Args:
        sol: a particular solution, the CZ pattern followed by the free letter bits.
        null: the nullspace basis, one vector per row.
        x1: the X bits of the terms after the single-qubit layer.
        z1: the Z bits of the terms after the single-qubit layer.
        edges: the allowed CZ pairs.
        num_edges: ``len(edges)``.

    Returns:
        The improved solution.
    """
    count = null.shape[0]
    k, n = x1.shape
    if count == 0:
        return sol
    delta = np.zeros((count, k, n), np.uint8)
    xf = x1.astype(np.float32)
    ed = np.array(edges, int).reshape(-1, 2)
    edge_part = null[:, :num_edges].astype(np.float32)
    for q in range(n):
        first, second = ed[:, 0] == q, ed[:, 1] == q
        ids = np.flatnonzero(first | second)
        if ids.size == 0:
            continue
        other = np.where(first[ids], ed[ids, 1], ed[ids, 0])
        delta[:, :, q] = (edge_part[:, ids] @ xf[:, other].T).astype(np.int64) & 1
    zp = z1.copy()
    for i in np.flatnonzero(sol[:num_edges]):
        apply_cz(x1, zp, *edges[int(i)])  # writes only into zp
    scale_total, scale_cz = k * n + 1, num_edges + 1

    def rank_key(zp_c: np.ndarray, ncz) -> np.ndarray:
        wt = (x1 | zp_c).sum(axis=-1, dtype=np.int64)
        key: np.ndarray = (wt.max(axis=-1) * scale_total + wt.sum(axis=-1)) * scale_cz + ncz
        return key

    best = int(rank_key(zp, int(sol[:num_edges].sum())))
    improved = True
    while improved:
        improved = False
        start = 0
        while start < count:
            stop = min(count, start + CHUNK)
            keys = rank_key(
                zp[None] ^ delta[start:stop],
                (sol[:num_edges][None] ^ null[start:stop, :num_edges]).sum(axis=1, dtype=np.int64),
            )
            hit = np.flatnonzero(keys < best)
            if hit.size == 0:
                start = stop
                continue
            j = start + int(hit[0])
            sol, zp, best, improved = sol ^ null[j], zp ^ delta[j], int(keys[hit[0]]), True
            start = j + 1
    return sol


def incidence(
    num_qubits: int, edges: Sequence[tuple[int, int]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per qubit, the arrays (ids of the incident edges, neighboring qubits)."""
    ids: list[list[int]] = [[] for _ in range(num_qubits)]
    nbr: list[list[int]] = [[] for _ in range(num_qubits)]
    for i, (u, v) in enumerate(edges):
        ids[u].append(i)
        nbr[u].append(v)
        ids[v].append(i)
        nbr[v].append(u)
    return [(np.array(a, int), np.array(b, int)) for a, b in zip(ids, nbr, strict=True)]


def solve_gamma_joint(
    x1: np.ndarray,
    z1: np.ndarray,
    edges: Sequence[tuple[int, int]],
    inc: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[tuple[int, int]], list[int]] | None:
    """The fewest-edge CZ graph and the final letters, solved together.

    Qubits that never carry an X keep the letter Z (no condition); every other qubit keeps X or Y,
    with a free bit ``t_q`` (0 = X, 1 = Y), so the CZ graph and the letters come out of one GF(2)
    solve.

    Args:
        x1: the X bits of the terms after the single-qubit layer.
        z1: the Z bits after the layer.
        edges: the allowed CZ pairs.
        inc: ``incidence(n, edges)``.

    Returns:
        ``(gamma, letters)`` with the letters 0 = X, 1 = Y, 2 = Z, or ``None`` if there is no
        solution.
    """
    k, n = x1.shape
    num_edges = len(edges)
    free = [q for q in range(n) if x1[:, q].any()]
    letters = [2] * n
    chosen = np.zeros(num_edges, np.uint8)
    if free:
        nfree = len(free)
        blocks = np.zeros((nfree, k, num_edges + nfree), np.uint8)
        for i, q in enumerate(free):
            ids, nb = inc[q]
            if len(ids):
                blocks[i][:, ids] = x1[:, nb]
            blocks[i][:, num_edges + i] = x1[:, q]
        rhs_all = np.stack([z1[:, q] for q in free])
        system = np.concatenate(
            [blocks.reshape(nfree * k, num_edges + nfree), rhs_all.reshape(nfree * k, 1)], axis=1
        )
        system = unique_rows_any_order(system)
        system = system[system.any(axis=1)]
        res = solve_packed(system[:, :-1], system[:, -1])
        if res is None:
            return None
        sol, null = res
        sol = descend_nullspace(sol, null, x1, z1, edges, num_edges)
        chosen = sol[:num_edges]
        for i, q in enumerate(free):
            letters[q] = int(sol[num_edges + i])
    return [edges[int(i)] for i in np.nonzero(chosen)[0]], letters


def cooccurring_edges(
    x: np.ndarray, z: np.ndarray, edges: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """The edges whose two qubits lie in the support of one common term.

    A CZ on a pair that no term touches together only adds Z support to terms that carry an X on one
    of the two qubits, so it cannot lower an image weight. It could still matter for feasibility by
    cancelling other edges in the linear system; this was checked empirically, not proven, so
    callers fall back to the full edge set.
    """
    support = (x | z).astype(np.int32)
    together = (support.T @ support) > 0
    return [e for e in edges if together[e[0], e[1]]]
