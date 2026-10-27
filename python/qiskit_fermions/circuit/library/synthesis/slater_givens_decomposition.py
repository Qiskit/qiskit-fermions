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

"""The Givens decomposition of a Slater determinant preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qiskit_fermions.linalg import givens_decomposition_slater

from ._givens_factors import append_factor, givens_block

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit

    from ..prepare_slater_determinant import PrepareSlaterDeterminant

# Absolute tolerance below which a coefficient of an occupied orbital does not count towards its
# support. It matches ``EPSILON`` in ``crates/core/src/linalg/givens.rs`` (the "equivalent to zero"
# cutoff of the Givens sweep that each block is handed to), so a block is never split on a
# coefficient that the sweep would still treat as nonzero.
_SUPPORT_ATOL = 1e-10


def _split_mixing_blocks(
    orbital_coeffs: np.ndarray,
) -> list[tuple[list[int], list[int], np.ndarray]]:
    r"""Partition the occupied orbitals into disjoint-support blocks, one Givens sweep each.

    The leading-reference sweep of :func:`~qiskit_fermions.linalg.givens_decomposition_slater` seeds
    the occupation on the block's leading modes and transports it out to the physical support; when
    two occupied orbitals live on *disjoint* mode windows, the modes between their windows (and any
    empty leading/trailing modes) carry no occupation, and the sweep would spend bare swaps (a
    rotation with ``c = 0``) shuttling occupation across them. Synthesizing each disjoint cluster on
    its own contiguous mode window instead drops that inter-cluster transport entirely. An orbital
    that is a basis vector is the degenerate case of a window of a single mode, and needs no rotation
    at all.

    Blocks are formed by a classic interval merge on each orbital's **mode window**
    :math:`[\min\text{-support}, \max\text{-support}]`: sort by window start, and merge an orbital
    into the running block whenever its window overlaps. This *must* group on the mode-index window,
    not the raw support: two orbitals with disjoint support but nested or interleaved windows (e.g.
    support ``{0, 3}`` and ``{1, 2}``) must share a block, since the window ``[0, 3]`` spans the
    occupied modes ``1, 2`` -- splitting them would rotate across an occupied mode. The window merge
    guarantees each block occupies a **contiguous** window whose only occupied modes are the block's
    own swept modes, so every rotation within a block is between physically adjacent modes.

    Args:
        orbital_coeffs: the :math:`m \times n` matrix of occupied-orbital coefficients (orthonormal
            rows), i.e. ``rotation_unitary[:, occupied].T``.

    Returns:
        One 3-tuple ``(block_rows, block_modes, sub)`` per block, where ``block_rows`` are the row
        indices of ``orbital_coeffs`` in the block, ``block_modes`` are the modes spanning the
        block's contiguous window (in ascending order; the first ``len(block_rows)`` of them are the
        block's leading reference modes), and ``sub`` is the C-contiguous
        ``len(block_rows) x len(block_modes)`` sub-matrix ready to hand to
        :func:`~qiskit_fermions.linalg.givens_decomposition_slater`. There are none if no orbital is
        occupied.
    """
    num_orbitals = orbital_coeffs.shape[0]
    if num_orbitals == 0:
        return []

    # each occupied orbital's support window [lo, hi]
    windows: list[tuple[int, int]] = []
    for row in range(num_orbitals):
        support = np.nonzero(np.abs(orbital_coeffs[row]) > _SUPPORT_ATOL)[0]
        windows.append((int(support.min()), int(support.max())))

    # interval merge: sort orbitals by window start, coalesce overlapping windows into one block
    order = sorted(range(num_orbitals), key=lambda row: windows[row][0])
    grouped_rows: list[list[int]] = [[order[0]]]
    running_hi = windows[order[0]][1]
    for row in order[1:]:
        lo, hi = windows[row]
        if lo <= running_hi:
            grouped_rows[-1].append(row)
            running_hi = max(running_hi, hi)
        else:
            grouped_rows.append([row])
            running_hi = hi

    blocks: list[tuple[list[int], list[int], np.ndarray]] = []
    for block_rows in grouped_rows:
        window_lo = min(windows[row][0] for row in block_rows)
        window_hi = max(windows[row][1] for row in block_rows)
        block_modes = list(range(window_lo, window_hi + 1))
        # ``np.ascontiguousarray`` guards the FFI boundary: a sliced array is not guaranteed to be
        # C-contiguous, and ``givens_decomposition_slater`` requires a contiguous array.
        sub = np.ascontiguousarray(orbital_coeffs[np.ix_(block_rows, block_modes)])
        blocks.append((block_rows, block_modes, sub))
    return blocks


class FermionicSlaterGivensDecomposition:
    r"""The decomposition of a Slater determinant preparation into Givens rotations.

    This exploits the known reference occupation to synthesize the gate with the rectangular
    :func:`~qiskit_fermions.linalg.givens_decomposition_slater`: only the :math:`m` occupied orbitals
    of the rotation need be realized, so at most :math:`m (n - m)` rotations are emitted (versus the
    :math:`n (n - 1) / 2` rotations plus :math:`n` phases that :class:`.FermionicGivensDecomposition`
    needs for the full :class:`.OrbitalRotation`). The reduced decomposition carries no diagonal
    phases (a global phase and any rotation within the occupied space leave the prepared Slater
    determinant unchanged), so the state is prepared correct up to a global phase.

    The result is an :class:`.InitializeModes` that sets the reference, followed by one atomic
    :class:`.Evolution` per rotation (see :class:`.FermionicGivensDecomposition` for how a rotation
    becomes one). Being fermionic, the circuit needs no knowledge of how the modes are encoded.

    .. note::
       :func:`~qiskit_fermions.linalg.givens_decomposition_slater` is defined against a fixed
       *leading*-:math:`m` reference (the first :math:`m` modes occupied). The :class:`.InitializeModes`
       therefore seeds the leading modes of each decomposed block, **not** the modes pointed to by
       ``occupation`` -- the rotations then transport the occupation onto the physical target
       orbitals. The positions of ``occupation``'s occupied modes enter only through which columns of
       ``rotation_unitary`` are selected (``rotation_unitary[:, occupied]``); their *count*
       :math:`m` determines the reference. The final prepared state is correct for any occupation,
       contiguous or not. This is also why the circuit is only equivalent to the gate `as a state
       preparation from the vacuum`.

    .. note::
       The occupied orbitals are partitioned into disjoint-support **blocks**, each synthesized on
       its own contiguous mode window. When they split into several clusters separated by empty (or
       merely occupied) modes, this drops the bare swaps (rotations with ``c = 0``) that a single
       leading-reference sweep would otherwise spend on the gaps between clusters, and an orbital
       that is a basis vector needs no rotation at all. A single cluster spanning all modes is one
       block and reproduces the plain leading-reference sweep exactly. Every rotation is between
       adjacent modes, so each is a hop between adjacent qubits under Jordan-Wigner, which routes
       cheaply on a constrained coupling map.
    """

    def __init__(self, *, atol: float = 1e-12) -> None:
        """Initializing an instance of this synthesis method can be done with the arguments below.

        Args:
            atol: the absolute tolerance below which factors and entries are dropped. This is
                applied to the entries of the generator of a rotation, see
                :class:`.FermionicGivensDecomposition`.
        """
        self.atol = atol
        """The magnitude below which entries of a generator are dropped."""

    def synthesize(self, gate: PrepareSlaterDeterminant) -> FermionicCircuit:
        """Synthesizes the gate into the reference and the mesh of atomic :class:`.Evolution` gates.

        See the class documentation for the decomposition this implements.

        Args:
            gate: the gate to synthesize.

        Returns:
            A :class:`.FermionicCircuit` holding an :class:`.InitializeModes` of the reference
            (absent if there is nothing to occupy), followed by one :class:`.Evolution` gate per
            non-trivial rotation, in the order in which they are applied.
        """
        # deferred imports: `qiskit_fermions.circuit` imports this subpackage's siblings at module
        # load
        from qiskit_fermions.circuit import FermionicCircuit

        from ..initialize_modes import InitializeModes

        definition = FermionicCircuit(gate.num_modes)

        # the occupied local modes select the occupied orbitals; their columns of the rotation
        # unitary (following a†_i ↦ Σ_j U_ji a†_j) span the prepared Slater determinant's occupied
        # space. ``givens_decomposition_slater`` takes them as the m rows of an m x n matrix.
        occupied = np.nonzero(gate.occupation)[0]
        blocks = _split_mixing_blocks(gate.rotation_unitary[:, occupied].T)

        # the reference: the leading modes of every block, each of which the block's reduced
        # decomposition prepares its orbitals from (``sub`` has ``len(block_rows)`` rows). The
        # rotations then move the occupation onto the physical target orbitals within the block's
        # window.
        reference = np.zeros(gate.num_modes, dtype=bool)
        for block_rows, block_modes, _ in blocks:
            reference[np.asarray(block_modes[: len(block_rows)], dtype=int)] = True
        if reference.any():
            definition.append(InitializeModes(reference.tolist()), definition.modes)

        # apply each block's Givens rotations in order, remapping each block-local sweep index
        # through ``block_modes`` to its physical mode. No phases: the reduced decomposition carries
        # none, so the state is prepared up to a global phase (physically irrelevant for state
        # preparation).
        for _, block_modes, sub in blocks:
            for c, s, i, j in givens_decomposition_slater(sub):
                append_factor(
                    definition, givens_block(c, s), (block_modes[i], block_modes[j]), atol=self.atol
                )

        return definition
