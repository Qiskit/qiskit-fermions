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

"""Slater determinant preparation synthesis."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import XGate, XXPlusYYGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit_fermions.linalg import givens_decomposition_slater

from ... import F2QLayout
from ..utils import map_node_single_register

# Absolute tolerance for treating an O(1) quantity derived from the orthonormal orbital
# coefficients as equal to its ideal 0/1 value. It gates both halves of the same peeling decision:
# whether a projector column ``P[:, k]`` equals ``e_k`` (identifying a disjoint unit-orbital), and
# whether a singular value of the restricted matrix is nonzero (counting the reduced-space rank).
# Both compare O(1) quantities against a round-off floor of a few ULPs (~1e-15, growing only with the
# tiny mode count), so a single tolerance well above that floor -- and identical for both tests, so
# they can never disagree on a borderline mode -- is the sensible choice. The value matches
# ``EPSILON`` in ``crates/core/src/linalg/givens.rs`` (the "equivalent to zero" cutoff of the Givens
# sweep that ``reduced`` is handed to): keeping the peeler no looser than that downstream floor means
# it never declares a mode a clean unit-orbital while the sweep would still treat the residual as
# nonzero.
_PEEL_ATOL = 1e-10


def _peel_disjoint_unit_orbitals(
    orbital_coeffs: np.ndarray,
) -> tuple[list[int], list[int], np.ndarray]:
    r"""Split the occupied space into disjoint unit-orbitals and a genuinely-mixing remainder.

    An occupied orbital that is a basis vector on a mode disjoint from every other occupied orbital
    contributes a single fixed occupation on that mode. The reduced Givens sweep would otherwise seed
    it on a leading mode and transport it out with a chain of bare ``SWAP``\ s
    (``XXPlusYYGate`` with ``c = 0``); such orbitals can instead be realized by placing an ``XGate``
    directly on their target mode, and excluded from the decomposition entirely.

    The test is basis-independent: with the :math:`m \times n` occupied-orbital matrix ``A``
    (``orbital_coeffs``), the occupied-space projector is :math:`P = A^\dagger A` (an :math:`n \times
    n` rank-:math:`m` projector). Mode :math:`k` is a disjoint unit-orbital iff column :math:`k` of
    :math:`P` equals :math:`e_k` -- i.e. :math:`e_k` lies in the occupied space *and* is orthogonal to
    the rest of it.

    Args:
        orbital_coeffs: the :math:`m \times n` matrix of occupied-orbital coefficients (orthonormal
            rows), i.e. ``rotation_unitary[:, occupied].T``.

    Returns:
        A 3-tuple ``(peel_modes, kept_modes, reduced)`` where ``peel_modes`` are the mode indices to
        seed with a direct ``XGate``, ``kept_modes`` are the remaining mode indices (spanning the
        genuinely-mixing occupied space), and ``reduced`` is the :math:`(m - p) \times k` orthonormal
        matrix of that mixing space restricted to ``kept_modes`` (:math:`p` peeled modes,
        :math:`k = n - p` kept modes; its :math:`m - p` rows are an orthonormal basis of the
        kept-mode occupied subspace) -- ready to hand to
        :func:`~qiskit_fermions.linalg.givens_decomposition_slater`. When no mode peels, ``peel_modes``
        is empty, ``kept_modes`` is ``range(n)`` and ``reduced`` is ``orbital_coeffs`` unchanged.
    """
    n = orbital_coeffs.shape[1]
    projector = orbital_coeffs.conj().T @ orbital_coeffs
    identity = np.eye(n)
    peel_modes = [k for k in range(n) if np.allclose(projector[:, k], identity[k], atol=_PEEL_ATOL)]

    if not peel_modes:
        return peel_modes, list(range(n)), orbital_coeffs

    kept_modes = [k for k in range(n) if k not in peel_modes]
    # Restrict the occupied space to the ``k = n - p`` kept modes and take an orthonormal basis of
    # what remains. An SVD is used rather than dropping "the peeled rows" directly: after in-space
    # mixing a peeled orbital need not appear as a clean single-support row of ``orbital_coeffs``, so
    # we reduce the subspace rather than individual rows. The ``rank`` right-singular vectors with
    # nonzero singular value span the kept-mode occupied space with orthonormal rows; peeling ``p``
    # disjoint unit-orbitals drops ``p`` dimensions, so ``rank == m - p`` and ``reduced`` is
    # ``(m - p) x k``.
    restricted = orbital_coeffs[:, kept_modes]
    _, singular_values, right_vectors = np.linalg.svd(restricted, full_matrices=False)
    rank = int((singular_values > _PEEL_ATOL).sum())
    # ``np.ascontiguousarray`` guards the FFI boundary: the sliced SVD output is not guaranteed
    # C-contiguous, and ``givens_decomposition_slater`` requires a contiguous array.
    reduced = np.ascontiguousarray(right_vectors[:rank])
    return peel_modes, kept_modes, reduced


def _split_mixing_blocks(
    reduced: np.ndarray,
    kept_modes: list[int],
) -> list[tuple[list[int], list[int], np.ndarray]]:
    r"""Partition the mixing remainder into disjoint-support blocks, one Givens sweep each.

    The leading-reference sweep of :func:`~qiskit_fermions.linalg.givens_decomposition_slater` seeds
    the occupation on the block's leading modes and transports it out to the physical support; when
    two mixing orbitals live on *disjoint* mode windows, the modes between their windows (and any
    empty leading/trailing modes) carry no occupation, and the sweep would spend bare ``SWAP``\ s
    (``XXPlusYYGate`` with ``c = 0``) shuttling occupation across them. Synthesizing each disjoint
    cluster on its own contiguous mode window instead drops that inter-cluster transport entirely.
    (This complements the disjoint-unit-orbital peeling of :func:`_peel_disjoint_unit_orbitals`, which
    is the degenerate single-mode-window case and is handled *before* this split -- so ``reduced`` is
    the genuinely-mixing remainder only.)

    Blocks are formed by a classic interval merge on each orbital's **physical-mode window**
    :math:`[\min\text{-support}, \max\text{-support}]` (mapped through ``kept_modes``): sort by window
    start, and merge an orbital into the running block whenever its window overlaps. This *must* group
    on the mode-index window, not the raw support: two orbitals with disjoint support but
    nested/interleaved windows (e.g. support ``{0, 3}`` and ``{1, 2}``) must share a block, since the
    window ``[0, 3]`` spans the occupied modes ``1, 2`` -- splitting them would emit a rotation across
    an occupied mode. The window merge guarantees each block occupies a **contiguous** window whose
    only occupied modes are the block's own swept modes, so every emitted rotation is physically
    mode-adjacent within the block and the framework's no-Z-string hop recipe stays valid. (A
    *peeled* mode may still fall inside a block's window; that Z-string is compensated at emission
    time by the caller -- see :meth:`GivensDecompositionSlaterDeterminantSynthesis.run`.)

    Args:
        reduced: the :math:`(m - p) \times k` orthonormal mixing-remainder matrix from
            :func:`_peel_disjoint_unit_orbitals` (rows in kept-mode-local column space).
        kept_modes: the length-:math:`k` list mapping each ``reduced`` column to its physical mode.

    Returns:
        One 3-tuple ``(block_rows, block_modes, sub)`` per block, where ``block_rows`` are the
        ``reduced`` row indices in the block, ``block_modes`` are the physical modes spanning the
        block's contiguous window (in ascending order; the first ``len(block_rows)`` of them are the
        block's leading reference modes), and ``sub`` is the C-contiguous
        ``len(block_rows) x len(block_modes)`` sub-matrix ready to hand to
        :func:`~qiskit_fermions.linalg.givens_decomposition_slater`.
    """
    num_mixing = reduced.shape[0]

    # each mixing orbital's physical-mode support window [lo, hi]
    windows: list[tuple[int, int]] = []
    for row in range(num_mixing):
        support = [kept_modes[col] for col in np.nonzero(np.abs(reduced[row]) > _PEEL_ATOL)[0]]
        windows.append((min(support), max(support)))

    # interval merge: sort orbitals by window start, coalesce overlapping windows into one block
    order = sorted(range(num_mixing), key=lambda row: windows[row][0])
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
        # the kept-local columns and physical modes spanning this block's contiguous window
        local_cols = [
            col for col in range(len(kept_modes)) if window_lo <= kept_modes[col] <= window_hi
        ]
        block_modes = [kept_modes[col] for col in local_cols]
        # ``np.ascontiguousarray`` guards the FFI boundary (see ``_peel_disjoint_unit_orbitals``).
        sub = np.ascontiguousarray(reduced[np.ix_(block_rows, local_cols)])
        blocks.append((block_rows, block_modes, sub))
    return blocks


class GivensDecompositionSlaterDeterminantSynthesis:
    r"""A :class:`.F2QSynthesisPlugin` for transpiling :class:`.PrepareSlaterDeterminant`.

    This plugin exploits the known reference occupation to synthesize the gate with the rectangular
    :func:`~qiskit_fermions.linalg.givens_decomposition_slater`: only the :math:`m` occupied orbitals
    of the rotation need be realized, so at most :math:`m (n - m)` ``XXPlusYYGate`` rotations are
    emitted (versus the :math:`n (n - 1) / 2` brick-wall plus :math:`n` phase gates that
    :class:`.GivensDecompositionOrbitalRotationSynthesis` would use for the full
    :class:`.OrbitalRotation`). The reduced decomposition carries no diagonal phases -- a global phase
    and any rotation within the occupied space leave the prepared Slater determinant unchanged -- so
    the state is prepared correct up to a global phase.

    .. note::
       :func:`~qiskit_fermions.linalg.givens_decomposition_slater` is defined against a fixed
       *leading*-:math:`m` reference (the first :math:`m` modes occupied). The emitted ``XGate``\ s
       therefore seed the leading modes of the decomposed block, **not** the qubits pointed to by
       ``occupation`` -- the Givens sweep then transports the occupation onto the physical target
       orbitals. The positions of ``occupation``'s occupied modes enter only through which columns of
       ``rotation_unitary`` are selected (``rotation_unitary[:, occupied]``); their *count* :math:`m`
       determines the reference. The final prepared state is correct for any occupation, contiguous
       or not.

    .. note::
       *Optionally* (when :attr:`minimize_2q_gate_count` is set; it is **off** by default -- see the
       caution below) occupied orbitals that are basis vectors on a mode disjoint from every other
       occupied orbital are *peeled off* before the decomposition: their ``XGate`` is placed directly
       on the target mode and they are excluded from the Givens sweep. This avoids the chain of bare
       ``SWAP``\ s (``XXPlusYYGate`` with ``c = 0``) that the leading-reference sweep would otherwise
       emit to transport such an orbital out to its mode. Only the genuinely-mixing remainder is then
       decomposed; when the whole occupied space is a signed permutation every orbital peels and no
       ``XXPlusYYGate`` is emitted at all.

       A peeled mode may sit *between* the physical endpoints of a sweep rotation, making that
       ``XXPlusYYGate`` act on physically non-adjacent qubits. Because a peeled mode is an occupied
       unit-orbital, the Jordan-Wigner Z-string it would contribute is folded back into the sign of
       the ``XXPlusYYGate`` phase when an odd number of peeled modes lies strictly between the
       endpoints, so the prepared state stays correct regardless of how the peeled modes interleave
       the mixing space.

    .. note::
       The occupied space (the genuinely-mixing remainder that survives peeling, when
       :attr:`minimize_2q_gate_count` is set; the full occupied space otherwise) is further
       partitioned into disjoint-support **blocks**, each synthesized on its own contiguous mode
       window. When it splits into several mixing clusters separated by empty (or peeled) modes, this
       drops the bare ``SWAP``\ s (``XXPlusYYGate`` with ``c = 0``) the single leading-reference sweep
       would otherwise emit to transport occupation across the gaps between clusters. A single mixing
       cluster spanning all kept modes is one block and reproduces the plain leading-reference sweep
       exactly. This block splitting runs regardless of :attr:`minimize_2q_gate_count`.

    .. caution::
       The unit-orbital peeling above trades circuit metrics: it minimizes the emitted two-qubit gate
       *count* (replacing a swept orbital by a direct ``XGate``), but only under free (all-to-all)
       connectivity. On a constrained coupling map it can *increase* the two-qubit gate *depth* after
       routing -- peeling occupied modes that the leading-reference sweep would have used as
       nearest-neighbor stepping-stones can strand the residual mixing orbital across a wide,
       physically non-adjacent window (a single long-range ``XXPlusYYGate`` that routing must then
       bridge with many ``SWAP``\ s). The full leading-reference sweep instead keeps every emitted
       rotation nearest-neighbor along the mode order -- more two-qubit gates, but no routing overhead.
       The two are a genuine, non-dominated trade-off (neither ever beats the other on both metrics),
       so which is preferable depends on the target: minimizing two-qubit *depth* generally matters
       more than raw gate *count*, both on hardware and in simulation, so peeling is **off** by
       default. Enable it via :attr:`minimize_2q_gate_count` when raw gate count is the bottleneck
       (e.g. richly-connected hardware). Both settings prepare the same state (up to a global phase).

    .. warning::
       This transpilation pass plugin makes the following assumptions:

       - an occupation-basis encoding (like Jordan-Wigner)
       - a trivial fermion-to-qubit layout (i.e. no change in their register lengths)
       - a 1-to-1 mapping of fermionic mode indices to qubit indices
    """

    def __init__(self, *, minimize_2q_gate_count: bool = False) -> None:
        """Initializing this plugin can be done with the arguments listed below.

        Args:
            minimize_2q_gate_count: which of two equivalent syntheses to emit, selected by the circuit
                metric to optimize. When ``False`` (the default), the full leading-reference Givens
                sweep is used: every emitted two-qubit rotation is nearest-neighbor along the mode
                order, minimizing the post-routing two-qubit *depth* on a constrained coupling map at
                the cost of more two-qubit gates. When ``True``, disjoint unit-orbitals (occupied
                orbitals that are basis vectors on a mode disjoint from all other occupied orbitals)
                are peeled out of the decomposition and realized with a direct ``XGate``, minimizing
                the two-qubit gate *count* but potentially emitting long-range rotations that inflate
                the routed depth (see the caution in the class docstring). Both settings prepare the
                same state (up to a global phase); the default favors depth because it generally
                matters more than gate count, both on hardware and in simulation.
        """
        self.minimize_2q_gate_count = minimize_2q_gate_count
        """Whether to peel disjoint unit-orbitals to minimize the two-qubit gate count (see
        ``__init__``). When ``False`` (default), the nearest-neighbor leading-reference sweep is used
        instead, minimizing the routed two-qubit depth."""

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
        """Runs this transpilation plugin.

        Args:
            in_node: the input fermion-based circuit instruction. When this plugin gets called, the
                ``in_node.op`` attribute `must` be of type :class:`.PrepareSlaterDeterminant`.
            out_dag: the output qubit-based circuit.
            f2q_layout: the global transpilation :class:`~qiskit_fermions.transpiler.F2QLayout`
                setting.

        .. seealso::
           The documentation of :class:`.F2QSynthesisPlugin` for more detailed explanations of the
           arguments.

        Raises:
            NotImplementedError: when ``in_node`` acts on fermionic modes that are spread across
                multiple :type:`~qiskit_fermions.circuit.FermionicRegister` instances.
        """
        freg_indices, qreg = map_node_single_register(in_node, f2q_layout)

        occupation = in_node.op.occupation
        rotation_unitary = in_node.op.rotation_unitary

        # the occupied local modes select the occupied orbitals; their columns of the rotation
        # unitary (following a†_i ↦ Σ_j U_ji a†_j) span the prepared Slater determinant's occupied
        # space. ``givens_decomposition_slater`` takes them as the m rows of an m x n matrix.
        occupied = np.nonzero(occupation)[0]
        orbital_coeffs = rotation_unitary[:, occupied].T

        # to minimize the two-qubit gate count (see ``minimize_2q_gate_count``), peel off occupied
        # orbitals that are disjoint unit-orbitals: seed their XGate directly on the target mode and
        # hand only the genuinely-mixing remainder to the decomposition. ``kept_modes`` maps the
        # reduced decomposition's local indices back to physical mode indices. Otherwise (the default),
        # keep the entire occupied space in the sweep so the block splitting below reduces to the full
        # leading-reference sweep -- every emitted rotation then stays nearest-neighbor along the mode
        # order (at the cost of more two-qubit gates), which routes far cheaper on a constrained
        # coupling map.
        if self.minimize_2q_gate_count:
            peel_modes, kept_modes, reduced = _peel_disjoint_unit_orbitals(orbital_coeffs)
        else:
            peel_modes, kept_modes, reduced = (
                [],
                list(range(orbital_coeffs.shape[1])),
                orbital_coeffs,
            )

        for mode in peel_modes:
            out_dag.apply_operation_back(XGate(), (qreg[freg_indices[mode]],))

        # a fully-peeled occupied space (e.g. a permutation rotation) leaves no mixing remainder to
        # decompose; the direct X gates above already prepare the state.
        if reduced.shape[0] == 0:
            return

        # partition the genuinely-mixing remainder into disjoint-support blocks and synthesize each on
        # its own contiguous mode window, dropping the transport SWAPs a single leading sweep would
        # emit across the gaps between clusters (see ``_split_mixing_blocks``).
        for block_rows, block_modes, sub in _split_mixing_blocks(reduced, kept_modes):
            # the reduced decomposition of this block prepares its mixing subspace from the reference
            # in which the first ``len(block_rows)`` modes of the block's window are occupied (``sub``
            # has ``len(block_rows)`` rows); seed that reference. The Givens sweep below then moves the
            # occupation onto the physical target orbitals within the block's window.
            for i in range(len(block_rows)):
                out_dag.apply_operation_back(XGate(), (qreg[freg_indices[block_modes[i]]],))

            # apply the block's Givens rotations in order (same XXPlusYYGate convention as the square
            # plugin), remapping each block-local sweep index through ``block_modes`` to its physical
            # mode. No phase gates: the reduced decomposition carries none, so the state is prepared up
            # to a global phase (physically irrelevant for state preparation).
            #
            # Jordan-Wigner Z-string compensation. A bare ``XXPlusYYGate`` on adjacent qubits is the
            # correct JW fermionic hop only when the two modes are physically adjacent; the framework
            # relies on this (no explicit Z-string is emitted). The sweep is adjacent within the
            # block's contiguous window, but a *peeled* mode (excluded from ``reduced``) may fall
            # strictly between a rotation's physical endpoints. Every peeled mode is an occupied
            # unit-orbital, so each one between the endpoints contributes a JW sign; the missing
            # Z-string over an odd number of them flips the hop's sign. We fold that sign back into the
            # ``XXPlusYYGate`` phase, which is ``-pi/2`` for an even Z-string parity and ``+pi/2`` for
            # an odd one (the two differ by ``pi``, and the gate is ``2 pi``-periodic in its phase).
            # Only *peeled* modes need counting: any occupied mode inside the block's window is itself
            # swept, so its Z-string is already carried implicitly by the nearest-neighbor chain.
            for c, s, i, j in givens_decomposition_slater(sub):
                c_angle = np.acos(c)
                if not np.isclose(c_angle, 0.0):
                    mode_i, mode_j = block_modes[i], block_modes[j]
                    lo, hi = min(mode_i, mode_j), max(mode_i, mode_j)
                    z_string_sign = -1 if sum(1 for p in peel_modes if lo < p < hi) % 2 == 0 else 1
                    out_dag.apply_operation_back(
                        XXPlusYYGate(2 * c_angle, np.angle(s) + z_string_sign * 0.5 * np.pi),
                        (qreg[freg_indices[mode_i]], qreg[freg_indices[mode_j]]),
                    )
