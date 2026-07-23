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
       Occupied orbitals that are basis vectors on a mode disjoint from every other occupied orbital
       are *peeled off* before the decomposition: their ``XGate`` is placed directly on the target
       mode and they are excluded from the Givens sweep (see
       :func:`_peel_disjoint_unit_orbitals`). This avoids the chain of bare ``SWAP``\ s
       (``XXPlusYYGate`` with ``c = 0``) that the leading-reference sweep would otherwise emit to
       transport such an orbital out to its mode. Only the genuinely-mixing remainder is decomposed.
       When the whole occupied space is a signed permutation every orbital peels and no
       ``XXPlusYYGate`` is emitted at all.

    .. warning::
       This transpilation pass plugin makes the following assumptions:

       - an occupation-basis encoding (like Jordan-Wigner)
       - a trivial fermion-to-qubit layout (i.e. no change in their register lengths)
       - a 1-to-1 mapping of fermionic mode indices to qubit indices
    """

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

        # peel off occupied orbitals that are disjoint unit-orbitals: seed their XGate directly on the
        # target mode and hand only the genuinely-mixing remainder to the decomposition. ``kept_modes``
        # maps the reduced decomposition's local indices back to physical mode indices.
        peel_modes, kept_modes, reduced = _peel_disjoint_unit_orbitals(orbital_coeffs)

        for mode in peel_modes:
            out_dag.apply_operation_back(XGate(), (qreg[freg_indices[mode]],))

        # the reduced decomposition prepares the mixing remainder from the reference in which the
        # first ``m - p`` *kept* modes are occupied (``reduced`` has ``m - p`` rows); seed that
        # reference. The Givens sweep below then moves the occupation onto the physical target
        # orbitals within the kept modes.
        num_mixing = reduced.shape[0]
        for i in range(num_mixing):
            out_dag.apply_operation_back(XGate(), (qreg[freg_indices[kept_modes[i]]],))

        # a fully-peeled occupied space (e.g. a permutation rotation) leaves no mixing remainder to
        # decompose; the direct X gates above already prepare the state.
        if num_mixing == 0:
            return

        # apply the Givens rotations in order (same XXPlusYYGate convention as the square plugin),
        # remapping each local sweep index through ``kept_modes`` to its physical mode. No phase
        # gates: the reduced decomposition carries none, so the state is prepared up to a global phase
        # (physically irrelevant for state preparation).
        for c, s, i, j in givens_decomposition_slater(reduced):
            c_angle = np.acos(c)
            if not np.isclose(c_angle, 0.0):
                out_dag.apply_operation_back(
                    XXPlusYYGate(2 * c_angle, np.angle(s) - 0.5 * np.pi),
                    (qreg[freg_indices[kept_modes[i]]], qreg[freg_indices[kept_modes[j]]]),
                )
