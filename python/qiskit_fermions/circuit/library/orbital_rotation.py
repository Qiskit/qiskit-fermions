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

"""Orbital rotation gate."""

from __future__ import annotations

import numbers

import numpy as np

from qiskit_fermions.utils.optionals import HAS_FFSIM

from .. import FermionicGate


class OrbitalRotation(FermionicGate):
    r"""Implements an orbital rotation.

    Given an :math:`n \times n` unitary matrix :math:`U` (``rotation_unitary``), this gate
    implements the single-particle basis change that maps the creation operators as

    .. math::

        a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j,

    which is equivalent to applying the many-body unitary

    .. math::

        \exp\left(\sum_{ij} \log(U)_{ij} \, a^\dagger_i a_j\right).

    The number of fermionic modes the gate acts on is the dimension :math:`n` of
    ``rotation_unitary``.
    """

    def __init__(self, rotation_unitary: np.ndarray) -> None:
        r"""Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            rotation_unitary: the :math:`n \times n` unitary matrix :math:`U` defining the orbital
                rotation via :math:`a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j`. It must be
                square and unitary; this is the caller's responsibility and is not verified.
        """
        self.rotation_unitary = rotation_unitary
        """The unitary matrix representing the orbital rotation coefficients."""

        super().__init__("OrbitalRotation", self.rotation_unitary.shape[0], [])

    @classmethod
    def from_t1_amplitudes(cls, t1: np.ndarray) -> OrbitalRotation:
        r"""Constructs an orbital rotation from :math:`t_1` (singles) amplitudes.

        The rotation is the unitary :math:`\exp(t_1 - t_1^\dagger)`, where the
        :math:`n_\text{occ} \times n_\text{virt}` amplitude matrix :math:`t_1` is embedded into the
        anti-Hermitian generator over all :math:`n = n_\text{occ} + n_\text{virt}` orbitals

        .. math::

            G = \begin{pmatrix} 0 & -t_1^* \\ t_1^\top & 0 \end{pmatrix},

        with the occupied orbitals ordered before the virtual ones. This is the single-excitation
        orbital rotation entering the (L)UCJ ansatz when it is initialized from coupled-cluster
        amplitudes; see :class:`.UCJ`.

        Args:
            t1: the :math:`t_1` amplitudes of shape ``(nocc, nvrt)``, where ``nocc`` is the number
                of occupied orbitals and ``nvrt`` is the number of virtual orbitals.

        Returns:
            An :class:`.OrbitalRotation` acting on :math:`n = n_\text{occ} + n_\text{virt}` modes,
            whose ``rotation_unitary`` is :math:`\exp(t_1 - t_1^\dagger)`.
        """
        import scipy.linalg

        nocc, nvrt = t1.shape
        norb = nocc + nvrt
        generator = np.zeros((norb, norb), dtype=complex)
        generator[:nocc, nocc:] = -t1.conj()
        generator[nocc:, :nocc] = t1.T
        return cls(scipy.linalg.expm(generator))

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Applies this orbital rotation to an ffsim state vector.

        This implements ffsim's ``SupportsApplyUnitary`` protocol. See
        :meth:`_apply_unitary_placed_` for the details; this method assumes the gate acts on the
        modes ``0..num_modes`` of the state vector (i.e. an identity mode placement).
        """
        return self._apply_unitary_placed_(vec, norb, nelec, copy, list(range(self.num_modes)))

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        r"""Applies the orbital rotation after placing it onto the vector's global modes.

        The gate's local :attr:`rotation_unitary` (an :math:`n \times n` matrix acting on the gate's
        ``num_modes`` modes) is first embedded into the full register: an identity matrix of the
        state vector's mode count with ``rotation_unitary`` written into the rows/columns picked out
        by ``freg_indices``.

        In the spinful case the embedded matrix must be block-diagonal across the alpha/beta split.
        A rotation with nonzero alpha/beta off-diagonal blocks mixes the spin sectors, which does not
        conserve the individual alpha/beta electron counts and hence maps amplitude out of the fixed
        ``(n_alpha, n_beta)`` sector -- an operation the fixed-sector state vector cannot represent.
        Such a rotation is rejected with a :class:`ValueError`.

        The embedded matrix is then applied in one of two ways:

        - **Fast path** (only when ``ffsim`` is installed): the embedded matrix is applied via
          :func:`ffsim.apply_orbital_rotation`'s Givens-rotation kernel. Under the spinful block-spin
          convention (modes ``0..norb`` are alpha orbitals, modes ``norb..2*norb`` are beta orbitals)
          the two diagonal blocks are the per-spin rotations passed to ffsim as ``(mat_a, mat_b)``.
        - **General path**: otherwise (i.e. when ``ffsim`` is unavailable) the rotation is applied as
          the evolution :math:`\exp(G)` under its generator
          :math:`G = \sum_{ij} \log(U)_{ij} a^\dagger_i a_j`, where :math:`U` is the embedded matrix.
          :math:`G` is turned into a ``scipy`` ``LinearOperator`` via
          :meth:`~qiskit_fermions.operators.FermionOperator._linear_operator_` (backed by the native
          FCI matrix-vector kernel) and applied via :func:`scipy.sparse.linalg.expm_multiply`. This
          mirrors :meth:`.Evolution._apply_unitary_placed_`.

        Args:
            vec: the state vector to act on.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions. An integer selects the spinless mode
                interpretation (the ``norb`` modes are orbitals); a pair selects the spinful
                ``(orb, spin)`` block-spin interpretation of the ``2 * norb`` modes.
            copy: whether to copy the vector before operating on it.
            freg_indices: the absolute (global) mode indices that this gate's local modes map onto.
                The rotation is embedded onto these global modes before being applied.

        Returns:
            The transformed vector.

        Raises:
            ValueError: if ``nelec`` is a spinful pair and the (placed) rotation mixes the alpha and
                beta spin sectors.
        """
        # normalize a numpy integer (e.g. ``np.int64``) to a plain ``int`` so the spinless sector is
        # classified correctly here and downstream (ffsim's kernels classify with ``isinstance(int)``)
        if isinstance(nelec, numbers.Integral):
            nelec = int(nelec)
        num_modes = norb if isinstance(nelec, int) else 2 * norb

        # embed the local rotation onto its global modes: identity everywhere except the placed
        # rows/columns, which carry the gate's rotation_unitary
        full = np.eye(num_modes, dtype=complex)
        full[np.ix_(freg_indices, freg_indices)] = self.rotation_unitary

        # resolve the argument ffsim's kernel (and, when it mixes spins, the whole gate) accepts
        mat = self._resolve_orbital_rotation(full, norb, nelec)

        if HAS_FFSIM:
            import ffsim

            return ffsim.apply_orbital_rotation(vec, mat, norb=norb, nelec=nelec, copy=copy)

        return self._apply_via_generator(full, vec, norb, nelec, copy)

    @staticmethod
    def _resolve_orbital_rotation(
        full: np.ndarray, norb: int, nelec: int | tuple[int, int]
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Returns the per-spin rotation(s) for the embedded matrix, rejecting spin-mixing rotations.

        The spinless case returns ``full`` directly. The spinful case returns the ``(mat_a, mat_b)``
        diagonal blocks, but only when ``full`` is block-diagonal across the alpha/beta split; a
        rotation with nonzero off-diagonal blocks mixes the spin sectors -- which does not conserve
        the per-spin electron counts and so cannot act on a fixed ``(n_alpha, n_beta)`` sector -- and
        raises :class:`ValueError`. The off-block test uses a small tolerance so that a
        genuinely block-diagonal rotation carrying only floating-point round-off in its off-blocks
        (e.g. one assembled via ``scipy.linalg.expm`` of a block-diagonal generator) is accepted. The
        result is also exactly the argument :func:`ffsim.apply_orbital_rotation` expects.
        """
        if isinstance(nelec, numbers.Integral):
            return full

        if not np.allclose(full[:norb, norb:], 0.0) or not np.allclose(full[norb:, :norb], 0.0):
            raise ValueError(
                "OrbitalRotation mixes the alpha and beta spin sectors, which does not conserve the "
                "individual electron counts and cannot act on a fixed (n_alpha, n_beta) sector."
            )

        return full[:norb, :norb], full[norb:, norb:]

    @staticmethod
    def _apply_via_generator(
        full: np.ndarray,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
    ) -> np.ndarray:
        r"""Applies ``exp(G)`` for ``G = sum_ij log(full)_ij a^\dagger_i a_j`` via the native kernel.

        This is the ``ffsim``-free fallback, requiring only ``scipy`` plus the native FCI
        matrix-vector kernel. ``full`` has already been validated by :meth:`_resolve_orbital_rotation`
        to be sector-preserving (spinless, or block-diagonal across the alpha/beta split), so the
        generator ``G`` conserves the ``(norb, nelec)`` sector and no amplitude is dropped.
        """
        import scipy.linalg
        import scipy.sparse.linalg

        from qiskit_fermions._lib.operators.fermion_operator import FermionOperator

        if copy:
            vec = vec.copy()

        log_mat = scipy.linalg.logm(full)
        # ``logm`` of a block-diagonal / sparse rotation leaves ~1e-16 round-off in nominally-zero
        # entries; drop them with a tolerance rather than an exact ``!= 0.0`` so the generator carries
        # only the genuine terms (an exactly-zero threshold keeps the junk, bloating the operator).
        tol = 1e-12
        # the generator's modes are already global (``full`` was embedded onto them), so no relabel
        # is needed -- unlike Evolution, which relabels a local operator onto its global modes
        terms = {
            ((True, i), (False, j)): complex(log_mat[i, j])
            for i in range(full.shape[0])
            for j in range(full.shape[1])
            if abs(log_mat[i, j]) > tol
        }
        generator = FermionOperator.from_dict(terms)

        linop = generator._linear_operator_(norb, nelec)
        # ``traceA=0.0`` mirrors Evolution._apply_unitary_placed_: it is only a scipy conditioning
        # hint (it factors out ``exp(traceA / n)``), not a correctness input.
        return scipy.sparse.linalg.expm_multiply(linop, vec, traceA=0.0)
