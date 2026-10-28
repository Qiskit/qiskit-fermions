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
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from .. import FermionicGate

if TYPE_CHECKING:
    from .synthesis import F2FSynthesisPlugin


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

    .. note::
       Breaking this gate down in fermionic space (into a circuit of smaller gates) is
       **optional**, and it is what the :attr:`~.FermionicGate.synthesis` attribute governs. The gate can equally be
       handed straight to the fermion-to-qubit stage. When it `is` decomposed (by
       :meth:`~.FermionicCircuit.decompose`, for instance) :attr:`~.FermionicGate.synthesis`
       decides how, and the
       circuit it produces implements the same unitary as the gate: it `is` the definition of the
       gate. It defaults to :class:`.FermionicGivensDecomposition`. See
       :mod:`~qiskit_fermions.circuit.library.synthesis`.

    .. note::
       No fermion-to-qubit plugin maps this gate directly. Break it down in fermionic space first,
       with the :class:`.F2FSynthesis` transpiler pass, so that the fermion-to-qubit stage only sees
       the :class:`.Evolution` gates it produces.

    .. note::
       The state-vector simulation path does `not` go through :attr:`~.FermionicGate.synthesis`: it applies the
       whole ``rotation_unitary`` at once, so the gate's definition is never read there.
    """

    def __init__(
        self,
        rotation_unitary: np.ndarray,
        *,
        synthesis: F2FSynthesisPlugin | None = None,
    ) -> None:
        r"""Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            rotation_unitary: the :math:`n \times n` unitary matrix :math:`U` defining the orbital
                rotation via :math:`a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j`. It must be
                square and unitary; this is the caller's responsibility and is not verified.
            synthesis: the fermion-to-fermion synthesis method with which to decompose this gate,
                when it gets decomposed at all. If ``None`` (the default), a
                :class:`.FermionicGivensDecomposition` instance is used.
        """
        # deferred: the synthesis subpackage imports this module's siblings at load time
        from .synthesis import FermionicGivensDecomposition

        self.rotation_unitary = np.asarray(rotation_unitary, dtype=complex)
        """The unitary matrix representing the orbital rotation coefficients."""

        super().__init__(
            "OrbitalRotation",
            self.rotation_unitary.shape[0],
            [],
            synthesis=synthesis or FermionicGivensDecomposition(),
        )

    def inverse(self, annotated: bool = False) -> OrbitalRotation:
        r"""Returns the inverse of this gate, whose unitary is :math:`U^\dagger`.

        Args:
            annotated: ignored. The inverse of this gate is another :class:`.OrbitalRotation`, so it
                never needs to be an :class:`~qiskit.circuit.AnnotatedOperation`.

        Returns:
            An :class:`.OrbitalRotation` gate with the adjoint unitary and the same synthesis.
        """
        # `Instruction.inverse` would recurse through the definition and return a plain `Gate`,
        # losing the unitary and this gate's synthesis method; the adjoint is both cheaper and exact.
        return OrbitalRotation(self.rotation_unitary.conj().T, synthesis=self.synthesis)

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` is an orbital rotation with exactly the same unitary.

        The synthesis is not compared: it decides how the gate is decomposed, not what it is.
        """
        if not isinstance(other, OrbitalRotation):
            return False
        return self.rotation_unitary.shape == other.rotation_unitary.shape and bool(
            np.array_equal(self.rotation_unitary, other.rotation_unitary)
        )

    def __hash__(self) -> int:
        """Hashes the unitary, consistently with :meth:`__eq__`."""
        return hash(
            (type(self).__name__, self.rotation_unitary.shape, self.rotation_unitary.tobytes())
        )

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
        nocc, nvrt = t1.shape
        norb = nocc + nvrt
        generator = np.zeros((norb, norb), dtype=complex)
        generator[:nocc, nocc:] = -t1.conj()
        generator[nocc:, :nocc] = t1.T
        return cls(scipy.linalg.expm(generator))

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

        The embedded matrix is then applied via :func:`ffsim.apply_orbital_rotation`'s Givens-rotation
        kernel. Under the spinful block-spin convention (modes ``0..norb`` are alpha orbitals, modes
        ``norb..2*norb`` are beta orbitals) the two diagonal blocks are the per-spin rotations passed
        to ffsim as ``(mat_a, mat_b)``.

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
            MissingOptionalLibraryError: if ffsim is not installed.
            ValueError: if ``nelec`` is a spinful pair and the (placed) rotation mixes the alpha and
                beta spin sectors.
        """
        num_modes = norb if isinstance(nelec, int) else 2 * norb

        # embed the local rotation onto its global modes: identity everywhere except the placed
        # rows/columns, which carry the gate's rotation_unitary
        full = np.eye(num_modes, dtype=complex)
        full[np.ix_(freg_indices, freg_indices)] = self.rotation_unitary

        # resolve the argument ffsim's kernel (and, when it mixes spins, the whole gate) accepts
        mat = self._resolve_orbital_rotation(full, norb, nelec)

        import ffsim

        return ffsim.apply_orbital_rotation(vec, mat, norb=norb, nelec=nelec, copy=copy)

    @staticmethod
    def _resolve_orbital_rotation(
        full: np.ndarray, norb: int, nelec: int | tuple[int, int]
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Returns the per-spin rotations for the embedded matrix, rejecting spin-mixing rotations.

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
                "individual electron counts and cannot act on a fixed (n_alpha, n_beta) sector "
                f"(norb={norb}, nelec={nelec})."
            )

        return full[:norb, :norb], full[norb:, norb:]
