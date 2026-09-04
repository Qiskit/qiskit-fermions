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

"""Unitary cluster Jastrow (UCJ) ansatz gate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
from qiskit_fermions.operators.fermion_action import ann, cre
from qiskit_fermions.utils.optionals import HAS_FFSIM

from .. import FermionicGate
from .evolution import Evolution
from .orbital_rotation import OrbitalRotation

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit


class UCJ(FermionicGate):
    r"""Implements the (local) unitary cluster Jastrow ((L)UCJ) ansatz.

    A unitary cluster Jastrow operator has the form

    .. math::

        \left(\prod_{k=1}^{L} \mathcal{U}_k\, e^{i \mathcal{J}_k}\, \mathcal{U}_k^\dagger\right)
        \mathcal{U}_\text{final}

    where each :math:`\mathcal{U}_k` is an :class:`.OrbitalRotation`, each :math:`\mathcal{J}_k` is a
    diagonal Coulomb operator

    .. math::

        \mathcal{J} = \frac12 \sum_{ij,\sigma\tau} \mathbf{J}^{\sigma\tau}_{ij}\,
        n_{i\sigma}\, n_{j\tau},

    and :math:`\mathcal{U}_\text{final}` is an optional final orbital rotation. The number of terms
    :math:`L` is the number of ansatz repetitions.

    The operator itself is built by `ffsim <https://qiskit-community.github.io/ffsim/>`__, and this
    gate turns it into a :class:`.FermionicCircuit`. That division of labor is deliberate: ffsim owns
    the ansatz math (double factorization of :math:`t_2` amplitudes, the compressed variant, the
    parameter-vector packing that a variational optimizer drives), while this gate expresses the
    result as fermionic modes so that the transpiler can lower it through *any* fermion-to-qubit
    encoding. Passing an ffsim operator through unchanged is what keeps the two consistent; see the
    :ref:`ffsim guide <ffsim_backend_explanation>`.

    Accepts any of ffsim's three UCJ operators, whose type fixes the spin variant and the number of
    modes this gate acts on:

    - :external:class:`~ffsim.UCJOpSpinBalanced` acts on ``2 * norb`` block-spin modes, with one
      orbital rotation shared by both spin sectors and ``[alpha-alpha, alpha-beta]`` diagonal Coulomb
      matrices (beta-beta reuses alpha-alpha, beta-alpha reuses alpha-beta).
    - :external:class:`~ffsim.UCJOpSpinUnbalanced` acts on ``2 * norb`` block-spin modes, with
      independent ``[alpha, beta]`` rotations and ``[alpha-alpha, alpha-beta, beta-beta]`` matrices.
    - :external:class:`~ffsim.UCJOpSpinless` acts on ``norb`` spinless modes.

    .. note::
       ffsim's :external:class:`~ffsim.UCJOpSpinless` is also valid on a spinful sector, where its
       tensors act on both spin sectors with no cross-spin term. A gate has to fix its width when it
       is constructed, so this gate always reads that type as a single ``norb``-mode register. Build
       the two-register reading as an :external:class:`~ffsim.UCJOpSpinBalanced` whose alpha-beta
       block is zero, which is the same operator.

    .. note::
       ffsim does not support Windows (through its unconditional PySCF dependency), so this gate
       requires the ``ffsim`` extra (``pip install "qiskit-fermions[ffsim]"``) and is unavailable
       there. Use `WSL <https://learn.microsoft.com/windows/wsl/>`__ on Windows.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.

    .. invisible-code-block: python

        >>> from qiskit_fermions.utils.optionals import HAS_FFSIM

    .. skip: start if(not HAS_FFSIM)

    .. doctest::

        >>> import ffsim
        >>> from qiskit_fermions.circuit.library import UCJ
        >>> ucj_op = ffsim.random.random_ucj_op_spin_balanced(3, n_reps=1, seed=1234)
        >>> gate = UCJ(ucj_op)
        >>> gate.norb, gate.num_modes, gate.n_reps
        (3, 6, 1)

    .. skip: end
    """

    def __init__(self, ucj_op: Any) -> None:
        """Initializing an instance of this gate can be done with the argument listed below.

        Args:
            ucj_op: the ffsim UCJ operator to build the circuit from, one of
                :external:class:`~ffsim.UCJOpSpinBalanced`,
                :external:class:`~ffsim.UCJOpSpinUnbalanced` or
                :external:class:`~ffsim.UCJOpSpinless`. Its type determines the spin variant and the
                number of modes this gate acts on (see the class docstring).

        Raises:
            MissingOptionalLibraryError: if ``ffsim`` is not installed.
            TypeError: if ``ucj_op`` is not one of ffsim's three UCJ operator types.
        """
        HAS_FFSIM.require_now("UCJ")
        import ffsim

        if not isinstance(
            ucj_op, (ffsim.UCJOpSpinBalanced, ffsim.UCJOpSpinUnbalanced, ffsim.UCJOpSpinless)
        ):
            raise TypeError(
                "UCJ requires one of ffsim's UCJ operators (UCJOpSpinBalanced, "
                f"UCJOpSpinUnbalanced or UCJOpSpinless), but got {type(ucj_op).__name__}."
            )

        self.ucj_op = ucj_op
        """The ffsim UCJ operator this gate builds its circuit from."""

        self._spinless = isinstance(ucj_op, ffsim.UCJOpSpinless)
        self._unbalanced = isinstance(ucj_op, ffsim.UCJOpSpinUnbalanced)

        num_modes = ucj_op.norb if self._spinless else 2 * ucj_op.norb
        super().__init__("UCJ", num_modes, [])

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return int(self.ucj_op.norb)

    @property
    def n_reps(self) -> int:
        """The number of ansatz repetitions."""
        return int(self.ucj_op.n_reps)

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        """Applies the ansatz after placing its modes onto the vector's global modes.

        This builds the gate's definition (the per-repetition orbital rotations and diagonal Coulomb
        evolutions) and applies it to ``vec``, with the definition circuit placed onto the global
        modes ``freg_indices`` (each of its instructions is relabeled onto the corresponding absolute
        modes). See :meth:`_define` for the exact gate sequence.

        Args:
            vec: the state vector to act on.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions.
            copy: whether to copy the vector before operating on it.
            freg_indices: the absolute (global) mode indices that this gate's local modes map onto.

        Returns:
            The transformed vector.
        """
        return self._build_definition()._apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)

    def _build_definition(self) -> FermionicCircuit:
        """Builds the ansatz as a :class:`.FermionicCircuit` (shared by ``_define``)."""
        from qiskit_fermions.circuit import FermionicCircuit

        definition = FermionicCircuit(self.num_modes)

        for diag_coulomb_mat, orbital_rotation in zip(
            self.ucj_op.diag_coulomb_mats, self.ucj_op.orbital_rotations, strict=True
        ):
            diag_coulomb = self._diag_coulomb_operator(diag_coulomb_mat)
            self._append_orbital_rotation(definition, self._conj_transpose(orbital_rotation))
            definition.append(
                Evolution(self.num_modes, diag_coulomb, time=-1.0),  # type: ignore[arg-type]
                definition.modes,
            )
            self._append_orbital_rotation(definition, orbital_rotation)

        if self.ucj_op.final_orbital_rotation is not None:
            self._append_orbital_rotation(definition, self.ucj_op.final_orbital_rotation)

        return definition

    def _define(self) -> None:
        self._definition = self._build_definition()._inner

    def _append_orbital_rotation(self, definition, rotation: np.ndarray) -> None:
        """Appends an orbital rotation, respecting the spin variant's per-spin placement.

        The spinless operator places a single rotation on all ``norb`` modes. Otherwise the
        block-spin register is split into alpha modes ``0..norb`` and beta modes ``norb..2*norb``:
        the balanced case applies the same rotation to both halves, while the unbalanced case
        applies the independent ``[alpha, beta]`` rotations.
        """
        if self._spinless:
            definition.append(OrbitalRotation(rotation), definition.modes)
            return

        if self._unbalanced:
            rotation_a, rotation_b = rotation[0], rotation[1]
        else:
            rotation_a = rotation_b = rotation

        norb = self.norb
        definition.append(OrbitalRotation(rotation_a), definition.modes[:norb])
        definition.append(OrbitalRotation(rotation_b), definition.modes[norb:])

    @staticmethod
    def _conj_transpose(rotation: np.ndarray) -> np.ndarray:
        """Returns the conjugate transpose, per spin sector for a stacked ``(2, norb, norb)``."""
        if rotation.ndim == 3:
            return np.asarray(rotation.transpose(0, 2, 1).conj())
        return np.asarray(rotation.conj().T)

    def _diag_coulomb_operator(self, diag_coulomb_mat: np.ndarray) -> FermionOperator:
        r"""Builds the diagonal Coulomb operator ``J`` for one ansatz repetition.

        ``J = 1/2 sum_{ij, sigma tau} J^{sigma tau}_{ij} n_{i sigma} n_{j tau}`` in the block-spin
        mode convention (mode ``p`` is alpha orbital ``p``, mode ``norb + p`` is beta orbital ``p``).
        The alpha-alpha/alpha-beta/beta-beta blocks are resolved per spin variant. All terms commute,
        so the resulting :class:`.Evolution` is exact.

        Every term is a number-operator product ``n_{i sigma} n_{j tau}`` acting on the (at most two)
        modes ``{i + sigma*norb, j + tau*norb}``. Each ``(block, i, j)`` triple maps to a distinct
        term (no coefficient accumulation into a shared key is ever needed), so the terms are built
        directly with a :attr:`~qiskit_fermions.operators.FermionOperator.groups` index each, such
        that all same-group terms have mutually disjoint support and :class:`.Evolution` synthesis
        emits one parallel layer of two-mode gates per group. The group index is a closed-form edge
        coloring of the interaction graph (see :meth:`_term_group`), assigned on the fly.
        """
        norb = self.norb
        is_spinless = self._spinless
        num_modes = norb if is_spinless else 2 * norb
        mat_aa, mat_ab, mat_bb = self._resolve_diag_coulomb_blocks(diag_coulomb_mat)

        # a true spinless system has only the aa block on the norb spinless modes
        if is_spinless:
            blocks = {(0, 0): mat_aa}
        else:
            blocks = {(0, 0): mat_aa, (0, 1): mat_ab, (1, 0): mat_ab.T, (1, 1): mat_bb}

        # each (block, i, j) is a distinct term, so build them directly -- no collision-handling dict.
        terms_with_groups: list[tuple[tuple, complex, int]] = []
        for (sigma, tau), block in blocks.items():
            offset_i = sigma * norb
            offset_j = tau * norb
            for i in range(norb):
                for j in range(norb):
                    coeff = 0.5 * block[i, j]
                    if coeff == 0.0:
                        continue
                    mode_i = offset_i + i
                    mode_j = offset_j + j
                    term = (cre(mode_i), ann(mode_i), cre(mode_j), ann(mode_j))
                    group = self._term_group(num_modes, mode_i, mode_j)
                    terms_with_groups.append((term, coeff, group))

        return FermionOperator.from_terms_with_groups(terms_with_groups)

    @staticmethod
    def _term_group(num_modes: int, mode_i: int, mode_j: int) -> int:
        r"""Returns the disjoint-support group index for the term ``n_{mode_i} n_{mode_j}``.

        The diagonal Coulomb terms form a *doubled* complete graph on the ``num_modes`` modes: every
        unordered off-diagonal pair ``{p, q}`` carries two terms (the ``(i, j)`` and ``(j, i)``
        orderings) and every mode carries one on-site number term ``n_p`` (the ``i == j`` case). This
        assigns each term a color -- one :class:`.Evolution` group -- so that same-color terms share
        no mode, hence each group synthesizes as a single parallel layer of (at most two-mode) gates.

        The coloring is the closed-form *circle method* (round-robin) edge coloring of the complete
        graph, doubled to carry both orderings of each pair, so it needs no per-term state and is
        layer-**optimal**: it uses exactly the maximum mode degree ``2*num_modes - 1`` colors for even
        ``num_modes`` (always the case for a spinful register), and ``2*num_modes`` for odd
        ``num_modes`` -- the latter being unavoidable (an odd complete graph cannot be edge-colored in
        fewer than ``num_modes`` colors, i.e. degree ``+ 1``).

        Args:
            num_modes: the number of modes the diagonal Coulomb operator acts on.
            mode_i: the first mode of the term (its outer number operator).
            mode_j: the second mode of the term (its inner number operator).

        Returns:
            The group index (color) for this term.
        """
        n = num_modes
        # on-site term n_p (a self-loop at mode p): take the one color the circle method leaves free
        # at p. For odd n that free pair-color is (2p) mod n (doubled to its lower slot); for even n
        # every pair-color is occupied at every mode, so on-site terms share one extra top color.
        if mode_i == mode_j:
            return 2 * ((2 * mode_i) % n) if n % 2 else 2 * (n - 1)

        # off-diagonal pair {p, q}: circle-method color of the edge, doubled to fit both orderings.
        # The ordering copy index r (0 for i < j, 1 for i > j) separates the two same-spin terms that
        # share this support; cross-spin orderings land on different supports and never collide.
        p, q = (mode_i, mode_j) if mode_i < mode_j else (mode_j, mode_i)
        copy = 0 if mode_i < mode_j else 1
        if n % 2:
            edge_color = (p + q) % n
        else:
            m = n - 1
            edge_color = (2 * p) % m if q == m else (p + q) % m
        return 2 * edge_color + copy

    def _resolve_diag_coulomb_blocks(
        self, diag_coulomb_mat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the ``(aa, ab, bb)`` diagonal Coulomb blocks for one repetition, per operator type."""
        if self._spinless:
            # spinless: single matrix used for both same-spin sectors, no cross-spin term
            zero = np.zeros((self.norb, self.norb))
            return diag_coulomb_mat, zero, diag_coulomb_mat
        if self._unbalanced:
            return diag_coulomb_mat[0], diag_coulomb_mat[1], diag_coulomb_mat[2]
        # balanced: beta-beta reuses alpha-alpha
        mat_aa, mat_ab = diag_coulomb_mat[0], diag_coulomb_mat[1]
        return mat_aa, mat_ab, mat_aa
