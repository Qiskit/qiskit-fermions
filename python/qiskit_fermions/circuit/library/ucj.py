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

import numbers
from enum import Enum
from typing import TYPE_CHECKING, cast

import numpy as np

from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
from qiskit_fermions.operators.fermion_action import ann, cre

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

    This gate supports three spin variants (see :class:`.UCJ.Variant`), selected by the shapes of the
    supplied tensors and by ``nelec`` (mirroring ffsim's
    :external:class:`~ffsim.UCJOpSpinBalanced`, :external:class:`~ffsim.UCJOpSpinUnbalanced` and
    :external:class:`~ffsim.UCJOpSpinless`):

    - **spin-balanced** -- ``diag_coulomb_mats`` has shape ``(L, 2, norb, norb)`` (the
      ``[alpha-alpha, alpha-beta]`` matrices, with beta-beta reusing alpha-alpha and beta-alpha
      reusing alpha-beta) and ``orbital_rotations`` has shape ``(L, norb, norb)`` (one rotation
      applied to both spin sectors).
    - **spin-unbalanced** -- ``diag_coulomb_mats`` has shape ``(L, 3, norb, norb)`` (the
      ``[alpha-alpha, alpha-beta, beta-beta]`` matrices) and ``orbital_rotations`` has shape
      ``(L, 2, norb, norb)`` (independent ``[alpha, beta]`` rotations).
    - **spinless** -- ``diag_coulomb_mats`` and ``orbital_rotations`` both have shape
      ``(L, norb, norb)``. When ``nelec`` is an integer the gate acts on ``norb`` spinless modes;
      when ``nelec`` is a pair it acts on ``2 * norb`` block-spin modes with the single diagonal
      Coulomb matrix used for both same-spin sectors and no cross-spin interaction.

    .. note::
       This gate builds the ansatz from *exact* tensors. To use ffsim's optimized ("compressed")
       double factorization, construct an ``ffsim`` UCJ operator with ``optimize=True`` and pass its
       ``diag_coulomb_mats`` / ``orbital_rotations`` / ``final_orbital_rotation`` into this
       constructor directly.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.
    """

    class Variant(Enum):
        """The spin variant of a :class:`.UCJ` ansatz.

        Mirrors ffsim's three UCJ operator flavors:
        :external:class:`~ffsim.UCJOpSpinBalanced`, :external:class:`~ffsim.UCJOpSpinUnbalanced`, and
        :external:class:`~ffsim.UCJOpSpinless`. The variant is inferred from the supplied tensor
        shapes (and cross-checked against ``nelec``); its string value is accepted by
        :meth:`.UCJ.from_t_amplitudes`.
        """

        SPINLESS = "spinless"
        BALANCED = "balanced"
        UNBALANCED = "unbalanced"

    def __init__(
        self,
        norb: int,
        nelec: int | tuple[int, int],
        diag_coulomb_mats: np.ndarray,
        orbital_rotations: np.ndarray,
        *,
        final_orbital_rotation: np.ndarray | None = None,
    ) -> None:
        r"""Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            norb: the number of spatial orbitals.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions. Together with the tensor shapes this
                selects the spin variant (see the class docstring).
            diag_coulomb_mats: the diagonal Coulomb matrices, of shape ``(L, 2, norb, norb)``
                (spin-balanced), ``(L, 3, norb, norb)`` (spin-unbalanced), or ``(L, norb, norb)``
                (spinless), where ``L`` is the number of ansatz repetitions.
            orbital_rotations: the orbital rotations, of shape ``(L, norb, norb)`` (spin-balanced or
                spinless) or ``(L, 2, norb, norb)`` (spin-unbalanced).
            final_orbital_rotation: an optional final orbital rotation, of shape ``(norb, norb)``
                (spin-balanced or spinless) or ``(2, norb, norb)`` (spin-unbalanced).

        Raises:
            ValueError: if the tensor shapes are inconsistent with each other, with ``norb``, or
                with ``nelec``.
        """
        # normalize a numpy integer (e.g. ``np.int64``) to a plain ``int`` so the spinless sector is
        # classified correctly here and downstream (ffsim's kernels classify with ``isinstance(int)``)
        if isinstance(nelec, numbers.Integral):
            nelec = int(nelec)

        self.norb = norb
        """The number of spatial orbitals."""
        self.nelec = nelec
        """The number of electrons (spinless) or the ``(n_alpha, n_beta)`` pair (spinful)."""
        self.diag_coulomb_mats = self._to_real_diag_coulomb_mats(diag_coulomb_mats)
        """The diagonal Coulomb matrices defining each ansatz repetition."""
        self.orbital_rotations = np.asarray(orbital_rotations, dtype=complex)
        """The orbital rotations defining each ansatz repetition."""
        self.final_orbital_rotation = (
            None
            if final_orbital_rotation is None
            else np.asarray(final_orbital_rotation, dtype=complex)
        )
        """The optional final orbital rotation."""

        self._variant = self._infer_variant(norb)
        num_modes = norb if self._spinless else 2 * norb

        super().__init__("UCJ", num_modes, [])

    @property
    def _spinless(self) -> bool:
        """Whether the *sector* is spinless (integer ``nelec``, acting on ``norb`` modes).

        This is distinct from :attr:`Variant.SPINLESS` (which is about the tensor *shapes*): a
        spinless-variant operator may still be applied to a spinful ``(n_alpha, n_beta)`` sector, in
        which case it acts on ``2 * norb`` block-spin modes. Mode count, the Hartree-Fock reference,
        and the per-spin gate placement all key off the sector, not the variant.
        """
        # ``nelec`` is normalized to a plain ``int`` for the spinless case in ``__init__``.
        return isinstance(self.nelec, int)

    @classmethod
    def from_t_amplitudes(
        cls,
        nelec: int | tuple[int, int],
        t2: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        t1: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        variant: UCJ.Variant | str = "balanced",
        n_reps: int | tuple[int, int] | None = None,
        interaction_pairs: (
            list[tuple[int, int]] | tuple[list[tuple[int, int]] | None, ...] | None
        ) = None,
        tol: float = 1e-8,
    ) -> UCJ:
        r"""Constructs a UCJ ansatz from coupled-cluster :math:`t_2` (and optional :math:`t_1`) amplitudes.

        The ansatz layers are obtained from an *exact* double factorization of the :math:`t_2`
        amplitudes (via :func:`~qiskit_fermions.linalg.double_factorized_t2` /
        :func:`~qiskit_fermions.linalg.double_factorized_t2_alpha_beta`) and, when :math:`t_1` is
        supplied, a final orbital rotation from :meth:`.OrbitalRotation.from_t1_amplitudes`.

        .. note::
           Only the exact factorization is supported here. To use ffsim's optimized ("compressed")
           double factorization, build an ``ffsim`` UCJ operator with ``optimize=True`` and pass its
           tensors into :class:`.UCJ` directly.

        Args:
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions.
            t2: the :math:`t_2` amplitudes. For the ``"balanced"`` and ``"spinless"`` variants, a
                single array of shape ``(nocc, nocc, nvrt, nvrt)``. For the ``"unbalanced"`` variant,
                a tuple ``(t2aa, t2ab, t2bb)``.
            t1: the optional :math:`t_1` amplitudes producing the final orbital rotation. For
                ``"unbalanced"``, a pair ``(t1a, t1b)``; otherwise a single array of shape
                ``(nocc, nvrt)``.
            variant: the spin variant to build, a :class:`.UCJ.Variant` (or its string value
                ``"balanced"``, ``"unbalanced"``, or ``"spinless"``).
            n_reps: the number of ansatz repetitions. If ``None``, uses all terms of the double
                factorization; if larger, the ansatz is padded with identity rotations and zero
                diagonal Coulomb matrices. For the ``"unbalanced"`` variant a pair
                ``(n_reps_ab, n_reps_same_spin)`` independently sets the number of alpha-beta and
                same-spin terms; a tuple is only valid for that variant.
            interaction_pairs: the allowed diagonal Coulomb interactions (the "local" in LUCJ). For
                ``"spinless"`` a single list of upper-triangular ``(i, j)`` pairs; for ``"balanced"``
                a pair ``(pairs_aa, pairs_ab)``; for ``"unbalanced"`` a triple
                ``(pairs_aa, pairs_ab, pairs_bb)``. A list of pairs restricts that block to exactly
                those entries; the same-spin (aa/bb) masks are symmetrized while the alpha-beta (ab)
                mask is not. Use ``None`` -- not an empty list ``[]`` -- to impose *no* restriction:
                a ``None`` element (or the whole argument being ``None``) leaves the block untouched,
                whereas an empty list allows no interactions and so zeros the entire block.
            tol: the double-factorization truncation tolerance.

        Returns:
            The constructed :class:`.UCJ` gate.

        Raises:
            ValueError: if ``variant`` is not recognized, or if a tuple ``n_reps`` is passed for a
                variant other than :attr:`.UCJ.Variant.UNBALANCED`.
        """
        try:
            variant = UCJ.Variant(variant)
        except ValueError:
            raise ValueError(
                f"Unknown UCJ variant {variant!r}; expected 'balanced', 'unbalanced', or 'spinless'."
            ) from None

        if variant is UCJ.Variant.UNBALANCED:
            # the variant selects which argument shapes are valid; mypy cannot narrow the unions
            diag_coulomb_mats, orbital_rotations = cls._factorize_unbalanced(
                cast("tuple[np.ndarray, np.ndarray, np.ndarray]", t2),
                n_reps,
                interaction_pairs,
                tol,
            )
        else:  # BALANCED or SPINLESS -- both single-tensor same-spin factorizations
            diag_coulomb_mats, orbital_rotations = cls._factorize_same_spin(
                cast(np.ndarray, t2),
                variant,
                cast("int | None", n_reps),
                interaction_pairs,
                tol,
            )

        norb = orbital_rotations.shape[-1]
        final_orbital_rotation = cls._final_rotation_from_t1(t1, variant)

        return cls(
            norb,
            nelec,
            diag_coulomb_mats,
            orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )

    @staticmethod
    def _final_rotation_from_t1(
        t1: np.ndarray | tuple[np.ndarray, np.ndarray] | None, variant: UCJ.Variant
    ) -> np.ndarray | None:
        """Builds the final orbital rotation from ``t1`` (per spin for the unbalanced variant)."""
        if t1 is None:
            return None
        if variant is UCJ.Variant.UNBALANCED:
            t1a, t1b = t1
            return np.stack(
                [
                    OrbitalRotation.from_t1_amplitudes(t1a).rotation_unitary,
                    OrbitalRotation.from_t1_amplitudes(t1b).rotation_unitary,
                ]
            )
        return OrbitalRotation.from_t1_amplitudes(cast(np.ndarray, t1)).rotation_unitary

    @classmethod
    def _factorize_same_spin(
        cls,
        t2: np.ndarray,
        variant: UCJ.Variant,
        n_reps: int | None,
        interaction_pairs,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Factorizes a single ``t2`` tensor into the balanced/spinless ansatz tensors."""
        from qiskit_fermions._lib.linalg.double_factorized import double_factorized_t2

        if isinstance(n_reps, tuple):
            # a tuple ``n_reps`` is the ``(n_reps_ab, n_reps_same_spin)`` pair meaningful only to the
            # unbalanced variant; for balanced/spinless it has a single term count, so a tuple here
            # would otherwise be silently ignored (no truncation, no padding). Reject it instead.
            raise ValueError(
                f"A tuple n_reps={n_reps!r} is only valid for the 'unbalanced' variant; pass a "
                f"single integer (or None) for the '{variant.value}' variant."
            )
        max_terms = n_reps
        terms = double_factorized_t2(np.asarray(t2, dtype=complex), tol, max_terms=max_terms)
        norb = terms[0][1].shape[-1] if terms else np.asarray(t2).shape[0] + np.asarray(t2).shape[2]

        diag_coulomb_mats = np.stack([Z for Z, _ in terms]) if terms else np.empty((0, norb, norb))
        orbital_rotations = np.stack([U for _, U in terms]) if terms else np.empty((0, norb, norb))

        diag_coulomb_mats, orbital_rotations = cls._pad(
            diag_coulomb_mats, orbital_rotations, n_reps, norb
        )

        if variant is UCJ.Variant.SPINLESS:
            cls._mask(diag_coulomb_mats, interaction_pairs, symmetric=True)
            return diag_coulomb_mats, orbital_rotations

        # balanced: stack the single Z into the [aa, ab] layout (aa == ab == Z)
        diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
        if interaction_pairs is not None:
            pairs_aa, pairs_ab = interaction_pairs
            cls._mask(diag_coulomb_mats[:, 0], pairs_aa, symmetric=True)
            cls._mask(diag_coulomb_mats[:, 1], pairs_ab, symmetric=True)
        return diag_coulomb_mats, orbital_rotations

    @classmethod
    def _factorize_unbalanced(
        cls,
        t2: tuple[np.ndarray, np.ndarray, np.ndarray],
        n_reps: int | tuple[int, int] | None,
        interaction_pairs,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Factorizes the ``(t2aa, t2ab, t2bb)`` amplitudes into the unbalanced ansatz tensors."""
        from qiskit_fermions._lib.linalg.double_factorized import (
            double_factorized_t2,
            double_factorized_t2_alpha_beta,
        )

        t2aa, t2ab, t2bb = (np.asarray(t, dtype=complex) for t in t2)
        norb = t2aa.shape[0] + t2aa.shape[2]

        n_reps_ab: int | None
        n_reps_same: int | None
        if isinstance(n_reps, tuple):
            n_reps_ab, n_reps_same = n_reps
        else:
            n_reps_ab = n_reps_same = None

        # alpha-beta block -> (aa, ab, bb) Z-mats and (alpha, beta) rotations
        ab_terms = double_factorized_t2_alpha_beta(t2ab, tol, max_terms=n_reps_ab)
        dc_ab = (
            np.stack([np.stack(zs) for zs, _ in ab_terms])
            if ab_terms
            else np.empty((0, 3, norb, norb))
        )
        rot_ab = (
            np.stack([np.stack(us) for _, us in ab_terms])
            if ab_terms
            else np.empty((0, 2, norb, norb))
        )

        # same-spin (aa and bb) blocks each give a single Z + single rotation per term
        aa_terms = double_factorized_t2(t2aa, tol, max_terms=n_reps_same)
        bb_terms = double_factorized_t2(t2bb, tol, max_terms=n_reps_same)
        dc_same, rot_same = cls._assemble_same_spin_unbalanced(aa_terms, bb_terms, norb)

        diag_coulomb_mats = np.concatenate([dc_ab, dc_same])
        orbital_rotations = np.concatenate([rot_ab, rot_same])

        # total-rep truncation/padding for the single-int form
        if isinstance(n_reps, int):
            diag_coulomb_mats = diag_coulomb_mats[:n_reps]
            orbital_rotations = orbital_rotations[:n_reps]
        diag_coulomb_mats, orbital_rotations = cls._pad(
            diag_coulomb_mats, orbital_rotations, n_reps if isinstance(n_reps, int) else None, norb
        )

        if interaction_pairs is not None:
            pairs_aa, pairs_ab, pairs_bb = interaction_pairs
            cls._mask(diag_coulomb_mats[:, 0], pairs_aa, symmetric=True)
            cls._mask(diag_coulomb_mats[:, 1], pairs_ab, symmetric=False)
            cls._mask(diag_coulomb_mats[:, 2], pairs_bb, symmetric=True)

        return diag_coulomb_mats, orbital_rotations

    @staticmethod
    def _assemble_same_spin_unbalanced(aa_terms, bb_terms, norb):
        """Combines separate aa and bb same-spin factorizations into stacked unbalanced tensors."""
        n = max(len(aa_terms), len(bb_terms))
        dc = np.zeros((n, 3, norb, norb))
        # The rotations are generically complex (``double_factorized_t2`` returns complex ``U``
        # even for real ``t2``); allocate complex so the in-place assignments below don't
        # silently truncate the imaginary part and produce non-unitary rotations.
        rot = (
            np.stack(
                [
                    np.stack([np.eye(norb, dtype=complex), np.eye(norb, dtype=complex)])
                    for _ in range(n)
                ]
            )
            if n
            else np.empty((0, 2, norb, norb), dtype=complex)
        )
        for k in range(len(aa_terms)):
            dc[k, 0] = aa_terms[k][0]
            rot[k, 0] = aa_terms[k][1]
        for k in range(len(bb_terms)):
            dc[k, 2] = bb_terms[k][0]
            rot[k, 1] = bb_terms[k][1]
        return dc, rot

    @staticmethod
    def _pad(diag_coulomb_mats, orbital_rotations, n_reps, norb):
        """Pads the tensors to ``n_reps`` reps with zero J and identity rotation(s), in any variant.

        A ``None`` or tuple ``n_reps`` is a no-op (the tuple form targets per-block rep counts, which
        are already applied upstream). The padding block shapes are inferred from the existing tensors
        so this handles the spinless/balanced single rotation and the unbalanced per-spin pair alike.
        """
        if isinstance(n_reps, tuple) or n_reps is None:
            return diag_coulomb_mats, orbital_rotations
        n_have = diag_coulomb_mats.shape[0]
        if n_have >= n_reps:
            return diag_coulomb_mats, orbital_rotations
        pad = n_reps - n_have
        dc_block = np.zeros((pad, *diag_coulomb_mats.shape[1:]))
        # one identity per trailing rotation slot (a single rotation, or an [alpha, beta] pair)
        n_rot = orbital_rotations.shape[1] if orbital_rotations.ndim == 4 else 1
        eye = np.eye(norb) if n_rot == 1 else np.stack([np.eye(norb)] * n_rot)
        rot_block = np.stack([eye] * pad)
        return (
            np.concatenate([diag_coulomb_mats, dc_block]),
            np.concatenate([orbital_rotations, rot_block]),
        )

    @staticmethod
    def _mask(mats: np.ndarray, pairs: list[tuple[int, int]] | None, *, symmetric: bool) -> None:
        """Zeros diagonal Coulomb entries outside the allowed ``pairs``, in place.

        With ``symmetric=True`` a pair ``(i, j)`` also whitelists ``(j, i)`` (same-spin blocks, whose
        interactions are symmetric); with ``symmetric=False`` only the listed ordered entries survive
        (the alpha-beta block).
        """
        if pairs is None:
            return
        norb = mats.shape[-1]
        mask = np.zeros((norb, norb), dtype=bool)
        if pairs:
            rows, cols = zip(*pairs, strict=True)
            mask[rows, cols] = True
            if symmetric:
                mask[cols, rows] = True
        mats *= mask

    def _infer_variant(self, norb: int) -> UCJ.Variant:
        """Infers the spin variant from the tensor shapes and validates them against ``norb``."""
        dc = self.diag_coulomb_mats
        rot = self.orbital_rotations
        expected_dc: tuple[int, ...]
        expected_rot: tuple[int, ...]

        if dc.ndim == 3 and rot.ndim == 3:
            variant = UCJ.Variant.SPINLESS
            expected_dc = (dc.shape[0], norb, norb)
            expected_rot = (rot.shape[0], norb, norb)
        elif dc.ndim == 4 and dc.shape[1] == 2 and rot.ndim == 3:
            variant = UCJ.Variant.BALANCED
            expected_dc = (dc.shape[0], 2, norb, norb)
            expected_rot = (rot.shape[0], norb, norb)
        elif dc.ndim == 4 and dc.shape[1] == 3 and rot.ndim == 4 and rot.shape[1] == 2:
            variant = UCJ.Variant.UNBALANCED
            expected_dc = (dc.shape[0], 3, norb, norb)
            expected_rot = (rot.shape[0], 2, norb, norb)
        else:
            raise ValueError(
                "Could not infer the UCJ spin variant from the tensor shapes "
                f"diag_coulomb_mats={dc.shape} and orbital_rotations={rot.shape}. Expected "
                "(L, 2, norb, norb)/(L, norb, norb) [balanced], (L, 3, norb, norb)/"
                "(L, 2, norb, norb) [unbalanced], or (L, norb, norb)/(L, norb, norb) [spinless]."
            )

        if dc.shape != expected_dc or rot.shape != expected_rot:
            raise ValueError(
                f"Inconsistent {variant.value} UCJ tensor shapes for norb={norb}: got "
                f"diag_coulomb_mats={dc.shape}, orbital_rotations={rot.shape}."
            )
        if dc.shape[0] != rot.shape[0]:
            raise ValueError(
                f"diag_coulomb_mats and orbital_rotations must have the same number of repetitions; "
                f"got {dc.shape[0]} and {rot.shape[0]}."
            )
        # ``nelec`` is normalized to a plain ``int`` for the spinless case, so an integer nelec means
        # the caller intends the spinless variant; cross-check it against the shape-inferred variant.
        if isinstance(self.nelec, int) and variant is not UCJ.Variant.SPINLESS:
            raise ValueError(
                f"An integer nelec selects the spinless variant, but the tensor shapes imply the "
                f"{variant.value} variant."
            )
        # note: the converse (a spinless operator applied to a spinful sector) is allowed and needs
        # no branch here -- see the class docstring's spinless variant.

        return variant

    @staticmethod
    def _to_real_diag_coulomb_mats(diag_coulomb_mats: np.ndarray) -> np.ndarray:
        """Coerces the diagonal Coulomb matrices to real, rejecting a non-negligible imaginary part.

        Diagonal Coulomb matrices are real by definition, but the documented ffsim ``optimize=True``
        workflow hands over complex arrays whose imaginary parts are numerical round-off. A bare
        ``np.asarray(..., dtype=float)`` would drop those parts silently (only a ``ComplexWarning``),
        so instead take the real part explicitly and raise if the imaginary part is not negligible --
        surfacing genuinely complex input rather than silently truncating it.
        """
        arr = np.asarray(diag_coulomb_mats)
        if np.iscomplexobj(arr):
            if not np.allclose(arr.imag, 0.0):
                raise ValueError(
                    "diag_coulomb_mats has a non-negligible imaginary part; diagonal Coulomb "
                    "matrices must be real."
                )
            arr = arr.real
        return np.asarray(arr, dtype=float)

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
            self.diag_coulomb_mats, self.orbital_rotations, strict=True
        ):
            diag_coulomb = self._diag_coulomb_operator(diag_coulomb_mat)
            self._append_orbital_rotation(definition, self._conj_transpose(orbital_rotation))
            definition.append(
                Evolution(self.num_modes, diag_coulomb, time=-1.0),  # type: ignore[arg-type]
                definition.modes,
            )
            self._append_orbital_rotation(definition, orbital_rotation)

        if self.final_orbital_rotation is not None:
            self._append_orbital_rotation(definition, self.final_orbital_rotation)

        return definition

    def _define(self) -> None:
        self._definition = self._build_definition()._inner

    def _append_orbital_rotation(self, definition, rotation: np.ndarray) -> None:
        """Appends an orbital rotation, respecting the spin variant's per-spin placement.

        Spinless with an integer ``nelec`` places a single rotation on all ``norb`` modes.
        Otherwise the block-spin register is split into alpha modes ``0..norb`` and beta modes
        ``norb..2*norb``: the balanced/spinless case applies the same rotation to both halves, while
        the unbalanced case applies the independent ``[alpha, beta]`` rotations.
        """
        if self._spinless:
            definition.append(OrbitalRotation(rotation), definition.modes)
            return

        if self._variant is UCJ.Variant.UNBALANCED:
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
        num_modes = norb if self._spinless else 2 * norb
        mat_aa, mat_ab, mat_bb = self._resolve_diag_coulomb_blocks(diag_coulomb_mat)

        # a true spinless system has only the aa block on the norb spinless modes
        if self._spinless:
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
        """Returns the ``(aa, ab, bb)`` diagonal Coulomb blocks for one repetition, per variant."""
        norb = self.norb
        if self._variant is UCJ.Variant.BALANCED:
            mat_aa, mat_ab = diag_coulomb_mat[0], diag_coulomb_mat[1]
            return mat_aa, mat_ab, mat_aa
        if self._variant is UCJ.Variant.UNBALANCED:
            return diag_coulomb_mat[0], diag_coulomb_mat[1], diag_coulomb_mat[2]
        # spinless: single matrix used for both same-spin sectors, no cross-spin term
        zero = np.zeros((norb, norb))
        return diag_coulomb_mat, zero, diag_coulomb_mat
