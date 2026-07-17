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

"""Mode initialization gate."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from qiskit_fermions.utils.optionals import HAS_FFSIM

from .. import FermionicGate

logger = logging.getLogger(__name__)


class InitializeModes(FermionicGate):
    """Implements the fermionic mode initialization.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.
    """

    def __init__(self, occupation: Sequence[bool]) -> None:
        """Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            occupation: a sequence of booleans indicating the occupation for each mode in the
                :class:`.FermionicRegister` being initialized by this gate.
        """
        self.occupation = np.asarray(occupation, dtype=bool)
        """The sequence of booleans indicating the occupation for each mode in the
        :class:`~.FermionicRegister` being initialized by this gate."""

        super().__init__("InitializeModes", len(self.occupation), [])

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray | None,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        r"""Produces the occupation determinant, placing it onto the vector's global modes.

        Unlike :class:`.OrbitalRotation` or :class:`.Evolution`, this gate is a state *producer*, not
        a transform: it prepares the occupation determinant a Jordan-Wigner occupation would create
        from the vacuum. Because seeding a determinant *defines* the fixed ``(norb, nelec)`` particle
        number sector (the vacuum lives in a different sector), it cannot be expressed as a
        same-length linear map of an incoming vector -- so this method returns a freshly built state
        vector rather than transforming ``vec``.

        The gate's local :attr:`occupation` (one flag per local mode) is placed onto the global
        register via ``freg_indices``: local mode ``i`` occupies global mode ``freg_indices[i]``. The
        occupied global modes are then interpreted under the ``(norb, nelec)`` convention:

        - **Spinless** (``nelec`` is an ``int``): the ``norb`` modes are orbitals directly; the seed
          is a one-hot at the determinant's address in the ``C(norb, nelec)``-dimensional space.
        - **Spinful** (``nelec`` is a pair): under the block-spin convention modes ``0..norb`` are
          alpha orbitals and modes ``norb..2*norb`` are beta orbitals; the seed is a one-hot at the
          flat index ``addr_a * dim_b + addr_b`` (alpha slow, beta fast).

        The determinant is built via :func:`ffsim.slater_determinant` when ``ffsim`` is installed, and
        otherwise via the native ``slater_determinant_statevector`` kernel (both produce the same
        one-hot, since a position-indexed occupation is inherently sorted and carries no sign).

        Args:
            vec: the state vector to act on, or ``None`` to seed from no incoming state. When a real
                array is passed it must *agree* with the occupation -- same ``(norb, nelec)`` sector
                dimension -- in which case a warning is logged and the freshly seeded determinant is
                returned (the incoming amplitudes are replaced, since this gate defines the initial
                state; placing it mid-circuit therefore drops the preceding gates' effect). A vector
                of the wrong length is rejected.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions. An integer selects the spinless mode
                interpretation (the ``norb`` modes are orbitals); a pair selects the spinful
                ``(orb, spin)`` block-spin interpretation of the ``2 * norb`` modes.
            copy: accepted for protocol conformance but has no effect -- a fresh state vector is
                always returned, so any incoming ``vec`` is inherently left untouched.
            freg_indices: the absolute (global) mode indices that this gate's local modes map onto.

        Returns:
            The seeded occupation determinant as a state vector.

        Raises:
            ValueError: if the occupation's per-sector electron counts do not match ``nelec``, if an
                occupied mode falls outside the range implied by ``norb``, or if a non-``None`` ``vec``
                does not match the ``(norb, nelec)`` sector dimension.
        """
        spinless = isinstance(nelec, int)
        num_modes = norb if spinless else 2 * norb

        # place the local occupation onto its global modes
        global_occ = sorted(
            int(g) for g, occ in zip(freg_indices, self.occupation, strict=True) if occ
        )

        if global_occ and (global_occ[0] < 0 or global_occ[-1] >= num_modes):
            raise ValueError(
                f"InitializeModes places an occupied mode outside the range [0, {num_modes}) "
                f"implied by norb={norb} and nelec={nelec!r}."
            )

        # split the occupied global modes into per-sector occupied orbital lists and validate that
        # their electron counts define the requested (norb, nelec) sector
        if spinless:
            alpha_orbitals = global_occ
            beta_orbitals: list[int] = []
            counts: tuple[int, ...] = (len(alpha_orbitals),)
            expected: tuple[int, ...] = (nelec,)  # type: ignore[assignment]
        else:
            alpha_orbitals = [m for m in global_occ if m < norb]
            beta_orbitals = [m - norb for m in global_occ if m >= norb]
            counts = (len(alpha_orbitals), len(beta_orbitals))
            expected = nelec  # type: ignore[assignment]

        if counts != tuple(expected):
            raise ValueError(
                f"InitializeModes occupation defines the electron counts {counts}, which do not "
                f"match the requested nelec={nelec!r}; the occupation determinant is not in the "
                "target particle-number sector."
            )

        seed = self._seed_statevector(norb, spinless, alpha_orbitals, beta_orbitals)

        if vec is not None:
            # a real vector must agree with the occupation's own sector; the per-sector counts were
            # already validated above, so a matching length confirms agreement
            if len(vec) != len(seed):
                raise ValueError(
                    f"InitializeModes received a state vector of length {len(vec)}, which does not "
                    f"match the dimension {len(seed)} of the (norb={norb}, nelec={nelec!r}) sector "
                    "defined by its occupation."
                )
            logger.warning(
                "InitializeModes: discarding the incoming state vector and reseeding the "
                "occupation determinant for the (norb=%s, nelec=%r) sector. This gate is a state "
                "producer, not a transform, so any accumulated amplitudes are replaced -- placing "
                "it after other gates (rather than at the circuit start) drops their effect.",
                norb,
                nelec,
            )

        return seed

    @staticmethod
    def _seed_statevector(
        norb: int,
        spinless: bool,
        alpha_orbitals: list[int],
        beta_orbitals: list[int],
    ) -> np.ndarray:
        """Builds the one-hot occupation determinant for the given per-sector occupied orbitals.

        For a spinless system ``beta_orbitals`` is empty and ignored. Uses
        :func:`ffsim.slater_determinant` when ``ffsim`` is installed, and otherwise the native
        ``slater_determinant_statevector`` kernel. Both return the same one-hot at the determinant's
        FCI address; the occupied-orbital lists are already sorted, so the determinant carries no
        sign.
        """
        if HAS_FFSIM:
            import ffsim

            occupied = alpha_orbitals if spinless else (alpha_orbitals, beta_orbitals)
            return ffsim.slater_determinant(norb, occupied)

        from qiskit_fermions._lib.linalg.fci import slater_determinant_statevector

        alpha_str = sum(1 << orb for orb in alpha_orbitals)
        beta_str = None if spinless else sum(1 << orb for orb in beta_orbitals)
        return slater_determinant_statevector(norb, alpha_str, beta_str)
