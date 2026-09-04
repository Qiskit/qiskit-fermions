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

"""Protocols to indicate state-vector simulation support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class SupportsApplyUnitary(Protocol):
    """The state-vector simulation contract this package implements.

    This is the same protocol that :class:`ffsim.SupportsApplyUnitary` describes, so
    :func:`ffsim.apply_unitary` dispatches to the method below on any object of this package. It is
    stated here because the contract holds whether or not ffsim is installed: without it, call the
    method directly, as the doctest below does.

    .. doctest::

        >>> import numpy as np
        >>> from qiskit_fermions.circuit.library import OrbitalRotation
        >>> gate = OrbitalRotation(np.eye(2))
        >>> vec = np.array([1.0, 0.0], dtype=complex)
        >>> gate._apply_unitary_(vec, norb=2, nelec=1, copy=True)
        array([1.+0.j, 0.+0.j])
    """

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Applies a unitary transformation to a state vector.

        Args:
            vec: the state vector to apply the unitary transformation to.
            norb: the number of spatial orbitals.
            nelec: either a single integer representing the number of fermions for a spinless
                system, or a pair of integers storing the numbers of spin alpha and spin beta
                fermions.
            copy: whether to copy the vector before operating on it.

                - If ``copy=True`` then this method always returns a newly allocated vector and the
                  original vector is left untouched.
                - If ``copy=False`` then this method may still return a newly allocated vector, but
                  the original vector may have its data overwritten. It is also possible that the
                  original vector is returned, modified in-place.

        Returns:
            The transformed vector.
        """


class SupportsApplyUnitaryPlaced(Protocol):
    """A package-specific extension of :class:`.SupportsApplyUnitary` carrying a mode placement.

    Unlike :class:`.SupportsApplyUnitary`, this protocol has no ffsim equivalent:
    :meth:`~.FermionicCircuit._apply_unitary_placed_` is the only caller, and it dispatches to it
    directly via ``getattr`` duck-typing. It exists here purely for typing and documentation
    purposes, since every concrete :class:`.FermionicGate` (as well as :class:`.FermionicCircuit`
    itself) implements this method.

    See :meth:`.FermionicCircuit._apply_unitary_placed_` for the full semantics, including how a
    plain :class:`.SupportsApplyUnitary` implementation (with no mode-placement argument) is honored
    only on the identity placement.
    """

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        """Applies a unitary transformation to a state vector, after placing local modes onto global ones.

        Args:
            vec: the state vector to apply the unitary transformation to.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer representing the number of fermions for a spinless
                system, or a pair of integers storing the numbers of spin alpha and spin beta
                fermions.
            copy: whether to copy the vector before operating on it.
            freg_indices: the absolute (global) mode indices that this object's local modes map onto.

        Returns:
            The transformed vector.
        """
