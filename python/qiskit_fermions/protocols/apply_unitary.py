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


class SupportsApplyUnitaryPlaced(Protocol):
    """A package-specific extension of :class:`ffsim.SupportsApplyUnitary`.

    Where ffsim's protocol applies an object to a whole state vector, this one adds the placement of
    the object's local modes onto the global ones it acts on. It has no ffsim equivalent:
    :meth:`~.FermionicCircuit._apply_unitary_placed_` is the only caller, and it dispatches to it
    directly via ``getattr`` duck-typing. It exists here purely for typing and documentation
    purposes, since every concrete :class:`.FermionicGate` (as well as :class:`.FermionicCircuit`
    itself) implements this method.

    See :meth:`.FermionicCircuit._apply_unitary_placed_` for the full semantics, including how a
    plain ``_apply_unitary_`` implementation (with no mode-placement argument) is honored only on the
    identity placement.
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
