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

"""Slater determinant preparation gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .. import FermionicGate
from .initialize_modes import InitializeModes
from .orbital_rotation import OrbitalRotation

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit


class PrepareSlaterDeterminant(FermionicGate):
    r"""Prepares a Slater determinant from an occupation and an orbital rotation.

    This gate is the composition of an :class:`.InitializeModes` reference occupation followed by an
    :class:`.OrbitalRotation`: it declares that the modes it acts on start in the determinant defined
    by ``occupation`` and are then rotated by ``rotation_unitary``. Because the initial occupation is
    known, transpiling this gate can use the rectangular :func:`.givens_decomposition_slater` for a
    reduced-gate-count synthesis (see :class:`.GivensDecompositionSlaterDeterminantSynthesis`) rather
    than the full square decomposition an :class:`.OrbitalRotation` alone would require.

    Following the :class:`.OrbitalRotation` convention :math:`a^\dagger_i \mapsto \sum_j U_{ji}
    a^\dagger_j`, rotating the occupied modes maps them onto the corresponding columns of
    ``rotation_unitary``; those columns span the occupied space of the prepared Slater determinant.

    .. rubric:: Simulation semantics

    Under simulation this gate is **validate-then-rotate**, not a state producer: given an incoming
    state vector it validates that the vector is confined to the subspace ``occupation`` defines (via
    :class:`.InitializeModes`) and then applies ``rotation_unitary`` (via :class:`.OrbitalRotation`).
    It therefore requires a real reference state vector, prepared externally (e.g. via
    :func:`ffsim.slater_determinant`). The *producer* behavior -- emitting the gates that set the
    reference occupation -- lives in the synthesis plugin, not the simulation path.

    Applying (rather than dropping) the rotation is what guarantees that merging an
    :class:`.InitializeModes` and an :class:`.OrbitalRotation` into this single gate leaves the
    simulated final state unchanged.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.

    .. seealso::
       :class:`.InitializeModes`, :class:`.OrbitalRotation`, and
       :func:`~qiskit_fermions.linalg.givens_decomposition_slater`.
    """

    def __init__(self, occupation: Sequence[bool], rotation_unitary: np.ndarray) -> None:
        r"""Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            occupation: a sequence of booleans indicating the reference occupation for each mode this
                gate acts on.
            rotation_unitary: the :math:`n \times n` unitary matrix :math:`U` defining the orbital
                rotation via :math:`a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j`, where :math:`n` is
                the number of modes (``len(occupation)``). It must be square and unitary; this is the
                caller's responsibility and is not verified.

        Raises:
            ValueError: if ``rotation_unitary`` is not a square matrix whose dimension matches the
                length of ``occupation``.
        """
        self.occupation = np.asarray(occupation, dtype=bool)
        """The reference occupation (one boolean per mode) the rotation is applied to."""

        self.rotation_unitary = np.asarray(rotation_unitary, dtype=complex)
        """The unitary matrix representing the orbital rotation coefficients."""

        num_modes = len(self.occupation)
        if self.rotation_unitary.shape != (num_modes, num_modes):
            raise ValueError(
                f"rotation_unitary has shape {self.rotation_unitary.shape}, expected "
                f"({num_modes}, {num_modes}) to match the occupation length {num_modes}."
            )

        super().__init__("PrepareSlaterDeterminant", num_modes, [])

    def _build_definition(self) -> FermionicCircuit:
        """Builds the gate as a :class:`.FermionicCircuit` (shared by ``_define`` and the sim path).

        The definition is an :class:`.InitializeModes` reference validator followed by an
        :class:`.OrbitalRotation`, both on all modes. Walking this two-gate definition is exactly the
        validate-then-rotate semantics, which guarantees that merging an :class:`.InitializeModes`
        and an :class:`.OrbitalRotation` into this gate leaves the simulated state unchanged.
        """
        from qiskit_fermions.circuit import FermionicCircuit

        definition = FermionicCircuit(self.num_modes)
        definition.append(InitializeModes(self.occupation.tolist()), definition.modes)
        definition.append(OrbitalRotation(self.rotation_unitary), definition.modes)
        return definition

    def _define(self) -> None:
        self._definition = self._build_definition()._inner

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        """Validates the reference occupation and applies the rotation, placing modes onto ``vec``.

        Delegates to the gate's definition (:meth:`_build_definition`): the leading
        :class:`.InitializeModes` validates that ``vec`` is confined to the subspace ``occupation``
        defines and the following :class:`.OrbitalRotation` rotates it. See those gates'
        ``_apply_unitary_placed_`` for the full semantics (subspace confinement and the rejection of
        spin-mixing rotations).

        Args:
            vec: the reference state vector to validate and rotate.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions.
            copy: whether to copy the vector before operating on it.
            freg_indices: the absolute (global) mode indices that this gate's local modes map onto.

        Returns:
            The transformed vector.
        """
        return self._build_definition()._apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)
