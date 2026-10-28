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

"""The factors that the Givens-based orbital rotation syntheses have in common."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from qiskit_fermions.operators import FermionOperator

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit


def givens_block(c: float, s: complex) -> np.ndarray:
    r"""Returns the :math:`2 \times 2` unitary of a rotation as :func:`.givens_decomposition` emits it.

    The decomposition returns a rotation as the 4-tuple ``(c, s, i, j)`` of the matrix
    :math:`G = \begin{pmatrix} c & s \\ -s^* & c \end{pmatrix}`, and the circuit applies the
    element-wise *conjugate* :math:`\overline{G}` of it (see the documentation of
    :func:`.givens_decomposition`), which is what this returns.
    """
    return np.array([[c, np.conj(s)], [-s, c]], dtype=complex)


def factor_generator(block: np.ndarray, atol: float) -> FermionOperator:
    r"""Returns the Hermitian generator :math:`\sum_{ij} i \log(F)_{ij} a^\dagger_i a_j` of a factor.

    The matrix logarithm of a unitary is anti-Hermitian only up to round-off, so the result is built
    from the anti-Hermitian part of :math:`\log(F)` and is *exactly* Hermitian, as a
    :class:`~qiskit.circuit.library.PauliEvolutionGate` requires of the mapped operator. Entries at
    or below ``atol`` are dropped: ``logm`` of a sparse unitary leaves round-off in nominally-zero
    entries, and keeping them would bloat the operator with meaningless terms.

    This is correct for a block of any dimension, but it is only meant for the one- and two-mode
    blocks that :func:`append_factor` is handed today. The terms of a larger block would come out
    in no particular order (the operator does not preserve the order of its terms), which is of no
    consequence for the exponential but would not make for a structured circuit once mapped, unlike
    the single hop of a two-mode block.
    """
    log_mat = scipy.linalg.logm(block)
    hermitian = 1j * 0.5 * (log_mat - log_mat.conj().T)
    terms = {
        ((True, i), (False, j)): complex(hermitian[i, j])
        for i in range(hermitian.shape[0])
        for j in range(hermitian.shape[1])
        if abs(hermitian[i, j]) > atol
    }
    return FermionOperator.from_dict(terms)  # type: ignore[arg-type]


def append_factor(
    circuit: FermionicCircuit, block: np.ndarray, modes: Sequence[int], *, atol: float
) -> None:
    """Appends the rotation ``block`` on ``modes`` as an atomic :class:`.Evolution` at unit time.

    The operator holds a single group, because the factor is one unit that must not be split up any
    further. A factor whose generator is empty (an identity) is dropped.
    """
    from ..evolution import Evolution

    # the factor acts on its own modes only, so its generator is built on local modes 0, 1
    generator = factor_generator(block, atol)
    if len(generator) == 0:
        return
    generator.groups = [0] * len(generator)
    circuit.append(
        Evolution(len(modes), generator, time=1.0, atomic=True),  # type: ignore[arg-type]
        [circuit.modes[mode] for mode in modes],
    )
