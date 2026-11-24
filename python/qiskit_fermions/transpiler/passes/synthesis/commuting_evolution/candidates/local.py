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

"""The local candidates: no shared Clifford, every term is diagonalized on its own."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit

from .._diagonal import ParityShape, rotate_parities
from .._layers import first_fit_layers
from .._symplectic import LETTER_FROM_Z, LETTER_OF_BITS, LETTER_TO_Z, append_gates
from ..problem import Context, Problem
from .base import validated_variants


def synthesize_local(problem: Problem, shape: ParityShape) -> QuantumCircuit:
    """Diagonalizes each term locally and packs qubit-disjoint terms into layers.

    Exact because all terms commute, so any order is allowed. This is the right candidate when the
    terms overlap in a chain, where a shared Clifford degenerates into a serial CX ladder.

    Args:
        problem: the group of terms.
        shape: the parity shape of the rotations.

    Returns:
        The circuit.
    """
    circuit = QuantumCircuit(problem.num_qubits)
    terms: list[tuple[tuple[int, ...], list[int], float]] = []
    for x, z, coeff in zip(problem.x, problem.z, problem.coeffs, strict=True):
        angle = 2.0 * problem.time * float(coeff)
        support = tuple(int(q) for q in np.nonzero(x | z)[0])
        if not support:
            circuit.global_phase -= angle / 2
            continue
        terms.append((support, [LETTER_OF_BITS[int(x[q]), int(z[q])] for q in support], angle))
    for layer in first_fit_layers(terms, key=lambda term: term[0]):
        for support, letters, _ in layer:
            append_gates(
                circuit,
                (
                    (name, q)
                    for q, a in zip(support, letters, strict=True)
                    for name in LETTER_TO_Z[a]
                ),
            )
        rotate_parities(circuit, [(support, angle) for support, _, angle in layer], shape)
        for support, letters, _ in layer:
            append_gates(
                circuit,
                (
                    (name, q)
                    for q, a in zip(support, letters, strict=True)
                    for name in LETTER_FROM_Z[a]
                ),
            )
    return circuit


VARIANTS: tuple[ParityShape, ...] = ("tree", "ladder")
"""The variants of :class:`Local`, which are its parity shapes."""


class Local:
    """Each term diagonalized on its own, with no shared Clifford.

    Terms on disjoint qubits are packed into layers.

    Cheap, and the stock recipe on groups without structure.

    **Variants:** ``"tree"`` (log-depth parity trees) and ``"ladder"`` (nearest-neighbor parity
    ladders in sorted qubit order).

    Args:
        variants: the variants to build.

    Raises:
        ValueError: if ``variants`` is empty or names an unknown variant.
    """

    name = "local"

    def __init__(self, *, variants: Sequence[ParityShape] = VARIANTS) -> None:  # noqa: D107
        self.variants = validated_variants(variants, VARIANTS)

    def generate(self, problem: Problem, context: Context) -> dict[str, QuantumCircuit]:  # noqa: D102
        return {shape: synthesize_local(problem, shape) for shape in self.variants}
