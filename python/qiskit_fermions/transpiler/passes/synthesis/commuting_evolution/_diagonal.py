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

"""The middle part of the evolution, ``U^dagger exp(-i t M) U``, for a given Clifford ``U``.

``M`` is diagonal (:func:`evolve_from_clifford`).
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Sequence
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

from ._layers import edge_color_layers, first_fit_layers, pairwise_tree
from .problem import Problem

ParityShape = Literal["tree", "ladder"]
"""How the parity of a Z-string is folded onto one qubit: a log-depth tree, or a nearest-neighbor
ladder."""


class NotDiagonalizedError(Exception):
    """A Clifford did not send every term to a Hermitian Z-string.

    Such a Clifford cannot build an exact evolution.

    Only the candidates whose Clifford is a guess (the layout ones) may fail this way, and they
    catch it and offer no such variant. For any other candidate it is a bug, and since the plugin
    catches no exception around the candidates, it surfaces rather than silently cost a better
    circuit. It is deliberately not a :class:`ValueError`, so that catching it cannot hide an
    unrelated error.
    """


def parity_tree(
    qubits: Collection[int], shape: ParityShape = "tree"
) -> tuple[int, list[tuple[int, int]]]:
    """A CX network folding the parity of ``qubits`` onto one of them.

    Args:
        qubits: the qubits of a Z-string.
        shape: ``"tree"`` is a balanced tree of logarithmic depth that joins qubits at distance two
            and more; ``"ladder"`` joins neighbors in sorted order and so needs only
            nearest-neighbor connectivity on a line.

    Returns:
        The qubit that carries the parity and the CX gates as ``(control, target)``, in order.
    """
    if shape == "ladder":
        ordered = sorted(qubits)
        return ordered[-1], list(itertools.pairwise(ordered))
    return pairwise_tree(list(qubits), keep="target")


def rotate_parities(
    circuit: QuantumCircuit,
    layer: Sequence[tuple[Collection[int], float]],
    shape: ParityShape,
) -> None:
    """Appends ``exp(-i angle/2 Z...Z)`` for a layer of disjoint supports.

    The supports are disjoint, so their parity trees act on disjoint qubits and run side by side
    whatever their order in the circuit.

    Args:
        circuit: the circuit to append to.
        layer: ``(support, angle)`` pairs with pairwise disjoint supports.
        shape: the parity shape.
    """
    for support, angle in layer:
        root, gates = parity_tree(support, shape)
        for control, target in gates:
            circuit.cx(control, target)
        circuit.rz(angle, root)
        for control, target in reversed(gates):
            circuit.cx(control, target)


def evolve_from_clifford(
    problem: Problem,
    clifford: QuantumCircuit,
    *,
    shape: ParityShape = "tree",
    reverse_two_body: bool = False,
) -> QuantumCircuit:
    """Builds ``exp(-i t sum c P)`` as ``U^dagger exp(-i t sum c Z-string) U``, ``U`` a Clifford.

    ``clifford`` must send every term to a Z-type Pauli (checked, and :class:`NotDiagonalizedError`
    raised otherwise).

    Args:
        problem: the commuting terms, with real coefficients, and the time ``t``.
        clifford: the circuit ``U``.
        shape: the parity shape of the rotations of weight three and more.
        reverse_two_body: run the layers of weight-two rotations in the opposite order. They
            commute, so both orders are exact; which one routes better depends on how the layers
            meet the gates of ``U``.

    Returns:
        The circuit, on ``problem.num_qubits`` qubits.

    Raises:
        NotDiagonalizedError: if ``clifford`` sends a term to a Pauli that is not a Z-string.
        NotDiagonalizedError: if ``clifford`` sends a term to a non-Hermitian Pauli (a phase of
            ``i`` or ``-i``).
    """
    cliff = Clifford(clifford)
    circuit = QuantumCircuit(problem.num_qubits)
    circuit.compose(clifford, inplace=True)

    diagonal: dict[tuple[int, ...], float] = {}
    for pauli, coeff in zip(problem.paulis, problem.coeffs, strict=True):
        mapped = pauli.evolve(cliff, frame="s")  # U P U^dagger
        if mapped.x.any():
            raise NotDiagonalizedError(f"the Clifford does not send {pauli} to a Z-string")
        if int(mapped.phase) not in (0, 2):
            raise NotDiagonalizedError(f"the Clifford sends {pauli} to a non-Hermitian Pauli")
        sign = 1.0 if int(mapped.phase) == 0 else -1.0
        support = tuple(int(i) for i in np.nonzero(mapped.z)[0])
        diagonal[support] = diagonal.get(support, 0.0) + sign * float(coeff)

    _append_diagonal(circuit, diagonal, problem.time, shape, reverse_two_body)
    circuit.compose(clifford.inverse(), inplace=True)
    return circuit


def _append_diagonal(
    circuit: QuantumCircuit,
    diagonal: dict[tuple[int, ...], float],
    time: float,
    shape: ParityShape,
    reverse_two_body: bool,
) -> None:
    """Appends ``exp(-i t sum c Z-string)`` for the Z-strings ``diagonal``.

    ``diagonal`` maps every support to its coefficient.
    """
    two: dict[tuple[int, ...], float] = {}
    many: dict[tuple[int, ...], float] = {}
    for support, c in diagonal.items():
        angle = 2.0 * time * c
        if len(support) == 0:
            circuit.global_phase -= angle / 2
        elif len(support) == 1:
            circuit.rz(angle, support[0])
        elif len(support) == 2:
            two[support] = angle
        else:
            many[support] = angle
    two_layers = edge_color_layers(list(two))  # type: ignore[arg-type]
    for layer in reversed(two_layers) if reverse_two_body else two_layers:
        for a, b in layer:
            circuit.rzz(two[a, b], a, b)
    for parity_layer in first_fit_layers(many.items(), key=lambda item: item[0]):
        rotate_parities(circuit, parity_layer, shape)
