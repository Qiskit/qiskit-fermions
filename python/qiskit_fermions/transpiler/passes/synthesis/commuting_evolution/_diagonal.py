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

``M`` is diagonal (:func:`evolve_from_clifford`), or a layer of generic two-qubit blocks next to a
diagonal (:func:`evolve_with_pair_blocks`).
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Sequence
from functools import cache
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate, RYGate
from qiskit.quantum_info import Clifford, Pauli
from qiskit.synthesis import TwoQubitBasisDecomposer

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


_LETTERS = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
"""The Pauli letter of ``(x, z)`` bits."""

_ROTATIONS = {"X": RXGate, "Y": RYGate}
"""The rotation gates of the single-qubit images ``X`` and ``Y`` outside every block."""


@cache
def _two_qubit_decomposer() -> TwoQubitBasisDecomposer:
    """The exact KAK synthesis of a two-qubit unitary with at most three CX gates.

    It is built once, on first use.
    """
    return TwoQubitBasisDecomposer(CXGate())


def evolve_with_pair_blocks(
    problem: Problem,
    clifford: QuantumCircuit,
    *,
    shape: ParityShape = "tree",
) -> QuantumCircuit:
    """Builds ``exp(-i t sum c P)`` as ``U^dagger exp(-i t M) U``, with two-qubit blocks in ``M``.

    Every image ``U P U^dagger`` must be a Z-string or act on at most two qubits (checked, and
    :class:`NotDiagonalizedError` raised otherwise). The images on one or two qubits that are not
    Z-strings form one block per support. A Z-string on exactly the qubits of a two-qubit block, and
    an image on one qubit of it, join the block, which is then synthesized as one exact two-qubit
    unitary (at most three CX gates). A single-qubit image outside every block becomes a
    single-qubit rotation, and the remaining Z-strings are the diagonal of
    :func:`evolve_from_clifford`. All terms commute, so the blocks and the diagonal can be applied
    in any order.

    A block is worth it when the images on a pair form an entangled group such as ``{XX, YY, ZZ}``:
    the group cannot be diagonalized by single-qubit gates, so the diagonal form would need an
    entangling gate on each side of the rotation stage, while one block with all three coefficients
    costs three CX gates.

    Args:
        problem: the commuting terms, with real coefficients, and the time ``t``.
        clifford: the circuit ``U``.
        shape: the parity shape of the diagonal rotations of weight three and more.

    Returns:
        The circuit, on ``problem.num_qubits`` qubits.

    Raises:
        NotDiagonalizedError: if ``clifford`` sends a term to a non-Hermitian Pauli (a phase of
            ``i`` or ``-i``).
        NotDiagonalizedError: if ``clifford`` sends a term to a Pauli on three or more qubits that
            is not a Z-string.
    """
    time = problem.time
    cliff = Clifford(clifford)
    circuit = QuantumCircuit(problem.num_qubits)
    circuit.compose(clifford, inplace=True)

    diagonal: dict[tuple[int, ...], float] = {}
    # support -> Pauli label on it -> coefficient
    local: dict[tuple[int, ...], dict[str, float]] = {}
    for pauli, coeff in zip(problem.paulis, problem.coeffs, strict=True):
        mapped = pauli.evolve(cliff, frame="s")  # U P U^dagger
        if int(mapped.phase) not in (0, 2):
            raise NotDiagonalizedError(f"the Clifford sends {pauli} to a non-Hermitian Pauli")
        sign = 1.0 if int(mapped.phase) == 0 else -1.0
        support = tuple(int(i) for i in np.nonzero(mapped.x | mapped.z)[0])
        if not mapped.x.any():
            diagonal[support] = diagonal.get(support, 0.0) + sign * float(coeff)
        elif len(support) <= 2:
            # a Qiskit label lists the highest qubit first
            label = "".join(_LETTERS[int(mapped.x[q]), int(mapped.z[q])] for q in reversed(support))
            block = local.setdefault(support, {})
            block[label] = block.get(label, 0.0) + sign * float(coeff)
        else:
            raise NotDiagonalizedError(
                f"the Clifford sends {pauli} to neither a Z-string"
                " nor a Pauli on at most two qubits"
            )

    blocks = {s: dict(terms) for s, terms in sorted(local.items()) if len(s) == 2}
    owner: dict[int, tuple[int, ...]] = {}
    for s in blocks:
        for q in s:
            owner.setdefault(q, s)
    for s, block in blocks.items():
        if s in diagonal:
            block["ZZ"] = block.get("ZZ", 0.0) + diagonal.pop(s)
    for (q,), terms in sorted((s, t) for s, t in local.items() if len(s) == 1):
        if q in owner:
            a, b = owner[q]
            block = blocks[a, b]
            for letter, c in terms.items():
                padded = f"I{letter}" if q == a else f"{letter}I"
                block[padded] = block.get(padded, 0.0) + c
        else:
            for letter, c in terms.items():  # exp(-i t c X) = RX(2 t c), likewise for Y
                circuit.append(_ROTATIONS[letter](2.0 * time * c), [q])

    eye = np.eye(4, dtype=complex)
    for (a, b), block in blocks.items():
        unitary = eye
        for label, c in block.items():  # commuting factors: exp(-i t c P) = cos(t c) - i sin(t c) P
            theta = time * c
            unitary = (
                np.cos(theta) * eye - 1j * np.sin(theta) * Pauli(label).to_matrix()
            ) @ unitary
        circuit.compose(_two_qubit_decomposer()(unitary), qubits=[a, b], inplace=True)

    _append_diagonal(circuit, diagonal, time, shape, reverse_two_body=False)
    circuit.compose(clifford.inverse(), inplace=True)
    return circuit
