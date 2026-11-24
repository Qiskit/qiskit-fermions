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

"""Symplectic (GF(2)) bookkeeping of Pauli terms under Clifford gates.

A group of Pauli terms is stored as two 0/1 matrices ``x`` and ``z`` (terms by qubits). Signs are
not tracked here; they are recovered once, from the final Clifford, when the diagonal part is built.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping

from ._layers import edge_color_layers

Gate = tuple  # ("h", q) / ("sdg", q) / ("cx", control, target) / ("cz", a, b)
"""A gate as a tuple: its name followed by its qubits."""

_STANDARD_GATES = get_standard_gate_name_mapping()


def append_gates(circuit: QuantumCircuit, gates: Iterable[Gate]) -> None:
    """Appends gates given as :data:`Gate` tuples.

    Each gate is a parameterless standard gate, named as in Qiskit.
    """
    for name, *qubits in gates:
        circuit.append(_STANDARD_GATES[name], qubits)


BASIS_GATES: tuple[tuple[str, ...], ...] = (
    (),
    ("h",),
    ("s",),
    ("s", "h"),
    ("h", "s"),
    ("h", "s", "h"),
)
"""The six single-qubit Cliffords modulo Paulis (all permutations of the letters X, Y, Z), as gate
sequences."""

LETTER_TO_Z: dict[int, tuple[str, ...]] = {0: ("h",), 1: ("sdg", "h"), 2: ()}
"""The gates that turn a qubit's letter (0 = X, 1 = Y, 2 = Z) into Z."""

LETTER_FROM_Z: dict[int, tuple[str, ...]] = {0: ("h",), 1: ("h", "s"), 2: ()}
"""The inverse of :data:`LETTER_TO_Z`: the gates that turn Z back into the letter."""

LETTER_OF_BITS: dict[tuple[int, int], int] = {(1, 0): 0, (1, 1): 1, (0, 1): 2}
"""The letter (0 = X, 1 = Y, 2 = Z) of a qubit from its ``(x, z)`` bits."""


def apply_cx(x: np.ndarray, z: np.ndarray, control: int, target: int) -> None:
    """Conjugates the rows ``(x, z)`` by ``CX(control, target)``, in place."""
    x[:, target] ^= x[:, control]
    z[:, control] ^= z[:, target]


def apply_cz(x: np.ndarray, z: np.ndarray, a: int, b: int) -> None:
    """Conjugates the rows ``(x, z)`` by ``CZ(a, b)``, in place.

    Each qubit picks up a Z where the other has an X.
    """
    z[:, a] ^= x[:, b]
    z[:, b] ^= x[:, a]


def apply_1q(x: np.ndarray, z: np.ndarray, name: str, qubit: int) -> None:
    """Conjugates the rows ``(x, z)`` by an ``h``, ``s`` or ``sdg`` gate, in place."""
    if name == "h":
        x[:, qubit], z[:, qubit] = z[:, qubit].copy(), x[:, qubit].copy()
    else:  # s / sdg: X <-> Y
        z[:, qubit] ^= x[:, qubit]


def apply_1q_columns(xc: list[int], zc: list[int], name: str, qubit: int) -> None:
    """:func:`apply_1q` on bit-packed columns (see :func:`columns`), in place."""
    if name == "h":
        xc[qubit], zc[qubit] = zc[qubit], xc[qubit]
    else:  # s / sdg: X <-> Y
        zc[qubit] ^= xc[qubit]


def apply_gates(
    x: np.ndarray, z: np.ndarray, gates: Sequence[Gate]
) -> tuple[np.ndarray, np.ndarray]:
    """The images ``(x, z)`` of the terms under the Clifford gates, as copies.

    The gates are ``h``, ``s``, ``sdg``, ``cx`` and ``cz``.
    """
    bx, bz = x.copy(), z.copy()
    for name, *qubits in gates:
        if name == "cx":
            apply_cx(bx, bz, *qubits)
        elif name == "cz":
            apply_cz(bx, bz, *qubits)
        else:
            apply_1q(bx, bz, name, qubits[0])
    return bx, bz


def apply_layer(
    x: np.ndarray, z: np.ndarray, layer: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """The images ``(x, z)`` under a per-qubit layer, as copies.

    The layer holds one index into :data:`BASIS_GATES` per qubit.
    """
    bx, bz = x.copy(), z.copy()
    for q, b in enumerate(layer):
        for name in BASIS_GATES[b]:
            apply_1q(bx, bz, name, q)
    return bx, bz


def layer_size(layer: Sequence[int]) -> int:
    """The number of single-qubit gates of a per-qubit layer."""
    return sum(len(BASIS_GATES[b]) for b in layer)


def layer_gates(
    layer: Sequence[int],
    gamma: Sequence[tuple[int, int]] = (),
    letters: Sequence[int] | None = None,
) -> list[Gate]:
    """The gates of ``L``, ``CZ(gamma)`` in edge-colored layers, and those sending ``letters`` to Z.

    Args:
        layer: one index into :data:`BASIS_GATES` per qubit.
        gamma: the CZ pairs.
        letters: one letter per qubit (0 = X, 1 = Y, 2 = Z), or ``None`` for no final layer.

    Returns:
        The gates, in circuit order.
    """
    gates: list[Gate] = [(name, q) for q, b in enumerate(layer) for name in BASIS_GATES[b]]
    gates += [("cz", u, v) for lay in edge_color_layers(sorted(gamma)) for u, v in lay]
    if letters is not None:
        gates += [(name, q) for q, a in enumerate(letters) for name in LETTER_TO_Z[a]]
    return gates


def relabel(gates: Sequence[Gate], qubits: Mapping[int, int] | Sequence[int]) -> list[Gate]:
    """The gates with every qubit ``q`` replaced by ``qubits[q]``."""
    return [(name, *(qubits[q] for q in on)) for name, *on in gates]


Score = TypeVar("Score", bound=tuple)


def coordinate_descent(
    num_qubits: int,
    qubits: Sequence[int],
    score: Callable[[list[int], Score | None], Score],
    max_sweeps: int,
) -> tuple[list[int], Score]:
    """Coordinate descent over the six bases of every qubit in ``qubits``, from the identity layer.

    A sweep tries every other basis on one qubit at a time and keeps a trial whose score is smaller;
    the descent stops after ``max_sweeps`` sweeps or after a sweep without improvement.

    Args:
        num_qubits: the length of the layer.
        qubits: the qubits whose basis may change, in the order they are tried.
        score: ``score(layer, incumbent)`` for a trial layer and the best score so far (``None`` for
            the start); it returns a tuple whose first entry is compared, so that the rest can carry
            the solution along.
        max_sweeps: the upper bound on the number of sweeps, reached only if every sweep continues
            to result in an improvement.

    Returns:
        The best layer and its score.
    """
    layer = [0] * num_qubits
    best = score(layer, None)
    for _ in range(max_sweeps):
        improved = False
        for q in qubits:
            for b in range(len(BASIS_GATES)):
                if b == layer[q]:
                    continue
                trial = list(layer)
                trial[q] = b
                candidate = score(trial, best)
                if candidate[0] < best[0]:
                    layer, best, improved = trial, candidate, True
        if not improved:
            break
    return layer, best


def columns(a: np.ndarray) -> list[int]:
    """The columns of a 0/1 matrix as Python integers (bit ``r`` of entry ``q`` is ``a[r, q]``)."""
    packed = np.packbits(np.ascontiguousarray(a.T).astype(np.uint8), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def majority_basis(x: np.ndarray, z: np.ndarray) -> list[tuple[str, ...]]:
    """Single-qubit Cliffords that send each qubit's majority letter to Z.

    Ties prefer Z, then X.

    Args:
        x: the X bits of the terms, terms by qubits.
        z: the Z bits of the terms.

    Returns:
        One gate sequence per qubit, from :data:`LETTER_TO_Z`.
    """
    xb, zb = x.astype(bool), z.astype(bool)
    pre: list[tuple[str, ...]] = []
    for q in range(x.shape[1]):
        nx = int(np.sum(xb[:, q] & ~zb[:, q] & 1))
        ny = int(np.sum(xb[:, q] & zb[:, q]))
        nz = int(np.sum(~xb[:, q] & zb[:, q] & 1))
        # (count, tie preference: Z, then X, then Y, letter)
        _, _, letter = max((nz, 2, 2), (nx, 1, 0), (ny, 0, 1))
        pre.append(LETTER_TO_Z[letter])
    return pre
