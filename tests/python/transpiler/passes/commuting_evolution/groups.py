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

"""Groups of mutually commuting Pauli terms for the tests, each commuting by construction.

The structured groups are the flow sets of the Fermi-Hubbard models the plugin was developed for.
Their terms are the images of fermionic hops and on-site interactions under the one-ancilla
parity-delocalized encoding, written down directly as Pauli strings so that the tests do not depend
on the fermionic stack.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from qiskit.quantum_info import Operator, SparsePauliOp, random_clifford


def _coefficients(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 1.0, count) * rng.choice([-1.0, 1.0], count)


def _from_terms(terms: list[tuple[str, list[int]]], num_qubits: int, seed: int) -> SparsePauliOp:
    coeffs = _coefficients(len(terms), seed)
    return SparsePauliOp.from_sparse_list(
        [(label, qubits, c) for (label, qubits), c in zip(terms, coeffs, strict=True)],
        num_qubits=num_qubits,
    )


def zxz_windows(num_qubits: int = 7, seed: int = 0) -> SparsePauliOp:
    """The backward hops of a chain: ``Z X Z`` on consecutive windows. Any two overlapping windows
    commute."""
    terms = [("ZXZ", [k - 1, k, k + 1]) for k in range(1, num_qubits - 1)]
    return _from_terms(terms, num_qubits, seed)


def bare_x(num_qubits: int = 5, seed: int = 1) -> SparsePauliOp:
    """The forward hops of a chain: single-qubit ``X`` terms."""
    return _from_terms([("X", [k]) for k in range(1, num_qubits)], num_qubits, seed)


def yy_pairs(num_qubits: int = 7, seed: int = 2) -> SparsePauliOp:
    """The forward hops of an interleaved chain: ``Y Y`` on neighbors."""
    return _from_terms([("YY", [k, k + 1]) for k in range(1, num_qubits - 1)], num_qubits, seed)


def zxxz_windows(num_qubits: int = 8, seed: int = 3) -> SparsePauliOp:
    """The backward hops of an interleaved chain: ``Z X X Z`` on consecutive windows."""
    return _from_terms(
        [("ZXXZ", [k - 1, k, k + 1, k + 2]) for k in range(1, num_qubits - 2)], num_qubits, seed
    )


def zz_interactions(num_qubits: int = 7, seed: int = 4) -> SparsePauliOp:
    """Diagonal two-body terms on neighbors and next neighbors."""
    terms = [("ZZ", [k, k + 1]) for k in range(num_qubits - 1)] + [
        ("ZZ", [k, k + 2]) for k in range(num_qubits - 2)
    ]
    return _from_terms(terms, num_qubits, seed)


def plaquettes(num_sites: int = 3, seed: int = 5) -> SparsePauliOp:
    """The on-site terms of a two-chain block-ordered Hubbard model: ``Z Z Z Z`` on 2 x 2
    plaquettes.

    The chains are qubits ``0 .. L`` and ``L + 1 .. 2L + 1`` and the rungs join qubit ``q`` to
    ``q + L + 1``. The qubits of a rung occur in exactly the same terms, which is what ``twin-fold``
    exploits.
    """
    offset = num_sites + 1
    terms = [("ZZZZ", [j, j + 1, j + offset, j + 1 + offset]) for j in range(num_sites)]
    return _from_terms(terms, 2 * num_sites + 2, seed)


def disjoint_windows(copies: int = 2, qubits_per_copy: int = 4, seed: int = 7) -> SparsePauliOp:
    """Several copies of a ``Z X Z`` window set on disjoint blocks of qubits (components that are
    isomorphic)."""
    terms = []
    for c in range(copies):
        base = c * qubits_per_copy
        terms += [
            ("ZXZ", [base + k - 1, base + k, base + k + 1]) for k in range(1, qubits_per_copy - 1)
        ]
    return _from_terms(terms, copies * qubits_per_copy, seed)


def random_commuting(num_qubits: int = 6, num_terms: int = 8, seed: int = 6) -> SparsePauliOp:
    """A random commuting group: random Z-strings conjugated by a random Clifford (no lattice
    structure at all)."""
    rng = np.random.default_rng(seed)
    clifford = random_clifford(num_qubits, seed=int(rng.integers(1 << 30)))
    terms = []
    for _ in range(num_terms):
        weight = int(rng.integers(1, min(4, num_qubits) + 1))
        terms.append(
            ("Z" * weight, [int(q) for q in rng.choice(num_qubits, weight, replace=False)])
        )
    z_strings = _from_terms(terms, num_qubits, seed)
    return SparsePauliOp(
        [p.evolve(clifford, frame="h") for p in z_strings.paulis], z_strings.coeffs
    ).simplify()


def vertical_bond(seed: int = 8) -> SparsePauliOp:
    """A vertical bond of spinless fermions on a hexagonal lattice, in a flow-set encoding on a
    square qubit grid.

    Hop and density-density terms ``X_1 X_2 X_a``, ``Y_1 Y_2 X_a``, ``Z_1 Z_2`` on the wires
    ``(q_1, q_2, q_a)``, which form a line ``q_1 - q_2 - q_a`` on the device. One CZ on
    ``(q_a, q_2)`` turns them into the entangled set ``{XX, YY, ZZ}`` on the device edge
    ``(q_1, q_2)``.
    """
    return _from_terms([("XXX", [0, 1, 2]), ("YYX", [0, 1, 2]), ("ZZ", [0, 1])], 3, seed)


def horizontal_bond(seed: int = 9) -> SparsePauliOp:
    """A horizontal bond of the same lattice: ``Y_1 Y_a Y_2``, ``X_1 Y_a X_2``, ``Z_1 Z_2`` on the
    line ``q_1 - q_a - q_2``.

    One CZ layer can only bring the terms onto the non-edge ``(q_1, q_2)``; an fSWAP move brings
    them onto the edge ``(q_a, q_2)``.
    """
    return _from_terms([("YYY", [0, 1, 2]), ("XYX", [0, 1, 2]), ("ZZ", [0, 2])], 3, seed)


def hop_cell(seed: int = 10) -> SparsePauliOp:
    """Same-spin hops and same-spin density-density terms of two sites of an interleaved spinful
    Jordan-Wigner chain.

    Qubits ``(0 up, 0 down, 1 up, 1 down)``: ``X Z X``, ``Y Z Y`` on ``(0, 1, 2)`` and
    ``(1, 2, 3)``, and ``Z Z`` on ``(0, 2)`` and ``(1, 3)``. One CZ on ``(1, 2)`` turns them into
    ``{XX, YY, ZZ}`` on ``(0, 2)`` and on ``(1, 3)``.
    """
    terms = [
        ("XZX", [0, 1, 2]),
        ("YZY", [0, 1, 2]),
        ("XZX", [1, 2, 3]),
        ("YZY", [1, 2, 3]),
        ("ZZ", [0, 2]),
        ("ZZ", [1, 3]),
    ]
    return _from_terms(terms, 4, seed)


def non_commuting() -> SparsePauliOp:
    """Two terms that anticommute."""
    return SparsePauliOp.from_sparse_list([("XX", [0, 1], 1.0), ("ZI", [0, 1], 0.5)], num_qubits=2)


#: Structured and unstructured groups with at most eight qubits, so that dense comparisons
#: stay cheap.
GROUPS = {
    "zxz_windows": zxz_windows,
    "bare_x": bare_x,
    "yy_pairs": yy_pairs,
    "zxxz_windows": zxxz_windows,
    "zz_interactions": zz_interactions,
    "plaquettes": plaquettes,
    "disjoint_windows": disjoint_windows,
    "random": random_commuting,
    "vertical_bond": vertical_bond,
    "horizontal_bond": horizontal_bond,
    "hop_cell": hop_cell,
}


def commute(op: SparsePauliOp) -> bool:
    """Whether all terms of ``op`` mutually commute."""
    x = op.paulis.x.astype(int)
    z = op.paulis.z.astype(int)
    return not ((x @ z.T + z @ x.T) % 2).any()


def evolution_matrix(op: SparsePauliOp, time: float) -> np.ndarray:
    """The matrix of ``exp(-i time op)``."""
    return scipy.linalg.expm(-1j * time * op.to_matrix())


def line_edges(num_qubits: int) -> list[tuple[int, int]]:
    """The edges of a line of ``num_qubits`` qubits."""
    return [(k, k + 1) for k in range(num_qubits - 1)]


def same_unitary(circuit, op: SparsePauliOp, time: float, atol: float = 1e-10) -> bool:
    """Whether ``circuit`` equals ``exp(-i time op)`` (exactly, including the global phase)."""
    return bool(np.allclose(Operator(circuit).data, evolution_matrix(op, time), atol=atol))
