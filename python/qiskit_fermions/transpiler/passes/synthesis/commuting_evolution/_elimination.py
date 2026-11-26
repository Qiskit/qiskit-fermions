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

"""The diagonalizing-Clifford engine of the ``plain``, ``majority`` and ``search`` candidates.

For mutually commuting Pauli terms it builds a Clifford circuit ``U`` such that ``U P U^dagger`` is
a Z-string for every term ``P``. It does so by Gaussian elimination on the X part of the terms:
every term gets a pivot qubit, CX gates clear the rest of its X part onto the pivot, an ``sdg``
removes a Y, CZ gates clear the remaining Z entries of the pivot rows (they are symmetric because
the terms commute), and an ``h`` on every pivot turns its X into a Z.

The matrix is stored as one Python integer per column (bit ``r`` is row ``r``), so a CX is two XORs
and the number of rows touching a qubit is a popcount. The per-qubit starting basis, the fan-in
shape of the CX gates and the order in which rows are eliminated are parameters; the candidates
differ only in how they choose them.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit

from ._components import components_of
from ._diagonal import evolve_from_clifford
from ._layers import edge_color_layers, pairwise_tree
from ._symplectic import (
    BASIS_GATES,
    Gate,
    append_gates,
    apply_1q_columns,
    columns,
    coordinate_descent,
)
from .problem import Problem


def schedule_cx(cxs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reorders a CX list into qubit-disjoint layers, respecting only non-commuting pairs.

    Two CX gates commute unless the control of one is the target of the other, so gates sharing a
    control, or sharing a target, may be reordered freely. Emitting them layer by layer lets the
    circuit's dependency depth reflect that, where the elimination order would serialize them.

    Every gate counts its unfinished predecessors, and a layer takes the ready gates (in index
    order) whose qubits are still free, so one round costs O(ready gates) instead of a rescan of
    everything that is left.

    Args:
        cxs: the CX gates as ``(control, target)``, in elimination order.

    Returns:
        The same gates, reordered.
    """
    n = len(cxs)
    if n < 2:
        return list(cxs)
    as_control: dict[int, list[int]] = {}
    as_target: dict[int, list[int]] = {}
    successors: list[list[int]] = [[] for _ in range(n)]
    pending = [0] * n
    for j, (c, t) in enumerate(cxs):
        before = list(as_target.get(c, ())) + list(as_control.get(t, ()))
        pending[j] = len(before)
        for i in before:
            successors[i].append(j)
        as_control.setdefault(c, []).append(j)
        as_target.setdefault(t, []).append(j)
    ready = [j for j in range(n) if pending[j] == 0]
    out: list[tuple[int, int]] = []
    while ready:
        used: set[int] = set()
        layer: list[int] = []
        rest: list[int] = []
        for j in ready:
            c, t = cxs[j]
            if c in used or t in used:
                rest.append(j)
                continue
            layer.append(j)
            used.update((c, t))
        released: list[int] = []
        for j in layer:
            for k in successors[j]:
                pending[k] -= 1
                if pending[k] == 0:
                    released.append(k)
        out.extend(cxs[j] for j in layer)
        ready = sorted(rest + released) if released else rest
    return out


def colored_row_order(supports: Sequence[Sequence[int]]) -> list[int]:
    """The order in which rows are eliminated: one greedy color class after another.

    Rows that share a qubit get different colors. Rows of one class act on disjoint qubits, so their
    eliminations are independent and can run side by side; the number of classes (2 for a line of
    overlapping windows, however long) bounds the depth, not the length.

    Args:
        supports: ``supports[r]`` lists the qubits of row ``r``.

    Returns:
        The row indices in elimination order.
    """
    colors_at: dict[int, set[int]] = {}
    color = [-1] * len(supports)
    for r, qubits in enumerate(supports):
        # the colors of the earlier rows that share a qubit with this one; they need not be
        # contiguous
        taken: set[int] = set().union(*(colors_at.get(q, ()) for q in qubits))
        c = next(c for c in itertools.count() if c not in taken)
        color[r] = c
        for q in qubits:
            colors_at.setdefault(q, set()).add(c)
    return sorted(range(len(supports)), key=lambda r: (color[r], r))


def tree_cxs(support: Sequence[int], pivot: int, taken: set[int]) -> list[tuple[int, int]]:
    """CX list clearing the X bits of ``support`` onto ``pivot`` as a balanced fan-in tree.

    The depth is ``ceil(log2(weight))`` instead of the ``weight - 1`` of a star. Qubits in ``taken``
    are the pivots of rows eliminated earlier; as a CX *control* such a qubit would write into those
    rows, so they only ever appear as targets (cleared last, from the pivot).

    Args:
        support: the qubits with an X bit in the row.
        pivot: the qubit that keeps the row's X bit.
        taken: the pivots of the rows eliminated before.

    Returns:
        The CX gates as ``(control, target)``.
    """
    movable = [pivot, *sorted(int(q) for q in support if int(q) != pivot and int(q) not in taken)]
    pinned = sorted(int(q) for q in support if int(q) in taken and int(q) != pivot)
    _, cxs = pairwise_tree(movable, keep="control")  # the pivot comes first and stays at the root
    cxs += [(pivot, q) for q in pinned]
    return cxs


def clifford_gates(
    x0: np.ndarray,
    z0: np.ndarray,
    basis: Sequence[Sequence[str]],
    *,
    tree: bool = False,
    colored: bool = False,
    schedule: bool = True,
) -> list[Gate]:
    """The gate list of a diagonalizing Clifford for the given per-qubit starting basis.

    Args:
        x0: the X bits of the terms (0/1, terms by qubits).
        z0: the Z bits of the terms.
        basis: one gate sequence per qubit, applied first (``h``, ``s`` and ``sdg`` are understood).
        tree: clear the X bits with a balanced fan-in tree instead of a star around the pivot.
        colored: eliminate the rows color class by color class instead of in their natural order.
        schedule: reorder the CX gates into qubit-disjoint layers (see :func:`schedule_cx`), instead
            of emitting them in elimination order.

    Returns:
        The gates, in circuit order, as tuples ``(name, *qubits)``.
    """
    rows, n = x0.shape
    xc, zc = columns(x0), columns(z0)
    gates: list[Gate] = []
    for q, sequence in enumerate(basis):
        for name in sequence:
            gates.append((name, q))
            apply_1q_columns(xc, zc, name, q)

    order = list(range(rows))
    if colored:
        supports: list[list[int]] = [[] for _ in range(rows)]
        for q in range(n):
            bits = xc[q] | zc[q]
            while bits:
                low = bits & -bits
                supports[low.bit_length() - 1].append(q)
                bits ^= low
        order = colored_row_order(supports)

    cxs: list[tuple[int, int]] = []
    pivots: dict[int, int] = {}
    taken: set[int] = set()
    for r in order:
        support = [q for q in range(n) if xc[q] >> r & 1]
        if not support:
            continue
        free = [c for c in support if c not in taken]
        if not free:
            # The X part of this row is spanned by earlier pivots: once those are diagonalized its
            # image is a product of their Z-type images times a pure-Z remainder, so it needs no
            # pivot of its own.
            continue
        p = min(free, key=lambda c: (xc[c].bit_count(), c))
        pairs = tree_cxs(support, p, taken) if tree else [(p, t) for t in support if t != p]
        for c, t in pairs:
            cxs.append((c, t))
            xc[t] ^= xc[c]
            zc[c] ^= zc[t]
        pivots[r] = p
        taken.add(p)
    gates += [("cx", c, t) for c, t in (schedule_cx(cxs) if schedule else cxs)]

    for r, p in pivots.items():
        if zc[p] >> r & 1:
            gates.append(("sdg", p))
            apply_1q_columns(xc, zc, "sdg", p)
    edges: set[tuple[int, int]] = set()
    for r, p in pivots.items():
        for u in range(n):
            if u != p and zc[u] >> r & 1:
                edges.add((min(p, u), max(p, u)))
    gates += [("cz", a, b) for layer in edge_color_layers(sorted(edges)) for a, b in layer]
    gates += [("h", p) for p in pivots.values()]
    return gates


def to_circuit(gates: Sequence[Gate], num_qubits: int) -> QuantumCircuit:
    """Builds a circuit from a gate list as returned by :func:`clifford_gates`."""
    circuit = QuantumCircuit(num_qubits)
    append_gates(circuit, gates)
    return circuit


ORDERS = ("elimination-order", "layered")
"""The two orders in which the CX gates of the elimination can be emitted."""


def elimination_variants(
    problem: Problem,
    basis: Sequence[Sequence[str]],
    *,
    tree: bool = False,
    colored: bool = False,
    orders: Sequence[str] = ORDERS,
) -> dict[str, QuantumCircuit]:
    """The evolution from the diagonalizing Clifford of a starting basis, in the given CX orders.

    The CX gates of the elimination can be emitted in elimination order, or reordered into
    qubit-disjoint layers (see :func:`schedule_cx`). The layered order is shallower with all-to-all
    connectivity, but after routing neither order is always the better one, so the candidates offer
    both. When two orders give the same gate sequence, only the earlier one is returned.

    Args:
        problem: the group of terms.
        basis: the per-qubit starting basis, one gate sequence per qubit.
        tree: the fan-in shape of the elimination.
        colored: the row order of the elimination.
        orders: the orders to build, from :data:`ORDERS`, in tie-break order.

    Returns:
        The circuits by order.
    """
    built: dict[str, list[Gate]] = {}
    for order in orders:
        gates = clifford_gates(
            problem.x, problem.z, basis, tree=tree, colored=colored, schedule=order == "layered"
        )
        if gates not in built.values():
            built[order] = gates
    n = problem.num_qubits
    return {
        order: evolve_from_clifford(problem, to_circuit(gates, n)) for order, gates in built.items()
    }


def gate_cost(gates: Sequence[Gate]) -> tuple[int, int]:
    """The ``(two-qubit gate count, two-qubit depth)`` of a gate list, in emission order."""
    level: dict[int, int] = {}
    count = depth = 0
    for g in gates:
        if len(g) == 3:
            count += 1
            lvl = max(level.get(g[1], 0), level.get(g[2], 0)) + 1
            level[g[1]] = level[g[2]] = lvl
            depth = max(depth, lvl)
    return count, depth


def _descend(
    sx: np.ndarray, sz: np.ndarray, max_sweeps: int, tree: bool, colored: bool
) -> list[int]:
    """Coordinate descent over the six per-qubit bases of one connected component."""
    m = sx.shape[1]

    def score(
        choice: list[int], _incumbent: tuple[tuple[int, int]] | None
    ) -> tuple[tuple[int, int]]:
        gates = clifford_gates(sx, sz, [BASIS_GATES[b] for b in choice], tree=tree, colored=colored)
        return (gate_cost(gates),)

    basis, _ = coordinate_descent(m, range(m), score, max_sweeps)
    return basis


def search_basis(
    x: np.ndarray, z: np.ndarray, *, max_sweeps: int = 4, tree: bool = False, colored: bool = False
) -> list[int]:
    """Per-qubit bases (indices into ``BASIS_GATES``) that minimize the Clifford's two-qubit cost.

    The cost is the two-qubit gate count, then the two-qubit depth.

    The terms are split into connected components (rows sharing a qubit). Components act on disjoint
    qubits, so each is optimized on its own, and identical components (the lines of a lattice, say)
    are optimized once and reused.

    Args:
        x: the X bits of the terms.
        z: the Z bits of the terms.
        max_sweeps: the most passes of the coordinate descent over the qubits of a component.
        tree: the fan-in shape of the elimination being optimized (see :func:`clifford_gates`).
        colored: the row order of the elimination being optimized.

    Returns:
        One index into ``BASIS_GATES`` per qubit.
    """
    basis = [0] * x.shape[1]
    cache: dict[tuple, list[int]] = {}
    for comp in components_of(x, z):
        sx, sz = x[np.ix_(comp.rows, comp.qubits)], z[np.ix_(comp.rows, comp.qubits)]
        key = (sx.shape, sx.tobytes(), sz.tobytes())
        if key not in cache:
            cache[key] = _descend(sx, sz, max_sweeps, tree, colored)
        for col, b in zip(comp.qubits, cache[key], strict=True):
            basis[col] = b
    return basis
