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

"""The connected components of a commuting group, and the isomorphisms between them.

The terms of a component act on qubits that no other component touches, so the components can be
solved one by one, and a component that is a copy of another can reuse its solution.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

Edge = tuple[int, int]


class Component:
    """A connected component of a commuting group: its qubits and the terms restricted to them.

    Terms are frozensets of ``(local qubit, letter)`` with letter 1 = X, 2 = Z, 3 = Y; signs are
    irrelevant for the structure. ``color`` is an invariant of every qubit from color refinement,
    the one-dimensional Weisfeiler-Leman test, on the graph that links every term to its qubits: a
    term is recolored by the colors of its qubits and its letters on them, a qubit by the colors of
    its terms and its letters in them, until the number of colors stops growing. Isomorphic
    components have the same colors, so a bijection need only pair qubits of equal color, which
    prunes the isomorphism search.

    Args:
        x: the X bits of all terms.
        z: the Z bits of all terms.
        qubits: the qubits of the component, in increasing order.
        rows: the terms that touch it.
    """

    def __init__(self, x: np.ndarray, z: np.ndarray, qubits: Sequence[int], rows: Sequence[int]):
        self.qubits = [int(q) for q in qubits]
        local = {q: i for i, q in enumerate(self.qubits)}
        self.rows = [int(r) for r in rows]
        self.terms = [
            frozenset(
                (local[q], int(x[r, q]) + 2 * int(z[r, q]))
                for q in self.qubits
                if x[r, q] | z[r, q]
            )
            for r in self.rows
        ]
        self.size = len(self.qubits)
        self.termset = frozenset(self.terms)
        self.color = self._refine()

    def localize(self, edges: Iterable[Edge]) -> list[Edge]:
        """The pairs of ``edges`` within the component, in its local indices, sorted.

        The qubits are in increasing order, so a pair ``(u, v)`` with ``u < v`` stays ordered.
        """
        local = {q: i for i, q in enumerate(self.qubits)}
        return sorted((local[u], local[v]) for u, v in edges if u in local and v in local)

    def _refine(self) -> list[int]:
        color = [0] * self.size
        tcolor = [0] * len(self.terms)
        for step in range(self.size + 2):
            tnew = [
                hash((tcolor[i], tuple(sorted((color[q], lt) for q, lt in t))))
                for i, t in enumerate(self.terms)
            ]
            by_qubit: list[list] = [[] for _ in range(self.size)]
            for i, t in enumerate(self.terms):
                for q, lt in t:
                    by_qubit[q].append((lt, tnew[i]))
            cnew = [hash((color[q], tuple(sorted(by_qubit[q])))) for q in range(self.size)]
            stable = step > 0 and len(set(cnew)) == len(set(color))
            color, tcolor = cnew, tnew
            if stable:
                break
        return color


def components_of(x: np.ndarray, z: np.ndarray) -> list[Component]:
    """The connected components of the term-support hypergraph.

    Two qubits are linked when one term touches both.
    """
    k, n = x.shape
    support = x | z
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(k):
        qs = np.nonzero(support[i])[0]
        for q in qs[1:]:
            parent[find(int(q))] = find(int(qs[0]))
    groups: dict[int, list[int]] = {}
    for q in range(n):
        if support[:, q].any():
            groups.setdefault(find(q), []).append(q)
    return [
        Component(x, z, qs, np.nonzero(support[:, qs].any(axis=1))[0]) for qs in groups.values()
    ]


def isomorphisms(a: Component, b: Component, limit: int = 256) -> list[list[int]]:
    """Bijections ``a -> b`` of local qubits that map the terms of ``a`` onto those of ``b``.

    The terms are mapped letter for letter. When the two components have the same terms in their
    local indices (the rows of a lattice, say), the identity comes first, without a search.
    """
    if a.size != b.size or len(a.terms) != len(b.terms) or sorted(a.color) != sorted(b.color):
        return []
    identity = list(range(a.size))
    found: list[list[int]] = [identity] if a.termset == b.termset else []
    if len(found) >= limit:
        return found
    cands = [[j for j in range(b.size) if b.color[j] == a.color[i]] for i in range(a.size)]
    order = sorted(range(a.size), key=lambda i: len(cands[i]))
    term_qubits = [{q for q, _ in t} for t in a.terms]
    image: dict[int, int] = {}
    used: set[int] = set()

    def consistent() -> bool:
        for t, qs in zip(a.terms, term_qubits, strict=True):
            if (
                all(q in image for q in qs)
                and frozenset((image[q], lt) for q, lt in t) not in b.termset
            ):
                return False
        return True

    def extend(depth: int) -> None:
        if len(found) >= limit:
            return
        if depth == len(order):
            mapping = [image[i] for i in range(a.size)]
            if mapping != identity or not found or found[0] != identity:
                found.append(mapping)
            return
        i = order[depth]
        for j in cands[i]:
            if j in used:
                continue
            image[i] = j
            used.add(j)
            if consistent():
                extend(depth + 1)
            del image[i]
            used.discard(j)

    extend(0)
    return found


def isomorphism_classes(comps: Sequence[Component]) -> list[list[int]]:
    """The components grouped into classes of copies, as lists of indices.

    The first component of each class represents it.
    """
    classes: list[list[int]] = []
    for i, comp in enumerate(comps):
        for members in classes:
            if isomorphisms(comps[members[0]], comp, limit=1):
                members.append(i)
                break
        else:
            classes.append([i])
    return classes


class LocalProblem:
    """One component with the allowed edges among its qubits, in the component's local indices.

    Args:
        comp: the component.
        x: the X bits of all terms.
        z: the Z bits of all terms.
        edges: the allowed pairs, in the gate's qubit indices; those outside the component are
            dropped.
    """

    def __init__(self, comp: Component, x: np.ndarray, z: np.ndarray, edges: Iterable[Edge]):
        self.comp = comp
        self.x = x[np.ix_(comp.rows, comp.qubits)]
        self.z = z[np.ix_(comp.rows, comp.qubits)]
        self.edges: list[Edge] = comp.localize(edges)
