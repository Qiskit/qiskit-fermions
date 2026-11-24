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

"""The Cliffords of the pair-block variants of ``cz-local``.

The diagonal form ``U^dagger exp(-i t D) U`` needs every term to end as a Z-string. When the middle
may also hold generic two-qubit blocks (:func:`.evolve_with_pair_blocks`), a weaker target suffices:
every term must end on one qubit or on one allowed pair. The difference matters for groups whose
terms on a pair form an entangled set such as ``{XX, YY, ZZ}``: no single-qubit gate diagonalizes
it, so the diagonal form pays an entangling gate on each side of the rotation stage, while one block
with three nonzero coefficients costs three CX gates. Moreover, a Clifford that only has to
*decouple* the pairs from each other can be much cheaper than one that diagonalizes them.

The one-layer search (:func:`one_layer_solutions`) is an exact branch and bound over a single-qubit
layer and a CZ matching. The fSWAP search tries layers of fSWAP moves on the allowed pairs (see
:func:`fswap_gates`). Both work one component of the group at a time, and all their gates and blocks
are on allowed pairs, so the circuits need no routing and leave the layout unchanged.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from functools import cache, lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from . import _budget
from ._budget import NodeBudget
from ._components import (
    Component,
    Edge,
    LocalProblem,
    components_of,
    isomorphism_classes,
    isomorphisms,
)
from ._layers import sorted_pair
from ._symplectic import (
    BASIS_GATES,
    Gate,
    apply_gates,
    apply_layer,
    layer_gates,
    relabel,
)

ONE_LAYER_CACHE = 256
"""The most components whose one-layer solution is remembered (both pair-block variants of
``cz-local`` need it)."""

Key = tuple[int, int, int]
"""``(estimated two-qubit gates, CZ gates, single-qubit gates)``, smaller is better."""


@cache
def single_qubit_cliffords() -> tuple[tuple[int, int, int, int, int], ...]:
    """The six single-qubit Cliffords as ``(index into BASIS_GATES, a, b, c, d)``.

    A Clifford sends the bits of a qubit to ``x' = ax + bz, z' = cx + dz``.
    """
    out = []
    for idx in range(len(BASIS_GATES)):
        cols = []
        for xv, zv in ((1, 0), (0, 1)):
            xx, zz = apply_layer(np.array([[xv]], np.uint8), np.array([[zv]], np.uint8), [idx])
            cols.append((int(xx[0, 0]), int(zz[0, 0])))
        (a, c), (b, d) = cols
        out.append((idx, a, b, c, d))
    return tuple(out)


def column_images(
    x16: np.ndarray, z16: np.ndarray, q: int
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """``(x, z, index)`` of qubit ``q``'s column under each of the six single-qubit Cliffords."""
    return [
        ((a * x16[:, q] + b * z16[:, q]) & 1, (c * x16[:, q] + d * z16[:, q]) & 1, idx)
        for idx, a, b, c, d in single_qubit_cliffords()
    ]


def pair_images(
    x16: np.ndarray, z16: np.ndarray, u: int, v: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]:
    """``(xu, zu, xv, zv, index_u, index_v)`` of the columns of ``u < v`` under every option.

    An option is a pair of single-qubit Cliffords followed by ``CZ(u, v)``; the Clifford of ``u`` is
    the outer loop.
    """
    assert u < v
    right = column_images(x16, z16, v)
    for xu, zu, iu in column_images(x16, z16, u):
        for xv, zv, iv in right:
            # CZ: each qubit picks up a Z where the other has an X
            yield xu, zu ^ xv, xv, zv ^ xu, iu, iv


def row_masks(bits: np.ndarray) -> list[int]:
    """Every row of a 0/1 matrix as a bit mask over its columns (bit ``q`` for column ``q``).

    The masks are Python integers, so a component may have any number of qubits.
    """
    packed = np.packbits(np.ascontiguousarray(bits, np.uint8), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def support_masks(x: np.ndarray, z: np.ndarray) -> list[int]:
    """The support of every term as a bit mask over the qubits (bit ``q`` for qubit ``q``)."""
    return row_masks(x | z)


def pair_masks(edges: Sequence[Edge]) -> set[int]:
    """The pairs as bit masks over the qubits."""
    return {(1 << u) | (1 << v) for u, v in edges}


def pairs_ok(masks: Iterable[int], allowed: set[int]) -> bool:
    """Whether every support is one qubit or one allowed pair, and the distinct pairs disjoint.

    The supports are bit masks (see :func:`support_masks`).

    This is the target of the pair-block variants: a term supported on one qubit or one allowed pair
    is a single-qubit rotation or part of a two-qubit block.
    """
    taken = 0
    pairs: set[int] = set()
    for s in masks:
        count = s.bit_count()
        if count > 2:
            return False
        if count == 2 and s not in pairs:
            if s not in allowed or taken & s:
                return False
            taken |= s
            pairs.add(s)
    return True


def bandwidth_order(x: np.ndarray, z: np.ndarray) -> list[int]:
    """A qubit order in which the qubits of every term lie close together.

    It is the reverse Cuthill-McKee order of the graph that links two qubits when a term acts on
    both. Assigned in this order, a term's qubits are all assigned soon after its first one, so the
    one-layer search can reject a term early, and few terms are open at any point of it.
    """
    support = (x | z).astype(np.int64)
    adjacency = csr_matrix((support.T @ support) > 0)
    return [int(q) for q in reverse_cuthill_mckee(adjacency, symmetric_mode=True)]


def one_layer_solutions(
    x: np.ndarray, z: np.ndarray, edges: Sequence[Edge], node_limit: int
) -> Iterator[tuple[Key, list[int], list[Edge]]]:
    """Yields ``(key, layer, gamma)`` for ever cheaper ``U = CZ(gamma) L`` that put terms on pairs.

    Such a ``U`` puts every term on one qubit or one edge. ``L`` is a single-qubit layer and
    ``gamma`` a matching of ``edges``; the key is the :data:`Key` of the solution. Every solution
    yielded is cheaper than the ones before it, so the last one is the cheapest; among equally
    cheap solutions it is the first in the enumeration order. The search runs in
    :func:`bandwidth_order`, and its results are in the qubit indices of ``x`` and ``z``.

    Args:
        x: the X bits of the component's terms.
        z: the Z bits.
        edges: the allowed pairs.
        node_limit: the node budget of the search.

    Raises:
        TimeoutError: from the iteration, once the node budget is exhausted.
    """
    order = bandwidth_order(x, z)
    position = {q: i for i, q in enumerate(order)}
    local = sorted(sorted_pair((position[u], position[v])) for u, v in edges)
    found = _one_layer_search(
        np.ascontiguousarray(x[:, order]), np.ascontiguousarray(z[:, order]), local, node_limit
    )
    for key, layer, gamma in found:
        yield (
            key,
            [layer[position[q]] for q in range(len(order))],
            [sorted_pair((order[u], order[v])) for u, v in gamma],
        )


def _one_layer_search(
    x: np.ndarray, z: np.ndarray, edges: Sequence[Edge], node_limit: int
) -> Iterator[tuple[Key, list[int], list[Edge]]]:
    """:func:`one_layer_solutions` in the given qubit order.

    With ``gamma`` a matching, the images on a qubit depend only on its own Clifford and its
    partner's, so the problem splits into one block per edge of ``gamma`` (at most 36 pairs of
    single-qubit Cliffords) and one per unpaired qubit (at most six), as in the hardware-tailored
    diagonalization of Miller et al. (arXiv:2203.03646). A depth-first search assigns the blocks
    qubit by qubit. The target (see :func:`pairs_ok`) only involves the qubits assigned so far and
    never loosens when more are assigned, so it prunes a branch as soon as it fails.

    The search also prunes by dominance. A term is *settled* once every qubit that can still change
    its image (its own qubits, and their partners on allowed pairs) is assigned. What the rest of
    the search can do depends only on the *frontier*: the next qubit, the qubits assigned ahead of
    it (the partners of earlier ones), the images of the unsettled terms on the assigned qubits, and
    the image sets of the pairs that unsettled terms can still reach. The cost of everything else is
    fixed, and the cost adds up over the two parts. So a branch that reaches a frontier already
    reached at no higher cost cannot lead to a cheaper solution, nor to an equally cheap one earlier
    in the enumeration, and is cut. The images on an assigned qubit are only ever compared for
    equality later on, so the frontier names each qubit's letters in the order they first occur:
    assignments that differ by a relabeling of letters reach the same frontier.

    Without the dominance, the solutions of a component that repeats along a line multiply with
    every repetition, almost all of them relabelings of each other. With it, the search over such a
    component grows about linearly with its length. Nothing is transpiled and no random starts are
    used.
    """
    k, m = x.shape
    x16, z16 = x.astype(np.int16), z.astype(np.int16)
    allowed_supports = pair_masks(edges)

    # per qubit and per edge, the distinct images of the options, with the support bits that each
    # option adds to every term
    single: list[list] = []
    for q in range(m):
        seen: dict[bytes, tuple] = {}
        for xs, zs, idx in column_images(x16, z16, q):
            seen.setdefault(xs.tobytes() + zs.tobytes(), (xs, zs, idx))
        single.append(
            [([int(b) << q for b in xs | zs], xs, zs, idx) for xs, zs, idx in seen.values()]
        )
    pairs: dict[int, list] = {}
    for u, v in edges:
        u, v = min(u, v), max(u, v)
        seen = {}
        for xu, zpu, xv, zpv, iu, iv in pair_images(x16, z16, u, v):
            key = xu.tobytes() + zpu.tobytes() + xv.tobytes() + zpv.tobytes()
            if key not in seen:
                vec = [int(a) << u | int(b) << v for a, b in zip(xu | zpu, xv | zpv, strict=True)]
                seen[key] = (vec, xu, zpu, xv, zpv, iu, iv)
        pairs.setdefault(u, []).append((v, list(seen.values())))

    # the qubits that can change a term's image: its own, and their partners on allowed pairs
    partners = [0] * m
    for u, v in edges:
        partners[u] |= 1 << v
        partners[v] |= 1 << u
    reach = []
    for s in support_masks(x, z):
        r = s
        for u in range(m):
            if s >> u & 1:
                r |= partners[u]
        reach.append(r)

    budget = NodeBudget(node_limit)
    big_x, big_z = np.zeros((k, m), np.uint8), np.zeros((k, m), np.uint8)
    layer = [0] * m
    gamma: list[Edge] = []
    masks = [0] * k
    done = [False] * m
    # the cheapest cost so far of every frontier reached
    cheapest: dict[tuple, Key] = {}

    def frontier(q: int) -> tuple[tuple, Key]:
        """The frontier at qubit ``q`` (see above), and the cost so far."""
        assigned = sum(1 << u for u in range(m) if done[u])
        # only the columns of assigned qubits are current; the others hold stale values
        xb = [b & assigned for b in row_masks(big_x)]
        zb = [b & assigned for b in row_masks(big_z)]
        live = 0
        unsettled = []
        on_pair: dict[int, list[int]] = {}
        for t in range(k):
            if reach[t] & ~assigned:
                live |= reach[t]
                unsettled.append(t)
            elif (xb[t] | zb[t]).bit_count() == 2:
                on_pair.setdefault(xb[t] | zb[t], []).append(t)
        reachable = {s: terms for s, terms in on_pair.items() if s & live}
        # a pair that no unsettled term can reach keeps its images: its block is final
        fixed = sum(
            3 if len({(xb[t], zb[t]) for t in terms}) >= 3 else 2
            for s, terms in on_pair.items()
            if s not in reachable
        )
        ones = sum(len(BASIS_GATES[layer[u]]) for u in range(m) if assigned >> u & 1)
        # The images on an assigned qubit never change again, and the rest of the search only
        # compares them for equality (a support, a conflict, a count of distinct images). So
        # relabeling the letters of one assigned qubit is a symmetry of the rest of the search:
        # every qubit's letters are renamed in the order they first occur, term by term.
        relevant = sorted(unsettled + [t for terms in reachable.values() for t in terms])
        rename: list[dict[int, int]] = [{} for _ in range(m)]
        image: dict[int, int] = {}
        for t in relevant:
            code = 0
            for u in range(m):
                letter = (xb[t] >> u & 1) | (zb[t] >> u & 1) << 1
                if letter:
                    names = rename[u]
                    code |= names.setdefault(letter, len(names) + 1) << (2 * u)
            image[t] = code
        state = (
            q,
            assigned,
            tuple((t, image[t]) for t in unsettled),
            frozenset((s, frozenset(image[t] for t in terms)) for s, terms in reachable.items()),
        )
        return state, (2 * len(gamma) + fixed, len(gamma), ones)

    def extend(q: int) -> Iterator[tuple[Key, list[int], list[Edge]]]:
        if budget.exhausted():
            raise TimeoutError
        while q < m and done[q]:
            q += 1
        if not pairs_ok(masks, allowed_supports):
            return
        state, cost = frontier(q)
        if state in cheapest and cheapest[state] <= cost:
            return
        cheapest[state] = cost
        if q == m:
            # every term is settled, so the cost is the solution's key; all solutions share this
            # frontier, so only a cheaper one than every solution before it gets here
            yield cost, list(layer), list(gamma)
            return
        done[q] = True
        saved = list(masks)
        for j, options in pairs.get(q, ()):
            if done[j]:
                continue
            done[j] = True
            for vec, xu, zpu, xv, zpv, iu, iv in options:
                masks[:] = [a | b for a, b in zip(saved, vec, strict=True)]
                layer[q], layer[j] = iu, iv
                big_x[:, q], big_z[:, q], big_x[:, j], big_z[:, j] = xu, zpu, xv, zpv
                gamma.append((q, j))
                yield from extend(q + 1)
                gamma.pop()
            done[j] = False
        for vec, xs, zs, idx in single[q]:
            masks[:] = [a | b for a, b in zip(saved, vec, strict=True)]
            layer[q] = idx
            big_x[:, q], big_z[:, q] = xs, zs
            yield from extend(q + 1)
        masks[:] = saved
        done[q] = False

    yield from extend(0)


def block_cost(big_x: np.ndarray, big_z: np.ndarray) -> int:
    """The two-qubit gates of the blocks, for generic coefficients.

    The images on one pair commute, so they generate a group of at most three nontrivial elements. A
    block holding one or two distinct two-qubit images is a rotation of at most two of the three
    canonical coordinates (two CX gates); three distinct ones need all three (three CX gates).
    Images on a single qubit cost no two-qubit gate.

    Args:
        big_x: the X bits of the images (terms by qubits).
        big_z: their Z bits.
    """
    images: dict[int, set[bytes]] = {}
    for i, s in enumerate(support_masks(big_x, big_z)):
        if s.bit_count() == 2:
            images.setdefault(s, set()).add(big_x[i].tobytes() + big_z[i].tobytes())
    return sum(3 if len(v) >= 3 else 2 for v in images.values())


def _one_layer(
    x: np.ndarray, z: np.ndarray, edges: Sequence[Edge], node_limit: int
) -> tuple[Key, list[int], list[Edge]] | None:
    """The cheapest ``(key, layer, gamma)`` in the enumeration order, or ``None``.

    The enumeration stops at the node budget.
    """
    best = None
    try:
        # every solution yielded is cheaper than the ones before it
        for solution in one_layer_solutions(x, z, edges, node_limit):
            best = solution
    except TimeoutError:
        pass
    return best


def fswap_gates(u: int, v: int) -> list[Gate]:
    """The fermionic swap ``fSWAP(u, v) = SWAP . CZ`` with two CX gates, in circuit order.

    It exchanges two modes that are adjacent in the Jordan-Wigner order and supplies the sign when
    both are occupied, so it moves a mode next to its partner and removes the Z between them. As a
    two-qubit gate it has the Weyl coordinates ``(pi/4, pi/4, 0)`` of iSWAP: two CX gates, where a
    SWAP needs three.
    """
    return [("h", v), ("cx", v, u), ("cx", u, v), ("h", u)]


def on_pairs(big_x: np.ndarray, big_z: np.ndarray, allowed: set[int]) -> bool:
    """Whether every image is on one qubit or one allowed pair, and the distinct pairs disjoint.

    The allowed pairs are bit masks in ``allowed``.

    This is the target of :func:`one_layer_solutions` (:func:`pairs_ok`), checked on finished
    images.
    """
    return pairs_ok(support_masks(big_x, big_z), allowed)


def _fswap_seeds(edges: Sequence[Edge]) -> Iterator[tuple[int, list[Gate]]]:
    """``(number of fSWAPs, gates)`` of the layers to try, cheapest first.

    Every single edge, plainly and then after every pair of single-qubit Cliffords on its qubits
    (with the single-qubit gates that may follow, which the blocks absorb, this covers every gate of
    the fSWAP's Weyl class); then plain matchings of two and more edges.
    """
    for u, v in edges:
        yield 1, fswap_gates(u, v)
    for u, v in edges:
        for bu, bv in itertools.product(range(len(BASIS_GATES)), repeat=2):
            if bu or bv:
                pre = [(g, u) for g in BASIS_GATES[bu]] + [(g, v) for g in BASIS_GATES[bv]]
                yield 1, pre + fswap_gates(u, v)
    for size in range(2, len(edges) + 1):
        for combo in itertools.combinations(edges, size):
            if len({q for e in combo for q in e}) == 2 * size:
                yield size, [g for u, v in combo for g in fswap_gates(u, v)]


def _fswap_layer(
    x: np.ndarray, z: np.ndarray, edges: Sequence[Edge]
) -> tuple[Key, list[Gate]] | None:
    """The cheapest ``(key, gates)`` of a layer of fSWAP moves that puts terms on pairs.

    The layer has to put every term on one qubit or one edge.

    Such a layer is ``CZ(F) L2 CZ(F) L1`` with both CZ layers on the same matching ``F``. It reaches
    what one CZ layer cannot: it brings a mode next to its partner. At most
    ``PAIR_BLOCK_FSWAP_SEEDS`` layers are checked; each check is a support test, no search.
    """
    allowed = pair_masks(edges)
    best = None
    for count, gates in itertools.islice(_fswap_seeds(edges), _budget.PAIR_BLOCK_FSWAP_SEEDS):
        bx, bz = apply_gates(x, z, gates)
        if not on_pairs(bx, bz, allowed):
            continue
        single = sum(1 for g in gates if g[0] != "cx")
        key = (
            4 * count + block_cost(bx, bz),
            2 * count,
            single,
        )  # two CX per fSWAP, in U and in U^dagger
        if best is None or key < best[0]:
            best = (key, gates)
    return best


@lru_cache(maxsize=ONE_LAYER_CACHE)
def _one_layer_gates(
    shape: tuple[int, int], x: bytes, z: bytes, edges: tuple[Edge, ...]
) -> tuple[Gate, ...] | None:
    """The gates of the cheapest one-layer solution of one component (local indices), or ``None``.

    Memoized on the exact input: both pair-block variants need it for every component, and the
    search is the expensive part of both.
    """
    bx = np.frombuffer(x, np.uint8).reshape(shape)
    bz = np.frombuffer(z, np.uint8).reshape(shape)
    found = _one_layer(bx, bz, edges, _budget.PAIR_BLOCK_NODE_LIMIT)
    if found is None:
        return None
    _, layer, gamma = found
    return tuple(layer_gates(layer, gamma))


Solution = tuple[bool, tuple[Gate, ...]]
"""``(whether the component needed fSWAP moves, the gates of its U in its local indices)``."""


def _solve(local: LocalProblem, fswap: bool) -> Solution | None:
    """One component: one CZ layer if possible, otherwise (``fswap``) a layer of fSWAP moves."""
    x, z = np.ascontiguousarray(local.x, np.uint8), np.ascontiguousarray(local.z, np.uint8)
    one = _one_layer_gates(x.shape, x.tobytes(), z.tobytes(), tuple(local.edges))
    if one is not None:
        return False, one
    if not fswap:
        return None
    moves = _fswap_layer(x, z, local.edges)
    return None if moves is None else (True, tuple(moves[1]))


def _transport(
    solution: Solution, rep: Component, copy: Component, local: LocalProblem
) -> Solution | None:
    """The representative's solution moved onto an isomorphic copy, or ``None``.

    It is ``None`` if no isomorphism keeps the solution valid.

    The copies have the same terms letter for letter, so a moved solution gives the same images,
    relabeled; it is valid on the copy only if its gates and blocks land on the copy's own allowed
    pairs, which can differ from the representative's, so every moved solution is checked.
    """
    allowed_edges = {tuple(sorted(e)) for e in local.edges}
    allowed = pair_masks(local.edges)
    moved_fswap, gates = solution

    def maps() -> Iterator[list[int]]:
        # copies with the same local terms (the rows of a lattice, say) try the identity before
        # the isomorphism search runs at all
        identity = list(range(rep.size))
        if rep.termset == copy.termset:
            yield identity
        for pi in isomorphisms(rep, copy, limit=_budget.PAIR_BLOCK_ISOMORPHISMS):
            if pi != identity:
                yield pi

    for pi in maps():
        moved = relabel(gates, pi)
        if any(len(g) == 3 and tuple(sorted(g[1:])) not in allowed_edges for g in moved):
            continue
        if on_pairs(*apply_gates(local.x, local.z, moved), allowed):
            return moved_fswap, tuple(moved)
    return None


def _per_component(
    x: np.ndarray, z: np.ndarray, edges: Sequence[Edge], fswap: bool
) -> list[Gate] | None:
    """Solves the components and joins their gates, or ``None``.

    Components that are copies of each other (isomorphic, letter for letter) share the solution of a
    representative, moved through an isomorphism (see :func:`_transport`); a copy without a valid
    move is solved on its own. With ``fswap`` a component gets a layer of fSWAP moves only if one CZ
    layer has no solution, and the result is ``None`` when no component needed one (that circuit is
    the one-layer variant). Without it, ``None`` if some component has no one-layer solution.
    """
    comps = components_of(x, z)
    if not comps:
        return None
    locals_ = [LocalProblem(c, x, z, edges) for c in comps]
    gates: list[Gate] = []
    needed_fswap = False
    for members in isomorphism_classes(comps):
        rep = members[0]
        found = _solve(locals_[rep], fswap)
        if found is None:
            return None
        for i in members:
            solution = found if i == rep else _transport(found, comps[rep], comps[i], locals_[i])
            if solution is None:
                solution = _solve(locals_[i], fswap)
                if solution is None:
                    return None
            needed_fswap |= solution[0]
            gates += relabel(solution[1], comps[i].qubits)
    if fswap and not needed_fswap:
        return None
    return gates


def pair_block_one_layer(x: np.ndarray, z: np.ndarray, edges: Sequence[Edge]) -> list[Gate] | None:
    """``U = CZ(G) L``, ``G`` a matching of ``edges``, that puts terms on pairs, or ``None``.

    Such a ``U`` puts every term on one qubit or one edge.

    Args:
        x: the X bits of the terms.
        z: the Z bits of the terms.
        edges: the allowed pairs, for the CZ gates and for the blocks.

    Returns:
        The gates of ``U`` in circuit order.
    """
    return _per_component(x, z, edges, fswap=False)


def pair_block_fswap(x: np.ndarray, z: np.ndarray, edges: Sequence[Edge]) -> list[Gate] | None:
    """``U = CZ(F) L2 CZ(F) L1``, a layer of fSWAP moves that puts terms on pairs, or ``None``.

    Such a ``U`` puts every term on one qubit or one edge.

    A component that one CZ layer solves keeps that solution; the fSWAP moves are tried only for the
    others, and only a circuit in which some component needed them is returned.

    Args:
        x: the X bits of the terms.
        z: the Z bits of the terms.
        edges: the allowed pairs, for the CZ gates and for the blocks.

    Returns:
        The gates of ``U`` in circuit order.
    """
    return _per_component(x, z, edges, fswap=True)
