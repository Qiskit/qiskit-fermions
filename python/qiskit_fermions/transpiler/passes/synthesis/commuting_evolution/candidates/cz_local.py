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

"""The ``cz-local`` candidate: CZ layers on device edges between single-qubit layers.

Its diagonal variant is ``U = L2 CZ(Gamma) L1``. For fixed per-qubit layers the graph ``Gamma`` and
the final letters are the solution of a linear system over GF(2); the per-qubit layer ``L1`` is
found by coordinate descent. This is the hardware-tailored diagonalization of Miller et al.,
arXiv:2203.03646.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit

from .._components import LocalProblem, components_of
from .._diagonal import NotDiagonalizedError, evolve_from_clifford
from .._elimination import to_circuit
from .._gf2 import cooccurring_edges, incidence, solve_gamma_joint
from .._layers import edge_color_layers
from .._symplectic import Gate, apply_cz, apply_layer, coordinate_descent, layer_gates, relabel
from ..problem import Context, Problem

MAX_SWEEPS = 3
"""The most passes of the coordinate descent over the single-qubit layer."""

COMPONENT_CACHE = 256
"""The most components whose diagonal solution is remembered."""

INFEASIBLE = (1,)
"""The score of every infeasible layer; feasible ones start with ``0`` and so rank before it."""


def cz_local_gates(
    x: np.ndarray,
    z: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> list[Gate] | None:
    """The best CZ-then-local diagonalizer found, as a gate list, or ``None``.

    The connected components of the group act on disjoint qubits, so their linear systems are
    independent: each one is solved on its own by :func:`descend`, with the edges among its qubits,
    and the solutions are joined. A component that a single change of its own single-qubit layer
    makes feasible therefore does not wait for the others, which the descent over the whole group
    could not see (an infeasible layer has no measure of how far from feasible it is).

    When a component has no solution of its own, neither has the group. A CZ between two components
    only adds a Z where a term of one has no X on a qubit of the other, so the edges between
    components add conditions to the linear system and never relax one: a solution for the group
    holds a solution for every component, with the same single-qubit layer. And a descent over the
    whole group can only become feasible on a layer that differs from the start in at most one
    qubit (all infeasible layers score the same), which the descent of that qubit's component tries
    as well. So no search over the whole group is needed.

    Components that are exact copies of each other, in their local indices and edges, are solved
    once (see :func:`_component_gates`).

    Args:
        x: the X bits of the terms.
        z: the Z bits of the terms.
        edges: the allowed CZ pairs, sorted pairs of qubit indices.

    Returns:
        The gates in circuit order, or ``None`` if no feasible diagonalizer was found.
    """
    gates: list[Gate] = []
    for comp in components_of(x, z):
        local = LocalProblem(comp, x, z, edges)
        lx = np.ascontiguousarray(local.x, np.uint8)
        lz = np.ascontiguousarray(local.z, np.uint8)
        found = _component_gates(lx.shape, lx.tobytes(), lz.tobytes(), tuple(local.edges))
        if found is None:
            return None
        gates += relabel(found, comp.qubits)
    return gates


@lru_cache(maxsize=COMPONENT_CACHE)
def _component_gates(
    shape: tuple[int, int], x: bytes, z: bytes, edges: tuple[tuple[int, int], ...]
) -> tuple[Gate, ...] | None:
    """:func:`descend` on one component (local indices), memoized on the exact input.

    The components of a lattice group are often copies of each other (its rows, say), and the
    descent is the expensive part of the candidate.
    """
    bx = np.frombuffer(x, np.uint8).reshape(shape)
    bz = np.frombuffer(z, np.uint8).reshape(shape)
    found = descend(bx, bz, list(edges))
    return None if found is None else tuple(found)


def _weight_lower_bound(x: np.ndarray, z: np.ndarray) -> int:
    """The smallest largest weight that the images of the terms can have under any Clifford.

    A Clifford sends distinct terms to distinct Paulis, and the images here are Z-strings, of which
    there are only ``comb(n, w)`` of weight ``w`` on ``n`` qubits. So the distinct non-identity
    terms need images up to the weight at which the counts of the lighter ones add up to theirs.
    """
    rows = {r.tobytes() for r in np.concatenate([x, z], axis=1).astype(np.uint8) if r.any()}
    n = x.shape[1]
    weight = available = 0
    while available < len(rows):
        weight += 1
        available += math.comb(n, weight)
    return weight


def descend(
    x: np.ndarray,
    z: np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    filter_edges: bool = True,
) -> list[Gate] | None:
    """The best CZ-then-local diagonalizer found by coordinate descent, as a gate list, or ``None``.

    The score is (feasible, largest image weight, CZ depth, number of CZ gates). The image weight is
    the weight of the Z-string each term ends up as; it sets the cost of the rotation stage, so it
    is ranked before the CZ count. With ``filter_edges`` only the edges between qubits that share a
    term are used (a large saving when many edges are allowed, e.g. all-to-all); if that finds
    nothing, the full edge set is tried.

    Args:
        x: the X bits of the terms.
        z: the Z bits of the terms.
        edges: the allowed CZ pairs, sorted pairs of qubit indices.
        filter_edges: restrict to the edges whose qubits share a term first.

    Returns:
        The gates in circuit order, or ``None`` if no feasible diagonalizer was found.
    """
    if not x.any() and int(z.sum(axis=1).max(initial=0)) <= _weight_lower_bound(x, z):
        # Every term is already a Z-string, as light as distinct images can be: no layer can lower
        # the largest image weight, and no CZ count beats zero, so the empty Clifford is optimal.
        return []
    if filter_edges:
        kept = cooccurring_edges(x, z, edges)
        if len(kept) < len(edges):
            restricted = descend(x, z, kept, filter_edges=False)
            if restricted is not None:
                return restricted
    n = x.shape[1]
    nodes = sorted({q for e in edges for q in e})
    inc = incidence(n, edges)

    def score(l1: list[int], incumbent: tuple | None) -> tuple:
        """Scores a single-qubit layer.

        Returns the sort key plus the solution. All infeasible layers tie: an inconsistent linear
        system raises the rank by at most one, so it gives no measure of how many equations fail.
        The descent therefore only moves to a layer that is feasible or better; a layer that needs
        several qubits changed at once is out of its reach.
        """
        x1, z1 = apply_layer(x, z, l1)
        if (
            incumbent is not None
            and incumbent[0][0] == 0
            and int(x1.sum(axis=1).max()) > incumbent[0][1]
        ):
            # exact bound: the image of a term contains its X support, so this layer cannot beat a
            # feasible incumbent whose image weight is smaller
            return INFEASIBLE, None, None
        res = solve_gamma_joint(x1, z1, edges, inc)
        if res is None:
            return INFEASIBLE, None, None
        gamma, letters = res
        for u, v in gamma:
            apply_cz(x1, z1, u, v)
        weight = int((x1 | z1).sum(axis=1).max())
        return (0, weight, len(edge_color_layers(sorted(gamma))), len(gamma)), letters, gamma

    l1, (_, letters, gamma) = coordinate_descent(n, nodes, score, MAX_SWEEPS)
    if letters is None:
        return None
    return layer_gates(l1, gamma, letters)


class CzLocal:
    """CZ layers on the allowed pairs between single-qubit layers.

    It needs the allowed CZ pairs (:attr:`.Context.edges`): the device's, once the gate's qubits are
    placed, or the user's. Without them it offers nothing. Every CZ gate is on an allowed pair.

    **Variants:**

    * ``"diagonal"``: ``U = L2 CZ(G) L1`` sends every term to a Z-string, and the middle is a
      diagonal rotation stage. ``G`` and the letters come out of a linear system over GF(2); there
      is no variant when it has no solution.
    """

    name = "cz-local"

    def generate(self, problem: Problem, context: Context) -> dict[str, QuantumCircuit]:  # noqa: D102
        if not context.edges:
            return {}
        variants: dict[str, QuantumCircuit] = {}
        gates = cz_local_gates(problem.x, problem.z, context.edges)
        if gates is not None:
            with contextlib.suppress(NotDiagonalizedError):
                variants["diagonal"] = evolve_from_clifford(
                    problem, to_circuit(gates, problem.num_qubits)
                )
        return variants
