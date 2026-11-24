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

"""Packing of commuting operations into layers of disjoint qubits."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Literal, TypeVar

import rustworkx as rx


def sorted_pair(pair: tuple[int, int]) -> tuple[int, int]:
    """The two qubits of ``pair`` as an ordered pair (smaller index first)."""
    a, b = pair
    return (a, b) if a <= b else (b, a)


def pairwise_tree(
    qubits: Sequence[int], keep: Literal["control", "target"]
) -> tuple[int, list[tuple[int, int]]]:
    """A balanced binary tree of CX gates over ``qubits``, of depth ``ceil(log2(len(qubits)))``.

    Level by level, neighbors in the list are paired by a ``CX(first, second)``, and one qubit of
    every pair goes on to the next level: the target, to collect the parity of Z-strings onto it, or
    the control, to clear the X bits of the others. An odd qubit out goes on unpaired.

    Args:
        qubits: the qubits, at least one.
        keep: which qubit of a pair goes on to the next level.

    Returns:
        The qubit that is left at the root, and the CX gates as ``(control, target)``, in order.
    """
    gates: list[tuple[int, int]] = []
    level = list(qubits)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            gates.append((level[i], level[i + 1]))
            nxt.append(level[i + 1] if keep == "target" else level[i])
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0], gates


def edge_color_layers(edges: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Partitions edges into layers of vertex-disjoint pairs (greedy edge coloring).

    Args:
        edges: pairs of qubits.

    Returns:
        The layers, each a list of pairs that share no qubit. Empty if there are no edges.
    """
    if not edges:
        return []
    graph = rx.PyGraph(multigraph=False)
    # adds the nodes it needs, and a repeated pair stays one edge
    graph.extend_from_edge_list(list(edges))
    colors = rx.graph_greedy_edge_color(graph)
    layers: dict[int, list[tuple[int, int]]] = {}
    for edge_idx, (a, b) in enumerate(graph.edge_list()):
        layers.setdefault(colors[edge_idx], []).append((a, b))
    return [layers[c] for c in sorted(layers)]


T = TypeVar("T")


def first_fit_layers(
    items: Iterable[T], key: Callable[[T], Collection[int]] | None = None
) -> list[list[T]]:
    """Packs items into layers of pairwise disjoint supports (qubit sets), first fit.

    Args:
        items: the items, in the order they are placed.
        key: the support of an item; ``None`` if the items are their supports.

    Returns:
        The layers; every item goes into the first layer its support does not overlap.
    """
    layers: list[list[T]] = []
    used: list[set[int]] = []
    for item in items:
        s: Collection[int] = item if key is None else key(item)  # type: ignore[assignment]
        for layer, taken in zip(layers, used, strict=True):
            if taken.isdisjoint(s):
                layer.append(item)
                taken.update(s)
                break
        else:
            layers.append([item])
            used.append(set(s))
    return layers
