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

"""Deterministic limits for the searches of the layout candidates."""

from __future__ import annotations

PAIR_BLOCK_NODE_LIMIT = 2**16
"""Nodes per component for the one-layer pair-block search."""

PAIR_BLOCK_ISOMORPHISMS = 64
"""The most isomorphisms tried to move a component's pair-block solution onto a copy."""

PAIR_BLOCK_FSWAP_SEEDS = 2_000
"""The most layers of fSWAP moves checked per component by the fSWAP pair-block search. All single
edges with every pair of single-qubit Cliffords before them come first (36 per edge), so this covers
them up to about 50 edges."""


class NodeBudget:
    """A budget of visited search nodes.

    Unlike a wall-clock limit, a node budget gives the same result on every machine and under every
    load: a search that runs out returns the same prefix of its (deterministic) enumeration every
    time.

    Args:
        limit: the number of nodes after which the search is exhausted, or ``None`` for no limit.
    """

    __slots__ = ("limit", "nodes")

    def __init__(self, limit: int | None):
        self.limit = limit
        self.nodes = 0

    def exhausted(self) -> bool:
        """Counts one node and returns whether more than ``limit`` nodes were visited."""
        self.nodes += 1
        return self.limit is not None and self.nodes > self.limit
