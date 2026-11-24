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

"""The search candidate: coordinate descent over per-qubit bases before the elimination."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit

from .._elimination import elimination_variants, search_basis
from .._symplectic import BASIS_GATES
from ..problem import Context, Problem
from .base import validated_variants

_SHAPES: dict[str, tuple[bool, bool]] = {"star": (False, False), "tree-colored": (True, True)}
"""The elimination shapes, as ``(tree, colored)``: a star fan-in in natural row order, or a balanced
fan-in tree with the rows eliminated color class by color class."""

_ORDERS = ("layered", "elimination-order")
"""The CX orders, in tie-break order: the descent ranks the bases with the layered order."""

VARIANTS: tuple[str, ...] = tuple(f"{shape}-{order}" for shape in _SHAPES for order in _ORDERS)
"""The variants of :class:`Search`, in tie-break order."""


class Search:
    """The ``plain`` elimination after per-qubit bases found by a coordinate descent.

    The descent minimizes the two-qubit gate count of the Clifford, then its two-qubit depth.

    It is the most expensive candidate, and it pays off only for a gate that is not placed on the
    device, which is the case when the high-level synthesis runs in the ``init`` stage: there it
    often finds a shallower circuit than every other candidate, on structured groups too. Once the
    gate is placed (:attr:`.Context.placed`), the candidates that use the device's connectivity do
    better, so it offers nothing.

    **Variants:** ``"<shape>-<order>"`` for each elimination shape and each CX order (see
    :class:`.Plain`), in the order of :data:`VARIANTS`. The shape is ``"star"``, a star fan-in in
    natural row order, or ``"tree-colored"``, a balanced fan-in tree with the rows eliminated color
    class by color class. The descent ranks the bases with the layered order, which therefore comes
    first for each shape. When the two orders of a shape give the same circuit, only the first one
    requested is offered.

    Args:
        variants: the variants to build. Every shape among them costs one descent, shared by its two
            orders.

    Raises:
        ValueError: if ``variants`` is empty or names an unknown variant.
    """

    name = "search"

    def __init__(self, *, variants: Sequence[str] = VARIANTS) -> None:  # noqa: D107
        self.variants = validated_variants(variants, VARIANTS)

    def generate(self, problem: Problem, context: Context) -> dict[str, QuantumCircuit]:  # noqa: D102
        if context.placed:
            return {}
        out: dict[str, QuantumCircuit] = {}
        for shape, (tree, colored) in _SHAPES.items():
            orders = [order for order in _ORDERS if f"{shape}-{order}" in self.variants]
            if not orders:
                continue
            choice = search_basis(problem.x, problem.z, tree=tree, colored=colored)
            basis = [BASIS_GATES[b] for b in choice]
            circuits = elimination_variants(
                problem, basis, tree=tree, colored=colored, orders=orders
            )
            out.update({f"{shape}-{order}": circuit for order, circuit in circuits.items()})
        return out
