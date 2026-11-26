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

"""The interface of a candidate synthesis."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, TypeVar

from qiskit import QuantumCircuit

from ..problem import Context, Problem

S = TypeVar("S", bound=str)


class Candidate(Protocol):
    """One way of building ``exp(-i t H)`` for a group of mutually commuting Paulis.

    A candidate offers circuits; choosing between them, and between the candidates, is the
    selector's job. So a candidate never ranks its own output: when it can build the evolution in
    several ways (an order of commuting gates, parity trees or parity ladders), it returns each of
    them as a variant. A candidate whose variants are costly to build may let the user choose which
    of them it builds, with a ``variants`` argument. It may run any search or heuristic of its own
    on the way to a circuit.

    A candidate must be exact (every circuit it returns implements the evolution, including the
    global phase) and deterministic, and should keep no state between gates.
    """

    name: str
    """The candidate's name. It labels its variants, and must be unique among the plugin's
    candidates."""

    def generate(self, problem: Problem, context: Context) -> Mapping[str, QuantumCircuit]:
        """The candidate's circuits for this problem, by variant name.

        The mapping is empty when the candidate has nothing to offer.

        The order of the mapping is the order in which ties between the variants are broken. A
        candidate with a single circuit may name its variant ``""``.
        """
        ...


def validated_variants(variants: Sequence[S], known: Collection[S]) -> tuple[S, ...]:
    """The variants a candidate was asked to build, checked against the ones it knows.

    Raises:
        ValueError: if ``variants`` is empty or names an unknown variant.
    """
    unknown = [v for v in variants if v not in known]
    if unknown or not variants:
        raise ValueError(
            f"variants must be a non-empty subset of {list(known)}, got {list(variants)}"
        )
    return tuple(variants)
