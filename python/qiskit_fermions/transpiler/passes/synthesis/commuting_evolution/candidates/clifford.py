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

"""The candidates on one fixed diagonalizing Clifford: ``plain``, ``majority``, ``twin-fold``."""

from __future__ import annotations

from qiskit import QuantumCircuit

from .._diagonal import evolve_from_clifford
from .._elimination import clifford_gates, elimination_variants, to_circuit
from .._symplectic import apply_gates, majority_basis
from ..problem import Context, Problem

MAX_FOLD_ROUNDS = 4
"""The most rounds of twin folding; a fold can create new twins."""


class Plain:
    """GF(2) elimination of the terms with locally chosen pivots, then the diagonal part.

    **Variants:** ``"elimination-order"`` and ``"layered"``, the two orders in which the CX gates of
    the elimination can be emitted (one only, when they coincide).
    """

    name = "plain"

    def generate(self, problem: Problem, context: Context) -> dict[str, QuantumCircuit]:  # noqa: D102
        return elimination_variants(problem, [()] * problem.num_qubits)


class Majority:
    """``plain`` after a per-qubit basis change that sends each qubit's majority letter to Z.

    **Variants:** as :class:`Plain`.
    """

    name = "majority"

    def generate(self, problem: Problem, context: Context) -> dict[str, QuantumCircuit]:  # noqa: D102
        return elimination_variants(problem, majority_basis(problem.x, problem.z))


def twin_fold_clifford(
    problem: Problem, max_rounds: int = MAX_FOLD_ROUNDS
) -> QuantumCircuit | None:
    """The ``plain`` Clifford, then CX layers that fold twin qubits; ``None`` without twins.

    After the plain Clifford every term is a Z-string: a row of a GF(2) matrix ``Z`` (terms by
    qubits). A ``CX(c, t)`` sends column ``c`` to ``c + t`` and fixes ``t``. When two qubits occur
    in exactly the same terms, their columns of ``Z`` are identical, and ``CX(c, t)`` empties column
    ``c``: every term that contained the pair loses one of the two qubits. A matching of such twins
    is one CX layer, and it lowers the weight of every term that contains a pair. The result is
    still a Clifford and the terms stay Z-strings, so it is exact for the same reason ``plain`` is.

    Nothing about the model enters, only the equality of two columns. For the on-site terms
    ``Z_a Z_{a+1} Z_{a+L} Z_{a+L+1}`` of a two-chain block-ordered Hubbard model the twins are the
    rungs ``(a, a+L)`` and every term becomes the nearest-neighbor ``Z_a Z_{a+1}``.

    Args:
        problem: the group of terms.
        max_rounds: the most rounds of folding.

    Returns:
        The Clifford circuit, or ``None`` if no round found a twin.
    """
    n = problem.num_qubits
    gates = clifford_gates(problem.x, problem.z, [()] * n, schedule=False)
    folded = False
    for _ in range(max_rounds):
        _, columns = apply_gates(problem.x, problem.z, gates)  # the Z-strings, terms by qubits
        groups: dict[bytes, list[int]] = {}
        for q in range(n):
            if columns[:, q].any():
                groups.setdefault(columns[:, q].tobytes(), []).append(q)
        # control = the larger index; the smaller one keeps the parity
        pairs = [(qs[i + 1], qs[i]) for qs in groups.values() for i in range(0, len(qs) - 1, 2)]
        if not pairs:
            break
        gates += [("cx", control, target) for control, target in pairs]
        folded = True
    return to_circuit(gates, n) if folded else None


class TwinFold:
    """``plain`` followed by CX layers that fold qubits occurring in exactly the same terms.

    It applies only to groups that have such twins (see :func:`twin_fold_clifford`), and offers
    nothing otherwise.

    **Variants:** ``"forward"`` and ``"reversed"``, the two orders of the layers of weight-two
    rotations. They commute, so both are exact; which one routes better depends on how the layers
    meet the gates of the Clifford.
    """

    name = "twin-fold"

    def generate(self, problem: Problem, context: Context) -> dict[str, QuantumCircuit]:  # noqa: D102
        clifford = twin_fold_clifford(problem)
        if clifford is None:
            return {}
        return {
            name: evolve_from_clifford(problem, clifford, reverse_two_body=reverse)
            for name, reverse in (("forward", False), ("reversed", True))
        }
