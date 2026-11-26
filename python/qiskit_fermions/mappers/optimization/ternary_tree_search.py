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

"""Searching for a good ternary-tree encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit_fermions._lib.mappers.mappers_optimization.ternary_tree_search import (
    hatt_tree as _hatt_tree,
)
from qiskit_fermions.mappers.library import majorana_operator
from qiskit_fermions.operators import MajoranaOperator

if TYPE_CHECKING:
    from qiskit_fermions.mappers import TernaryTree
    from qiskit_fermions.protocols import SupportsMajoranaOperator

__all__ = ["hatt_tree"]


def hatt_tree(
    hamiltonian: MajoranaOperator | SupportsMajoranaOperator,
    num_modes: int | None = None,
    *,
    weighted: bool = False,
) -> TernaryTree:
    """Grow a ternary tree adapted to a Hamiltonian, following the HATT algorithm.

    The algorithm is topology-blind: it shapes the tree around the Hamiltonian's term structure.
    It works on the Majorana representation, so any other operator is first converted with
    :func:`.majorana_operator`.

    The tree is grown bottom-up, one node per step, following the HATT algorithm [1]_. Each step
    chooses the subtrees for a new node's ``X`` and ``Z`` links so that the Hamiltonian's Pauli
    weight on that node's qubit is as small as possible. The ``Y`` link is not chosen freely: it gets
    the subtree holding the partner of the Majorana operator that the ``X`` side contributes, so the
    two Majorana operators of a mode always meet on one qubit as ``X`` and ``Y``. This is the
    vacuum-preserving variant of the algorithm, and its bookkeeping matches how a
    :class:`.TernaryTreeEncoding` pairs legs into Majorana operators. Node ``j`` of the returned tree
    hosts mode ``j``.

    The search is greedy: it minimizes each step's cost in turn, not their sum. Summed over all
    steps, these costs are the total Pauli weight of the Hamiltonian's Majorana terms as given, each
    term's image counted on its own. Nothing is normal-ordered, merged, or dropped first, so terms
    that would cancel, and terms with negligible coefficients, all count. To target the Pauli weight
    of the simplified mapped operator instead, normal-order and simplify the Majorana form yourself
    before passing it in: its distinct terms then map to distinct Pauli strings, and the summed cost
    equals the total Pauli weight of the mapped operator after the same simplification.

    Args:
        hamiltonian: the operator to adapt to, either a :class:`.MajoranaOperator` or any operator
            implementing :class:`.SupportsMajoranaOperator`.
        num_modes: the number of fermionic modes, which is also the number of qubits. Inferred from
            the operator's support when omitted, which is what you want unless the encoding should
            cover modes the Hamiltonian does not act on.
        weighted: whether to weight each term by ``|coefficient|`` when scoring a candidate merge.
            The default, ``False``, counts terms and ignores their coefficients, as in the HATT
            publication.

    Returns:
        The tree.

    Raises:
        ValueError: if ``num_modes`` is not given and the operator acts on no mode, for example
            because it is empty or a constant.
        ValueError: if ``num_modes`` is zero, or more than a ternary tree can hold.
        ValueError: if the Hamiltonian acts outside ``num_modes``.
        ValueError: if a coefficient is infinite or NaN.
        OverflowError: if ``num_modes`` is negative or does not fit in 32 bits.

    .. doctest::

        >>> from qiskit_fermions.mappers.library import majorana_operator
        >>> from qiskit_fermions.mappers.optimization import hatt_tree
        >>> from qiskit_fermions.operators import FermionOperator
        >>>
        >>> hamiltonian = FermionOperator.from_dict(
        ...     {((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 1.0}
        ... )
        >>> # The conversion keeps all eight Majorana terms; four of them cancel.
        >>> majorana = majorana_operator(hamiltonian)
        >>> len(majorana), len(majorana.normal_ordered().simplify(1e-10))
        (8, 2)
        >>> tree = hatt_tree(majorana.normal_ordered().simplify(1e-10))
        >>> tree.num_nodes, tree.num_legs
        (2, 5)

    Note:
        This is a greedy heuristic, and Pauli weight is not the circuit cost, so it does not always
        beat Jordan-Wigner. Against Jordan-Wigner, reference 1 reports on average about 15% lower
        Pauli weight and 26% fewer CNOT gates on molecular Hamiltonians, but also cases where
        Jordan-Wigner gives fewer CNOT gates or a shallower circuit. Measure on your own Hamiltonian
        rather than assuming an improvement.

    Note:
        Reference 1 does not say how to break ties between equal-cost candidates. Ties go to the
        lowest subtree indices (legs first, in Majorana order, then nodes in the order they were
        created), so the result is deterministic. Because legs are numbered by mode, relabeling the
        modes of the Hamiltonian changes which candidate wins a tie.

    .. [1] Y. Liu, K. Yao, J. Hong, J. Froustey, E. Rrapaj, C. Iancu, G. Li, and Y. Shi, HATT:
           Hamiltonian Adaptive Ternary Tree for Optimizing Fermion-to-Qubit Mapping (2024),
           `arXiv:2409.02010 <https://arxiv.org/abs/2409.02010>`_.
    """
    # A `MajoranaOperator` does not implement the protocol itself, so it is passed through as is.
    if not isinstance(hamiltonian, MajoranaOperator):
        hamiltonian = majorana_operator(hamiltonian)

    if num_modes is None:
        num_modes = hamiltonian.min_num_modes()
        if num_modes == 0:
            raise ValueError(
                "num_modes must be given when the Hamiltonian acts on no mode (for example, when it "
                "is empty or a constant), since the mode count cannot be inferred"
            )

    return _hatt_tree(hamiltonian, num_modes, weighted=weighted)
