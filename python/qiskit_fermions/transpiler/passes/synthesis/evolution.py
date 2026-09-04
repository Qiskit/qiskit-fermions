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

"""Fermion-operator evolution gate synthesis."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import SparseObservable

from qiskit_fermions.operators import OperatorTrait

from ... import F2QLayout
from ..utils import map_node_single_register

if TYPE_CHECKING:
    from qiskit.synthesis.evolution import EvolutionSynthesis

MapperFunction = Callable[[OperatorTrait, int], SparseObservable]
"""The function signature for :attr:`mapper_fn`."""


def simplify(mapper_fn: MapperFunction) -> MapperFunction:
    """Wraps a mapper function so that it simplifies the operator it returns.

    Simplifying merges duplicate Pauli terms and, as a side effect, sorts the terms into a canonical
    order. That order is worth asking for explicitly in two situations:

    1. The term order an unsimplified mapper produces is **not** guaranteed to be stable between runs,
       because the operators of the Rust core do not preserve the order their terms were added in. A
       product formula synthesizes the terms in the order it receives them, so an unsimplified mapper
       can yield a different (though equally valid) circuit each time. Simplifying pins it.
    2. It merges terms that map to the same Pauli string, which shortens the synthesized circuit when a
       mapper produces such duplicates.

    Use it to wrap any :type:`MapperFunction`::

        MapperFnEvolutionSynthesis(simplify(jordan_wigner))

    Args:
        mapper_fn: the mapper function to wrap.

    Returns:
        A mapper function that simplifies whatever ``mapper_fn`` returns.
    """

    def simplified(operator: OperatorTrait, num_qubits: int) -> SparseObservable:
        return mapper_fn(operator, num_qubits).simplify()

    return simplified


def group_wise(mapper_fn: MapperFunction) -> MapperFunction:
    """Wraps a mapper function so that it maps an operator one group at a time.

    Rather than mapping the operator in one go, the wrapped function maps each of its
    :attr:`~qiskit_fermions.operators.OperatorTrait.groups` separately and sums the results (see
    :ref:`grouping_explanation`). The sum is the same operator either way, but its Pauli terms come out
    grouped: the terms of one group are adjacent instead of interleaved with everybody else's.

    That matters because a product formula synthesizes the terms in the order it receives them. Where
    the terms of a group act on disjoint qubits, having them adjacent lets them be scheduled in
    parallel, which can reduce the two-qubit depth substantially at an unchanged gate count.

    Use it to wrap any :type:`MapperFunction`::

        MapperFnEvolutionSynthesis(group_wise(jordan_wigner))

    An operator without groups is mapped in one go, exactly as ``mapper_fn`` would on its own.

    .. note::
       Composing this with :func:`simplify` defeats the purpose, since simplifying re-sorts the terms
       into a canonical order and so discards the grouping this produces.

    Args:
        mapper_fn: the mapper function to wrap.

    Returns:
        A mapper function that maps ``mapper_fn`` over the operator's groups and sums the results.
    """

    def grouped(operator: OperatorTrait, num_qubits: int) -> SparseObservable:
        if not operator.has_groups():
            return mapper_fn(operator, num_qubits)

        accumulated = SparseObservable.zero(num_qubits)
        for group in operator.split_out_groups():
            accumulated += mapper_fn(group, num_qubits)
        return accumulated

    return grouped


class MapperFnEvolutionSynthesis:
    r"""A :class:`.F2QSynthesisPlugin` for transpiling :class:`.Evolution` under a custom mapping.

    This plugin maps the fermionic Hamiltonian :math:`H` of the incoming :class:`.Evolution` gate to
    a qubit operator using :attr:`mapper_fn` and emits a
    :class:`~qiskit.circuit.library.PauliEvolutionGate`. It thereby preserves the
    :math:`e^{-i t H}` convention of the :class:`.Evolution` gate, with the same evolution time
    :math:`t`.

    How that :class:`~qiskit.circuit.library.PauliEvolutionGate` is subsequently decomposed into
    basis gates is governed by the product formula passed as :attr:`product_formula`. Leaving it at
    its default (``None``) defers to the :class:`~qiskit.circuit.library.PauliEvolutionGate`'s own
    default synthesis (a first-order :class:`~qiskit.synthesis.LieTrotter` decomposition with a
    single repetition). Supplying an explicit
    :class:`~qiskit.synthesis.EvolutionSynthesis` (for example a higher-order
    :class:`~qiskit.synthesis.SuzukiTrotter` or one with several repetitions) selects a different
    Trotter-Suzuki product formula, trading circuit depth for a smaller Trotter error.

    .. note::
       The operator returned by :attr:`mapper_fn` is passed on as-is, in particular **without** being
       simplified. A product formula synthesizes the Pauli terms in the order it receives them, so the
       term order that :attr:`mapper_fn` produces is part of its output and is preserved here. A
       :attr:`mapper_fn` that maps an operator group by group, for instance, emits the terms of each
       group together, which lets the terms of one group be scheduled in parallel where their supports
       are disjoint. See :func:`group_wise`, which wraps any mapper to do exactly that.

    .. caution::
       A consequence of preserving that order is that the synthesized circuit is only as reproducible
       as :attr:`mapper_fn` is. The operators of the Rust core do **not** preserve the order in which
       their terms were added, so a mapper that walks an operator's terms can emit them in a different
       order from one run to the next. The circuits that result are all equally valid (they
       approximate the same evolution of the same operator) but they need not be identical, and
       metrics such as depth or gate count can vary between them.

       Wrap the mapper in :func:`simplify` to pin a canonical order where that matters::

           MapperFnEvolutionSynthesis(simplify(jordan_wigner))
    """

    def __init__(
        self, mapper_fn: MapperFunction, product_formula: EvolutionSynthesis | None = None
    ) -> None:
        """Initializing this transpiler pass plugin can be done with the arguments listed below.

        Args:
            mapper_fn: the fermion-to-qubit operator mapping function.
            product_formula: the product formula with which to synthesize the emitted
                :class:`~qiskit.circuit.library.PauliEvolutionGate`. If ``None`` (the default), the
                gate's own default synthesis is used (a first-order
                :class:`~qiskit.synthesis.LieTrotter` decomposition with a single repetition).
        """
        super().__init__()

        self.mapper_fn: MapperFunction = mapper_fn
        """The fermion-to-qubit operator mapping function.

        The two input arguments should be the following:

        1. the operator to be mapped.
        2. the number of qubits that the resulting operator should be defined on.

        .. note::
           It is the user's responsibility to ensure that this function is in-sync with the global
           transpilation :class:`~qiskit_fermions.transpiler.F2QLayout` setting.
        """

        self.product_formula: EvolutionSynthesis | None = product_formula
        """The product formula used to synthesize the emitted
        :class:`~qiskit.circuit.library.PauliEvolutionGate`, or ``None`` to defer to that gate's own
        default synthesis."""

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
        r"""Runs this transpilation plugin.

        The fermionic Hamiltonian of the incoming :class:`.Evolution` gate is mapped to a qubit
        operator via :attr:`mapper_fn` and appended to ``out_dag`` as a
        :class:`~qiskit.circuit.library.PauliEvolutionGate` implementing :math:`e^{-i t H}` with the
        original evolution time :math:`t` and the :attr:`product_formula` synthesis.

        Args:
            in_node: the input fermion-based circuit instruction. When this plugin gets called, the
                ``in_node.op`` attribute `must` be of type :class:`.Evolution`.
            out_dag: the output qubit-based circuit.
            f2q_layout: the global transpilation :class:`~qiskit_fermions.transpiler.F2QLayout`
                setting.

        .. seealso::
           The documentation of :class:`.F2QSynthesisPlugin` for more detailed explanations of the
           arguments.

        Raises:
            NotImplementedError: when ``in_node`` acts on fermionic modes that are spread across
                multiple :type:`~qiskit_fermions.circuit.FermionicRegister` instances.
        """
        freg_indices, qreg = map_node_single_register(in_node, f2q_layout)
        local_op = in_node.op.operator
        global_op = local_op.relabel_modes(freg_indices)
        # NOTE: the mapped operator is deliberately *not* simplified here. `simplify` reorders the
        # Pauli terms into a canonical order, and a product formula synthesizes them in the order it
        # receives them -- so canonicalizing discards whatever ordering `mapper_fn` chose.
        pauli_op = self.mapper_fn(global_op, len(qreg))
        out_dag.apply_operation_back(
            PauliEvolutionGate(pauli_op, time=in_node.op.params[0], synthesis=self.product_formula),
            qreg,
        )
