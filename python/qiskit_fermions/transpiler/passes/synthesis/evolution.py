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

from qiskit_fermions.operators.protocol import OperatorTrait

from ... import F2QLayout
from ..utils import map_node_single_register

if TYPE_CHECKING:
    from qiskit.synthesis.evolution import EvolutionSynthesis

MapperFunction = Callable[[OperatorTrait, int], SparseObservable]
"""The function signature for :attr:`mapper_fn`."""


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
    :class:`~qiskit.synthesis.EvolutionSynthesis` -- for example a higher-order
    :class:`~qiskit.synthesis.SuzukiTrotter` or one with several repetitions -- selects a different
    Trotter-Suzuki product formula, trading circuit depth for a smaller Trotter error.
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
        operator via :attr:`mapper_fn`, simplified, and appended to ``out_dag`` as a
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
        pauli_op = self.mapper_fn(global_op, len(qreg)).simplify()
        out_dag.apply_operation_back(
            PauliEvolutionGate(pauli_op, time=in_node.op.params[0], synthesis=self.product_formula),
            qreg,
        )
