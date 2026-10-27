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

"""Backwards-compatibility shims for the former Givens fermion-to-qubit plugins.

An orbital rotation and a Slater determinant preparation used to be mapped to qubits by a plugin of
:class:`.F2QSynthesis` that hard-wired Jordan-Wigner. They are now broken down in fermionic space by
:class:`.F2FSynthesis`. The plugins of this module keep the old entry points working: each runs the
modern fermionic decomposition itself, and maps the factors under Jordan-Wigner.

This module is self-contained apart from the exports and the entry points: removing the shims means
deleting this module, those two, their test and their release note.
"""

from __future__ import annotations

import warnings

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.utils import deprecate_func

from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, InitializeModes
from qiskit_fermions.circuit.library.synthesis import (
    FermionicGivensDecomposition,
    FermionicSlaterGivensDecomposition,
)
from qiskit_fermions.mappers.library import jordan_wigner

from ... import F2QLayout
from ..utils import map_node_single_register
from .evolution import MapperFnEvolutionSynthesis
from .hop_evolution import match_hop
from .initialize_modes import TrivialOccupationInitializeModesSynthesis


def _map_factors(
    circuit: FermionicCircuit, in_node: DAGOpNode, out_dag: DAGCircuit, f2q_layout: F2QLayout
) -> None:
    """Maps the gates of a fermionic decomposition of ``in_node`` to qubits under Jordan-Wigner.

    A hop between adjacent modes becomes a single :class:`~qiskit.circuit.library.XXPlusYYGate`, as
    the former plugins emitted, rather than a Pauli evolution that relies on the qubit stage to
    recognize it.
    """
    initialize = TrivialOccupationInitializeModesSynthesis()
    evolve = MapperFnEvolutionSynthesis(jordan_wigner)

    for instruction in circuit._inner.data:
        gate = instruction.operation
        qargs = tuple(in_node.qargs[circuit._inner.find_bit(q).index] for q in instruction.qubits)
        node = DAGOpNode(gate, qargs=qargs, cargs=())

        if isinstance(gate, InitializeModes):
            initialize.run(node, out_dag, f2q_layout=f2q_layout)
        elif isinstance(gate, Evolution):
            freg_indices, qreg = map_node_single_register(node, f2q_layout)
            observable = jordan_wigner(gate.operator.relabel_modes(freg_indices), len(qreg))
            matched = match_hop(observable, float(gate.params[0]))
            if matched is None:
                evolve.run(node, out_dag, f2q_layout=f2q_layout)
            else:
                hop, (first, second) = matched
                out_dag.apply_operation_back(hop, (qreg[first], qreg[second]))
        else:
            raise TypeError(f"Cannot map a gate of type {type(gate).__name__} to qubits.")


class GivensDecompositionOrbitalRotationSynthesis:
    """The former plugin for transpiling an :class:`.OrbitalRotation` into Givens rotations.

    This is the :class:`.FermionicGivensDecomposition` followed by the Jordan-Wigner mapping of its
    factors.
    """

    @deprecate_func(
        since="0.4.0",
        package_name="qiskit-fermions",
        additional_msg=(
            "Break the gate down with F2FSynthesis({'OrbitalRotation': 'Givens'}) in the "
            "optimization stage instead, which works with any encoding."
        ),
    )
    def __init__(self) -> None:
        """Initializing this plugin takes no arguments."""

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
        """Runs this transpilation plugin.

        Args:
            in_node: the input fermion-based circuit instruction, an :class:`.OrbitalRotation`.
            out_dag: the output qubit-based circuit.
            f2q_layout: the global transpilation :class:`~qiskit_fermions.transpiler.F2QLayout`
                setting.
        """
        circuit = FermionicGivensDecomposition().synthesize(in_node.op)
        _map_factors(circuit, in_node, out_dag, f2q_layout)


class GivensDecompositionSlaterDeterminantSynthesis:
    """The former plugin for transpiling a :class:`.PrepareSlaterDeterminant` into Givens rotations.

    This is the :class:`.FermionicSlaterGivensDecomposition` followed by the Jordan-Wigner mapping
    of its gates.
    """

    @deprecate_func(
        since="0.4.0",
        package_name="qiskit-fermions",
        additional_msg=(
            "Break the gate down with F2FSynthesis({'PrepareSlaterDeterminant': 'Givens'}) in the "
            "optimization stage instead, which works with any encoding."
        ),
    )
    def __init__(self, *, minimize_2q_gate_count: bool = False) -> None:
        """Initializing this plugin can be done with the argument below.

        Args:
            minimize_2q_gate_count: no longer supported. The peeling of unit-orbitals that this
                enabled was removed, so ``True`` has no effect and warns.
        """
        if minimize_2q_gate_count:
            warnings.warn(
                "minimize_2q_gate_count=True is no longer supported and is ignored: the peeling of "
                "unit-orbitals it enabled was removed.",
                category=UserWarning,
                stacklevel=3,
            )
        self.minimize_2q_gate_count = False
        """Always ``False``, see ``__init__``."""

    def run(self, in_node: DAGOpNode, out_dag: DAGCircuit, *, f2q_layout: F2QLayout) -> None:
        """Runs this transpilation plugin.

        Args:
            in_node: the input fermion-based circuit instruction, a
                :class:`.PrepareSlaterDeterminant`.
            out_dag: the output qubit-based circuit.
            f2q_layout: the global transpilation :class:`~qiskit_fermions.transpiler.F2QLayout`
                setting.
        """
        circuit = FermionicSlaterGivensDecomposition().synthesize(in_node.op)
        _map_factors(circuit, in_node, out_dag, f2q_layout)
