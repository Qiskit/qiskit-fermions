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

"""Evolution synthesis tests."""

from __future__ import annotations

from qiskit.circuit.library import PauliEvolutionGate
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler.passes import EvolutionSynthesis, F2QSynthesis, TrivialF2QLayout
from qiskit_fermions.transpiler.passmanager import (
    FermionicPassManager,
    FermionicStagedPassManager,
    FermionicToQubitConverter,
)


def test_evolution_gate_synthesis():
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 2)): 2.0,
            ((True, 2), (False, 0)): 2.0,
            ((True, 1), (False, 3)): -2.0,
            ((True, 3), (False, 1)): -2.0,
        }
    )
    time = 1.5
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

    synth = F2QSynthesis()
    synth.plugins[Evolution] = EvolutionSynthesis(jordan_wigner)

    pm = FermionicStagedPassManager()
    pm.layout = FermionicPassManager(TrivialF2QLayout())
    pm.synthesis = FermionicToQubitConverter(synth)

    qu_circ = pm.run(circ)

    gates = qu_circ.data
    assert len(gates) == 1
    assert isinstance(gates[0].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 16


def test_custom_qubit_ordering():
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 2)): 2.0,
            ((True, 2), (False, 0)): 2.0,
            ((True, 1), (False, 3)): -2.0,
            ((True, 3), (False, 1)): -2.0,
        }
    )
    time = 1.5
    num_modes = 4
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

    custom_qubit_ordering = [0, 2, 1, 3]

    def custom_qubit_ordering_mapper_fn(op, num_qubits):
        relabeled = op.relabel_modes(custom_qubit_ordering)
        return jordan_wigner(relabeled, num_qubits)

    synth = F2QSynthesis()
    synth.plugins[Evolution] = EvolutionSynthesis(custom_qubit_ordering_mapper_fn)

    pm = FermionicStagedPassManager()
    pm.layout = FermionicPassManager(TrivialF2QLayout())
    pm.synthesis = FermionicToQubitConverter(synth)

    qu_circ = pm.run(circ)

    gates = qu_circ.data
    assert len(gates) == 1
    assert isinstance(gates[0].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 4
