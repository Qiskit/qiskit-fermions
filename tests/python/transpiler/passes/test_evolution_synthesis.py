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

"""Fermion-to-qubit synthesis tests."""

from __future__ import annotations

from functools import partial

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import PassManager
from qiskit_fermions.circuit import FermionCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler.passes.layout import TrivialF2QLayout
from qiskit_fermions.transpiler.passes.synthesis import EvolutionSynthesis, F2QSynthesis


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
    num_fermions = 4
    circ = FermionCircuit(num_fermions)
    evo = Evolution(num_fermions, hamil, time=time)
    circ.append(evo, circ.fermions)

    mapper_fn = partial(jordan_wigner, num_qubits=num_fermions)

    synth = F2QSynthesis()
    synth.plugins[Evolution] = EvolutionSynthesis(mapper_fn)

    pm = PassManager([TrivialF2QLayout(), synth])

    # TODO: update API of FermionCircuit to not require access to `_inner` QuantumCircuit here
    qu_circ = pm.run(circ._inner)

    gates = qu_circ.data
    assert len(gates) == 1
    assert isinstance(gates[0].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 16


def test_custom_layout():
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 2)): 2.0,
            ((True, 2), (False, 0)): 2.0,
            ((True, 1), (False, 3)): -2.0,
            ((True, 3), (False, 1)): -2.0,
        }
    )
    time = 1.5
    num_fermions = 4
    circ = FermionCircuit(num_fermions)
    evo = Evolution(num_fermions, hamil, time=time)
    circ.append(evo, circ.fermions)

    custom_layout = [0, 2, 1, 3]

    def custom_layout_mapper_fn(op):
        relabeled = op.relabel_modes(custom_layout)
        return jordan_wigner(relabeled, num_fermions)

    synth = F2QSynthesis()
    synth.plugins[Evolution] = EvolutionSynthesis(custom_layout_mapper_fn)

    pm = PassManager([TrivialF2QLayout(), synth])

    # TODO: update API of FermionCircuit to not require access to `_inner` QuantumCircuit here
    qu_circ = pm.run(circ._inner)

    gates = qu_circ.data
    assert len(gates) == 1
    assert isinstance(gates[0].operation, PauliEvolutionGate)

    qu_circ_decomp = qu_circ.decompose(reps=2)
    assert qu_circ_decomp.depth(lambda instr: len(instr.qubits) == 2) == 4
